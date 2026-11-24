"""Figure scripts for PLAN §9.7 (1, 2, 3, 4, 6, 7, 8), all read from ``results/bench.csv``.

Every point is a mean over *scenes* (rows of one scene/method/condition are averaged first),
with a 95% scene-level bootstrap CI as the error bar. Style comes from ``countonce.plotting``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bench.runner import read_rows
from bench.stats import bootstrap_ci, paired_bootstrap
from countonce.plotting import apply_style, color_for, save

TIER_ORDER = ["easy", "medium", "hard"]


@dataclass(frozen=True)
class Point:
    mean: float
    lo: float
    hi: float
    n_scenes: int


@dataclass(frozen=True)
class Row:
    scene: str
    method: str
    n_views: int
    sample: str
    ordered: bool
    pred: float
    gt: float
    low: float | None
    high: float | None
    tier: str
    # Appended CSV columns; blank in old files, so every one is optional.
    run_key: str = ""
    runtime_s: float | None = None
    peak_vram_mb: float | None = None
    timings: dict[str, float] | None = None
    assoc_f1: float | None = None
    run_dir: str = ""


def _f(s: str) -> float | None:
    return float(s) if s != "" else None


def load_rows(path: str | Path) -> list[Row]:
    """Rows with status ok and a ground truth; the tier is read from ``config_json``."""
    out: list[Row] = []
    for r in read_rows(path):
        pred, gt = _f(r["pred"]), _f(r["gt"])
        if r["status"] != "ok" or pred is None or gt is None or r["n_views"] == "":
            continue
        tier = json.loads(r["config_json"] or "{}").get("tier", "") or "unknown"
        out.append(
            Row(r["scene_id"], r["method"], int(r["n_views"]), r["sample"], bool(int(r["ordered"])),
                pred, gt, _f(r["low"]), _f(r["high"]), tier,
                r["run_key"], _f(r["runtime_s"]), _f(r["peak_vram_mb"]),
                json.loads(r["timings_json"]) if r["timings_json"] else None,
                _f(r["assoc_f1"]), r["run_dir"])
        )  # fmt: skip
    return out


def rel_error(r: Row) -> float:
    """|pred - gt| / gt in percent; NaN when gt == 0 (matches ``bench.metrics.mape``)."""
    return float("nan") if r.gt == 0 else 100.0 * abs(r.pred - r.gt) / r.gt


def covered(r: Row) -> float:
    return float("nan") if r.low is None or r.high is None else float(r.low <= r.gt <= r.high)


def scene_level(rows: Sequence[Row], value) -> dict[str, float]:  # type: ignore[no-untyped-def]
    """Average ``value`` over a scene's rows (its samples), ignoring NaN."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = value(r)
        if not np.isnan(v):
            per[r.scene].append(v)
    return {s: float(np.mean(v)) for s, v in per.items()}


def point(rows: Sequence[Row], value, seed: int = 0) -> Point | None:  # type: ignore[no-untyped-def]
    vals = list(scene_level(rows, value).values())
    if not vals:
        return None
    ci = bootstrap_ci(vals, n_boot=2000, seed=seed)
    return Point(ci.estimate, ci.low, ci.high, ci.n)


def _errbar(ax, x, p: Point, **kw) -> None:  # type: ignore[no-untyped-def]
    ax.errorbar(x, p.mean, yerr=[[p.mean - p.lo], [p.hi - p.mean]], **kw)


def _methods(rows: Sequence[Row]) -> list[str]:
    return sorted({r.method for r in rows})


def fig_mape_vs_views(rows: Sequence[Row], out: Path) -> Path:
    """Headline: MAPE vs number of views, one line per method (ordered input only)."""
    apply_style()
    rows = [r for r in rows if r.ordered]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, m in enumerate(_methods(rows)):
        xs, pts = [], []
        for n in sorted({r.n_views for r in rows if r.method == m}):
            p = point([r for r in rows if r.method == m and r.n_views == n], rel_error)
            if p:
                xs.append(n)
                pts.append(p)
        if not pts:
            continue
        c = color_for(m, i)
        mean = np.array([p.mean for p in pts])
        ax.plot(xs, mean, marker="o", color=c, label=m)
        ax.errorbar(xs, mean, yerr=[mean - [p.lo for p in pts], [p.hi for p in pts] - mean],
                    fmt="none", ecolor=c, capsize=3, alpha=0.8)  # fmt: skip
    ax.set_xscale("log")
    xt = sorted({r.n_views for r in rows})
    ax.set_xticks(xt, [str(x) for x in xt])
    ax.set_xlabel("number of views")
    ax.set_ylabel("MAPE (%)  (mean over scenes, 95% bootstrap CI)")
    ax.set_title("Counting error vs. number of views")
    ax.legend(fontsize=8)
    save(fig, out)
    return out


def fig_ordered_vs_unordered(rows: Sequence[Row], out: Path) -> Path:
    """Grouped bars per method: ordered vs shuffled input, pooled over view counts."""
    apply_style()
    methods = _methods(rows)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(methods) + 2), 4))
    width = 0.38
    for j, (ordered, label, alpha) in enumerate([(True, "ordered", 1.0), (False, "shuffled", 0.5)]):
        for i, m in enumerate(methods):
            p = point([r for r in rows if r.method == m and r.ordered == ordered], rel_error)
            if p:
                x = i + (j - 0.5) * width
                ax.bar(x, p.mean, width, color=color_for(m, i), alpha=alpha,
                       label=label if i == 0 else None)  # fmt: skip
                _errbar(ax, x, p, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Ordered vs. shuffled input")
    ax.legend(fontsize=8)
    save(fig, out)
    return out


def fig_tier_error(rows: Sequence[Row], out: Path) -> Path:
    """Error by scene tier (easy / medium / hard), one bar group per tier, one bar per method."""
    apply_style()
    rows = [r for r in rows if r.ordered]
    tiers = [t for t in TIER_ORDER if any(r.tier == t for r in rows)]
    tiers += sorted({r.tier for r in rows} - set(TIER_ORDER))
    methods = _methods(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        for k, t in enumerate(tiers):
            p = point([r for r in rows if r.method == m and r.tier == t], rel_error)
            if p:
                x = k + (i - (len(methods) - 1) / 2) * width
                ax.bar(x, p.mean, width, color=color_for(m, i), label=m if k == 0 else None)
                _errbar(ax, x, p, fmt="none", ecolor="black", capsize=2)
    ax.set_xticks(range(len(tiers)), tiers)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Error by scene tier")
    ax.legend(fontsize=8)
    save(fig, out)
    return out


def fig_coverage(rows: Sequence[Row], out: Path) -> Path:
    """Interval coverage per method (share of scenes whose truth is inside [low, high])."""
    apply_style()
    methods = [m for m in _methods(rows) if any(r.method == m and r.low is not None for r in rows)]
    fig, ax = plt.subplots(figsize=(max(4.5, 1.0 * len(methods) + 2), 4))
    for i, m in enumerate(methods):
        p = point([r for r in rows if r.method == m], covered)
        if p:
            ax.bar(i, p.mean, color=color_for(m, i))
            _errbar(ax, i, p, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage of [low, high]")
    ax.set_title("Confidence range coverage")
    save(fig, out)
    return out


def _value(attr: str):  # type: ignore[no-untyped-def]
    def get(r: Row) -> float:
        v = getattr(r, attr)
        return float("nan") if v is None else float(v)

    return get


def fig_assoc_f1(rows: Sequence[Row], out: Path) -> Path | None:
    """Association F1 by method (v1 / v2 / sync ...), only runs from scenes with identity labels
    (rows whose ``assoc_f1`` is filled). Returns None when no row carries a label-based F1."""
    rows = [r for r in rows if r.assoc_f1 is not None]
    methods = _methods(rows)
    if not methods:
        return None
    apply_style()
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(methods) + 2), 4))
    for i, m in enumerate(methods):
        p = point([r for r in rows if r.method == m], _value("assoc_f1"))
        if p:
            ax.bar(i, p.mean, color=color_for(m, i))
            _errbar(ax, i, p, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pairwise association F1 (mean over scenes, 95% CI)")
    ax.set_title("Association F1 by method (scenes with identity labels)")
    save(fig, out)
    return out


def fig_runtime_vs_views(rows: Sequence[Row], out: Path) -> Path | None:
    """Left: runtime vs views (log y). Right: peak VRAM vs views. One line per method.
    The VRAM panel only has methods that report a positive peak (baselines do not)."""
    rows = [r for r in rows if r.ordered and r.runtime_s is not None]
    if not rows:
        return None
    apply_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4))
    for i, m in enumerate(_methods(rows)):
        c = color_for(m, i)
        for axis, attr in ((ax, "runtime_s"), (bx, "peak_vram_mb")):
            xs, pts = [], []
            for n in sorted({r.n_views for r in rows if r.method == m}):
                sel = [r for r in rows if r.method == m and r.n_views == n]
                if attr == "peak_vram_mb":
                    sel = [r for r in sel if (r.peak_vram_mb or 0) > 0]
                p = point(sel, _value(attr))
                if p:
                    xs.append(n)
                    pts.append(p)
            if not pts:
                continue
            mean = np.array([p.mean for p in pts])
            axis.plot(xs, mean, marker="o", color=c, label=m)
            axis.errorbar(xs, mean, yerr=[mean - [p.lo for p in pts], [p.hi for p in pts] - mean],
                          fmt="none", ecolor=c, capsize=3, alpha=0.8)  # fmt: skip
    xt = sorted({r.n_views for r in rows})
    for axis in (ax, bx):
        axis.set_xscale("log")
        axis.set_xticks(xt, [str(x) for x in xt])
        axis.set_xlabel("number of views")
    ax.set_yscale("log")
    ax.set_ylabel("runtime per run (s)")
    ax.set_title("Runtime vs. views")
    ax.legend(fontsize=8)
    bx.set_ylabel("peak VRAM (MB)")
    bx.set_title("Peak VRAM vs. views")
    save(fig, out)
    return out


@dataclass(frozen=True)
class AblationDelta:
    method: str
    diff: float  # mean over scenes of (MAPE_ablation - MAPE_base); < 0 means the ablation is better
    lo: float
    hi: float
    n_scenes: int


def ablation_deltas(rows: Sequence[Row], base: str | None = None) -> list[AblationDelta]:
    """Paired-bootstrap delta MAPE of each ``<base>+<ablation>=<value>`` method vs ``base``.

    ``base`` defaults to the prefix shared by the ``+`` methods when there is exactly one.
    Scenes are paired on the scene's mean percent error (ordered runs); scenes missing from
    either side are dropped.
    """
    rows = [r for r in rows if r.ordered]
    names = _methods(rows)
    if base is None:
        bases = {m.split("+", 1)[0] for m in names if "+" in m}
        if len(bases) != 1:
            return []
        base = bases.pop()
    base_scene = scene_level([r for r in rows if r.method == base], rel_error)
    out: list[AblationDelta] = []
    for m in names:
        if not m.startswith(base + "+"):
            continue
        cur = scene_level([r for r in rows if r.method == m], rel_error)
        common = sorted(set(cur) & set(base_scene))
        if not common:
            continue
        d, lo, hi, _ = paired_bootstrap([cur[s] for s in common], [base_scene[s] for s in common])
        out.append(AblationDelta(m, d, lo, hi, len(common)))
    return out


def fig_ablation(rows: Sequence[Row], out: Path, base: str | None = None) -> Path | None:
    """One-factor ablation chart: delta MAPE vs the ablation base, paired-bootstrap 95% CIs."""
    deltas = ablation_deltas(rows, base)
    if not deltas:
        return None
    deltas.sort(key=lambda d: d.diff)
    apply_style()
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(deltas) + 1.5)))
    ys = np.arange(len(deltas))
    diff = np.array([d.diff for d in deltas])
    ax.barh(ys, diff, color=[color_for(d.method, i) for i, d in enumerate(deltas)], alpha=0.8)
    ax.errorbar(diff, ys, xerr=[diff - [d.lo for d in deltas], [d.hi for d in deltas] - diff],
                fmt="none", ecolor="black", capsize=3)  # fmt: skip
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(ys, [d.method.split("+", 1)[1] for d in deltas])
    ax.set_xlabel("Δ MAPE vs. base (percentage points; < 0 is better; paired bootstrap 95% CI)")
    ax.set_title(f"One-factor ablations vs. {deltas[0].method.split('+', 1)[0]}")
    save(fig, out)
    return out


def make_all(csv_path: str | Path, out_dir: str | Path, base: str | None = None) -> list[Path]:
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path}: no usable ok rows with ground truth")
    out = Path(out_dir)
    paths: list[Path | None] = [
        fig_mape_vs_views(rows, out / "mape_vs_views.png"),
        fig_ordered_vs_unordered(rows, out / "ordered_vs_unordered.png"),
        fig_tier_error(rows, out / "tier_error.png"),
        fig_coverage(rows, out / "coverage.png"),
        fig_assoc_f1(rows, out / "assoc_f1.png"),
        fig_runtime_vs_views(rows, out / "runtime_vs_views.png"),
        fig_ablation(rows, out / "ablation.png", base),
    ]
    return [p for p in paths if p is not None]


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Render benchmark figures from results/bench.csv")
    p.add_argument("--csv", type=Path, default=Path("results/bench.csv"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--ablation-base", default=None, help="base method name for the ablation chart")
    a = p.parse_args(argv)
    for f in make_all(a.csv, a.out, a.ablation_base):
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
