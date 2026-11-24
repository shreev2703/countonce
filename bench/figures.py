"""Figure scripts for PLAN §9.7 (1, 2, 3, 8), all read from ``results/bench.csv``.

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
from bench.stats import bootstrap_ci
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
                pred, gt, _f(r["low"]), _f(r["high"]), tier)
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


def make_all(csv_path: str | Path, out_dir: str | Path) -> list[Path]:
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path}: no usable ok rows with ground truth")
    out = Path(out_dir)
    return [
        fig_mape_vs_views(rows, out / "mape_vs_views.png"),
        fig_ordered_vs_unordered(rows, out / "ordered_vs_unordered.png"),
        fig_tier_error(rows, out / "tier_error.png"),
        fig_coverage(rows, out / "coverage.png"),
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Render benchmark figures from results/bench.csv")
    p.add_argument("--csv", type=Path, default=Path("results/bench.csv"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    a = p.parse_args(argv)
    for f in make_all(a.csv, a.out):
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
