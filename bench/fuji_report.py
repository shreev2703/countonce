"""Fuji-SfM Core report (PLAN §9.4, §9.7, §9.8): every number comes from Fuji rows of bench.csv.

A *cell* is one (method, k, metric). It is written only if every expected (tree, side) slice has
an ``ok`` row for it; otherwise it stays ``TODO: fill from results``. ``write_cell`` raises
``MissingBackingRow`` when asked for a cell with no backing rows, so a number cannot be typed in
by hand or carried over from another source.

    python -m bench.fuji_report --csv results/bench.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bench import fuji_core
from bench.runner import read_rows
from bench.stats import bootstrap_ci, paired_compare

TODO = "TODO: fill from results"
METHODS = ("countonce_v1", "countonce_v2", "countonce_sync", "countonce_v3", "sum", "max", "median")
KS = (3, 5, 10, 20)
TEST_TREES = tuple(f"Tree{i:02d}" for i in range(1, 9))
SIDES = ("east", "west")
PROTOCOL = fuji_core.PROTOCOL
N_BOOT = 1000
METRICS = ("mae", "signed_bias", "mape", "assoc_f1", "duplicate_rate", "miss_rate")
PIPELINE = ("countonce_v1", "countonce_v2", "countonce_sync", "countonce_v3")
# Paired MAE comparisons (a, b); negative difference favours a.
PAIRS = (
    ("countonce_v2", "countonce_v1"),
    ("countonce_v2", "sum"),
    ("countonce_v2", "max"),
    ("countonce_v2", "median"),
    ("countonce_sync", "countonce_v2"),
    ("countonce_v3", "countonce_v2"),
)


class MissingBackingRow(ValueError):
    """A cell was requested that no (complete) set of bench rows backs."""


def fuji_rows(csv_path: str | Path, split: str = "test") -> list[dict[str, str]]:
    """Fuji Core rows only (protocol tag in ``config_json``), any status."""
    p = Path(csv_path)
    if not p.exists():
        return []
    out = []
    for r in read_rows(p):
        try:
            cfg = json.loads(r["config_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if cfg.get("protocol") == PROTOCOL and r["split"] == split:
            out.append(r)
    return out


def _num(s: str) -> float:
    return float(s) if s != "" else float("nan")


def _metric(r: dict[str, str], metric: str) -> float:
    pred, gt = _num(r["pred"]), _num(r["gt"])
    if metric == "mae":
        return abs(pred - gt)
    if metric == "signed_bias":
        return pred - gt
    if metric == "mape":
        return float("nan") if gt == 0 else 100.0 * abs(pred - gt) / gt
    return _num(r[metric])


def tree_values(
    rows: Sequence[dict[str, str]],
    method: str,
    k: int,
    metric: str,
    trees: Sequence[str] = TEST_TREES,
    sides: Sequence[str] = SIDES,
) -> dict[str, float] | None:
    """Per-tree value (mean over its sides), or None unless all (tree, side) slices have ok rows
    and the metric is defined for each of them. MAPE is the exception: slices with GT = 0 are
    skipped (trees with no GT apple on any side drop out), never counted as 0 or NaN."""
    per: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["status"] == "ok" and r["method"] == method and r["n_views"] == str(k):
            per[(r["scene_id"], r["sample"])] = _metric(r, metric)
    out: dict[str, float] = {}
    for t in trees:
        vals = [per.get((t, s), float("nan")) for s in sides]
        if metric == "mape":
            # GT = 0 makes MAPE undefined: skip that slice, but a missing row still blocks the cell
            if any((t, s) not in per for s in sides):
                return None
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue  # every side of this tree has GT = 0
        elif any(np.isnan(v) for v in vals):
            return None
        out[t] = float(np.mean(vals))
    return out if out else None


def compute(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Cells keyed ``method|k|metric`` and paired tests keyed ``a|b|k|metric``."""
    cells: dict[str, dict[str, Any]] = {}
    for m in METHODS:
        for k in KS:
            for metric in METRICS:
                v = tree_values(rows, m, k, metric)
                if v is None:
                    continue
                ci = bootstrap_ci(list(v.values()), n_boot=N_BOOT, seed=0)
                cells[f"{m}|{k}|{metric}"] = {
                    "estimate": ci.estimate, "low": ci.low, "high": ci.high, "n_trees": ci.n,
                }  # fmt: skip
    paired: dict[str, dict[str, Any]] = {}
    for k in KS:
        for a, b in PAIRS:
            va, vb = tree_values(rows, a, k, "mae"), tree_values(rows, b, k, "mae")
            if va is None or vb is None:
                continue
            r = paired_compare([va[t] for t in TEST_TREES], [vb[t] for t in TEST_TREES], N_BOOT)
            paired[f"{a}|{b}|{k}|mae"] = {
                "diff": r.diff, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "p_bootstrap": r.p_bootstrap, "p_wilcoxon": r.p_wilcoxon, "n": r.n,
            }  # fmt: skip
    return {"n_boot": N_BOOT, "cells": cells, "paired": paired}


def write_cell(report: dict[str, Any], key: str, fmt: str = "{:.2f}") -> str:
    """Formatted ``estimate [low, high]`` of a cell; raises if no row backs it."""
    c = report["cells"].get(key)
    if c is None:
        raise MissingBackingRow(f"no complete Fuji rows back cell {key!r}")
    e, lo, hi = (fmt.format(c[x]) for x in ("estimate", "low", "high"))
    return f"{e} [{lo}, {hi}]"


def cell_or_todo(report: dict[str, Any], key: str, fmt: str = "{:.2f}") -> str:
    try:
        return write_cell(report, key, fmt)
    except MissingBackingRow:
        return TODO


def markdown_table(report: dict[str, Any]) -> str:
    head = "| Method | k | MAE (apples) | signed bias (apples) | MAPE (%) |"
    head += " assoc F1 (explains errors) | duplicate rate | miss rate |"
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for k in KS:
            c = [
                cell_or_todo(report, f"{m}|{k}|mae"),
                cell_or_todo(report, f"{m}|{k}|signed_bias", "{:+.2f}"),
                cell_or_todo(report, f"{m}|{k}|mape", "{:.1f}"),
            ]
            for metric in ("assoc_f1", "duplicate_rate", "miss_rate"):
                c.append(
                    cell_or_todo(report, f"{m}|{k}|{metric}", "{:.3f}") if m in PIPELINE else "n/a"
                )
            lines.append(f"| {m} | {k} | " + " | ".join(c) + " |")
    lines += ["", "Paired MAE difference over trees (a - b, negative favours a):", ""]
    lines += [
        "| a vs b | k | diff [95% CI] | p (bootstrap) | p (Wilcoxon) |",
        "|---|---|---|---|---|",
    ]
    for k in KS:
        for a, b in PAIRS:
            p = report["paired"].get(f"{a}|{b}|{k}|mae")
            label = f"{a.removeprefix('countonce_')} vs {b.removeprefix('countonce_')}"
            if p is None:
                lines.append(f"| {label} | {k} | {TODO} | {TODO} | {TODO} |")
            else:
                lines.append(
                    f"| {label} | {k} | {p['diff']:.2f} [{p['ci_low']:.2f}, {p['ci_high']:.2f}]"
                    f" | {p['p_bootstrap']:.3f} | {p['p_wilcoxon']:.3f} |"
                )
    return "\n".join(lines)


def latex_table(report: dict[str, Any]) -> str:
    body = []
    for m in METHODS:
        for k in KS:
            mae = cell_or_todo(report, f"{m}|{k}|mae").replace("_", r"\_")
            name = m.replace("_", r"\_")
            body.append(f"{name} & {k} & {mae} \\\\")
    return (
        "\\begin{table*}[t]\\centering\\small\n"
        "\\caption{Fuji-SfM test trees (Tree01--08): MAE in apples, 95\\% bootstrap CI over "
        "trees. Cells without finished runs are left as TODO.}\\label{tab:fuji}\n"
        "\\begin{tabular}{llr}\\toprule\nMethod & $k$ & MAE \\\\ \\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\\end{tabular}\n\\end{table*}\n"
    )


def failure_analysis(rows: Sequence[dict[str, str]], top: int = 5) -> str:
    """Worst slices by absolute error and every failed row, each with its ``run_key``."""
    ok = [r for r in rows if r["status"] == "ok" and r["pred"] != "" and r["gt"] != ""]
    bad = [r for r in rows if r["status"] != "ok"]
    if not rows:
        return (
            "No Fuji test-split rows exist in `results/bench.csv`: step 2 has not produced any "
            "(it needs the frozen `configs/fuji_core.yaml` from step 1). There is therefore no "
            "measured failure to analyse; worst trees, foliage occlusion, registration and VGGT "
            "failures are all TODO until runs exist."
        )
    lines = []
    worst = sorted(ok, key=lambda r: -_metric(r, "mae"))[:top]
    lines.append("Worst slices by absolute count error:")
    for r in worst:
        lines.append(
            f"- {r['scene_id']}/{r['sample']} k={r['n_views']} {r['method']}: pred {r['pred']}, "
            f"gt {r['gt']} (run `{r['run_key']}`)"
        )
    per_tree: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        if r["method"] in PIPELINE:
            per_tree[r["scene_id"]].append(_metric(r, "mape"))
    if per_tree:
        ranked = sorted(per_tree.items(), key=lambda kv: -np.nanmean(kv[1]))
        lines.append("Pipeline MAPE by tree, worst first: " + ", ".join(
            f"{t} {np.nanmean(v):.1f}%" for t, v in ranked))  # fmt: skip
    if bad:
        lines.append(f"{len(bad)} failed rows (registration, VGGT or runtime errors):")
        for r in bad:
            lines.append(
                f"- {r['scene_id']}/{r['sample']} k={r['n_views']} {r['method']}: "
                f"{r['status']} {r['error'][:120]!r} (run `{r['run_key']}`)"
            )
    else:
        lines.append("No failed rows were recorded.")
    lines.append(
        "Foliage occlusion is not scored separately by the runner; GT apples hidden behind "
        "foliage are excluded by the visibility test, so occlusion shows up as miss rate."
    )
    return "\n".join(lines)


def build(csv_path: Path, root: Path) -> dict[str, Any]:
    rows = fuji_rows(csv_path)
    report = compute(rows)
    done = sum(1 for r in rows if r["status"] == "ok")
    report["n_rows"] = len(rows)
    fa = failure_analysis(rows)
    md = (
        "# Fuji-SfM Core results (generated)\n\n"
        "Generated by `python -m bench.fuji_report` from the Fuji test-split rows of "
        f"`results/bench.csv` ({done} ok rows). Do not edit by hand. Cells are estimate "
        f"[95% bootstrap CI over trees, {N_BOOT} resamples]. A cell is filled only when all "
        "8 trees x 2 sides have an ok row; otherwise it stays TODO. Own-capture cells are "
        "not covered here. See [[HANDOFF]].\n\n"
        + markdown_table(report)
        + "\n\n## Failure analysis\n\n"
        + fa
        + "\n"
    )
    extra = root / "docs/RESULTS_FUJI_ANALYSIS.md"  # hand-written reading of the numbers above
    if extra.exists():
        md += "\n" + extra.read_text()
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/RESULTS_FUJI.md").write_text(md)
    (root / "experimental/paper/tables").mkdir(parents=True, exist_ok=True)
    (root / "experimental/paper/tables/fuji_core.tex").write_text(latex_table(report))
    out = root / "results/fuji_core_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--csv", type=Path, default=Path("results/bench.csv"))
    p.add_argument("--root", type=Path, default=Path("."))
    a = p.parse_args(argv)
    rep = build(a.csv, a.root)
    from bench import web_summary  # lazy: web_summary imports this module

    web_summary.write(a.csv, a.root / "results/benchmark_summary.csv")
    print(f"{rep['n_rows']} Fuji rows, {len(rep['cells'])} cells filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
