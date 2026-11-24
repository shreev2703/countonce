"""Systematic failure-gallery sampler (PLAN §9.8: worst 10% by error, never hand-picked)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from bench.figures import Row, load_rows, rel_error


def failure_cause(r: Row) -> str:
    """Coarse label from the count error alone; the real cause needs the overlay."""
    if r.pred > r.gt:
        return "over-count (merge failures / duplicates / hallucinations)"
    if r.pred < r.gt:
        return "under-count (wrong merges / misses)"
    return "exact"


def worst_fraction(
    rows: Sequence[Row], fraction: float = 0.10, per_method: bool = True
) -> list[Row]:
    """The worst ``fraction`` of runs by error, ranked worst first.

    Error is relative (percent) where gt > 0 and the absolute count error otherwise, so
    zero-count false positives are not silently dropped. Ties break on (scene, sample) so the
    selection is deterministic. At least one run is returned per group (ceil).
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    def err(r: Row) -> float:
        e = rel_error(r)
        return abs(r.pred - r.gt) if math.isnan(e) else e

    def key(r: Row) -> tuple[float, str, str, int, bool]:
        return (-err(r), r.scene, r.sample, r.n_views, r.ordered)

    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.method if per_method else "", []).append(r)
    out: list[Row] = []
    for g in groups.values():
        g = sorted(g, key=key)
        out.extend(g[: max(1, math.ceil(fraction * len(g)))])
    return sorted(out, key=key)


def write_gallery(csv_path: str | Path, out_path: str | Path, fraction: float = 0.10) -> list[Row]:
    """Write the sampled failures as a Markdown table (overlay path left for the reviewer)."""
    worst = worst_fraction(load_rows(csv_path), fraction)
    lines = [
        f"# Failure gallery (worst {fraction:.0%} by error, per method)",
        "",
        "| scene | method | views | sample | ordered | pred | gt | error | likely cause |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in worst:
        e = rel_error(r)
        es = f"{e:.1f}%" if not math.isnan(e) else f"{abs(r.pred - r.gt):g} (gt=0)"
        lines.append(
            f"| {r.scene} | {r.method} | {r.n_views} | {r.sample} | {int(r.ordered)} | "
            f"{r.pred:g} | {r.gt:g} | {es} | {failure_cause(r)} |"
        )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return worst
