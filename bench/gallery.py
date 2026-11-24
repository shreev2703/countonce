"""Systematic failure-gallery sampler (PLAN §9.8: worst 10% by error, never hand-picked)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from bench.figures import Row, load_rows, rel_error

MAX_SHEET_FRAMES = 12


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


def overlay_paths(r: Row, runs_dir: str | Path | None = None) -> list[Path]:
    """Overlay PNGs of a run: ``<run_dir>/overlays``, else ``<runs_dir>/<run_key>/overlays``."""
    cands = []
    if r.run_dir:
        cands.append(Path(r.run_dir))
    if runs_dir is not None and r.run_key:
        cands.append(Path(runs_dir) / r.run_key)
    for d in cands:
        files = sorted((d / "overlays").glob("frame_*.png"))
        if files:
            return files
    return []


def contact_sheet(
    images: Sequence[str | Path], out: str | Path, thumb: int = 240, cols: int = 4
) -> Path:
    """Tile ``images`` (evenly subsampled to ``MAX_SHEET_FRAMES``) into one PNG, aspect kept."""
    from PIL import Image

    if not images:
        raise ValueError("no images for a contact sheet")
    if len(images) > MAX_SHEET_FRAMES:
        step = len(images) / MAX_SHEET_FRAMES
        images = [images[int(i * step)] for i in range(MAX_SHEET_FRAMES)]
    cols = min(cols, len(images))
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for i, p in enumerate(images):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
            sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def write_gallery(
    csv_path: str | Path,
    out_path: str | Path,
    fraction: float = 0.10,
    runs_dir: str | Path | None = None,
) -> list[Row]:
    """Write the sampled failures as a Markdown table. Each run whose overlays exist gets a
    contact sheet in ``<out dir>/gallery/`` embedded in its row; others say "no overlay"."""
    worst = worst_fraction(load_rows(csv_path), fraction)
    lines = [
        f"# Failure gallery (worst {fraction:.0%} by error, per method)",
        "",
        "| scene | method | views | sample | ordered | pred | gt | error | likely cause "
        "| overlays |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    out_dir = Path(out_path).parent
    for r in worst:
        e = rel_error(r)
        es = f"{e:.1f}%" if not math.isnan(e) else f"{abs(r.pred - r.gt):g} (gt=0)"
        lines.append(
            f"| {r.scene} | {r.method} | {r.n_views} | {r.sample} | {int(r.ordered)} | "
            f"{r.pred:g} | {r.gt:g} | {es} | {failure_cause(r)} | "
            f"{_sheet_cell(r, out_dir, runs_dir)} |"
        )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return worst


def _sheet_cell(r: Row, out_dir: Path, runs_dir: str | Path | None) -> str:
    files = overlay_paths(r, runs_dir)
    if not files:
        return "no overlay"
    name = "_".join(
        "".join(c if c.isalnum() or c in "-." else "-" for c in str(v))
        for v in (r.scene, r.method, r.n_views, r.sample, int(r.ordered), r.run_key or "run")
    )
    contact_sheet(files, out_dir / "gallery" / f"{name}.png")
    return f"![{r.scene} {r.method}](gallery/{name}.png)"
