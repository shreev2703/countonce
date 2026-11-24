"""Frame selection for ingest (PLAN section 8.1): near-duplicate removal and the frame cap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from countonce.ingest.frames import sample_frames

THUMB = 16


def thumbnail(path: Path, size: int = THUMB) -> np.ndarray:
    """Downsampled grayscale thumbnail in [0, 1], zero-mean/unit-norm for comparison."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size, size), Image.Resampling.BOX), np.float64)
    return g / 255.0


def drop_near_duplicates(
    paths: list[Path], threshold: float = 0.02
) -> tuple[list[Path], dict[str, float]]:
    """Drop a frame when it is nearly identical to the last *kept* frame.

    Distance is the mean absolute difference of 16x16 grayscale thumbnails (0 = same,
    1 = black vs white). Comparing against the last kept frame, not the previous one,
    stops slow drift from chaining many almost-identical frames through. Returns the
    kept paths and each frame's distance to its reference (``inf`` for the first).
    """
    kept: list[Path] = []
    dists: dict[str, float] = {}
    ref: np.ndarray | None = None
    for p in paths:
        t = thumbnail(p)
        d = float("inf") if ref is None else float(np.abs(t - ref).mean())
        dists[p.name] = d
        if d >= threshold:
            kept.append(p)
            ref = t
    return kept, dists


def cap_frames(paths: list[Path], cap: int) -> list[Path]:
    """At most ``cap`` frames, evenly spaced over the order; unchanged when under the cap."""
    return sample_frames(paths, cap, None)
