"""Naive counting baselines (PLAN §9.3 items 1-3) over per-image detection counts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def sum_baseline(per_image_counts: Sequence[float]) -> float:
    """Sum of per-image counts (over-counts: every overlap is counted again)."""
    return float(np.sum(per_image_counts)) if len(per_image_counts) else 0.0


def max_baseline(per_image_counts: Sequence[float]) -> float:
    """Largest single-image count (under-counts: no image sees everything)."""
    return float(np.max(per_image_counts)) if len(per_image_counts) else 0.0


def median_baseline(per_image_counts: Sequence[float]) -> float:
    """Median per-image count (sanity check). Returned unrounded, so an even number of
    images can give a half-integer."""
    return float(np.median(per_image_counts)) if len(per_image_counts) else 0.0


BASELINES: dict[str, Callable[[Sequence[float]], float]] = {
    "sum": sum_baseline,
    "max": max_baseline,
    "median": median_baseline,
}
