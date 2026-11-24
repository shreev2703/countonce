"""Sensitivity-range confidence (PLAN 8.8, Core tier).

Re-count the scene over a grid of merge strictness ``k`` and detector score thresholds. The
reported ``count`` is the default setting; ``low``/``high`` are the 10th/90th percentile of the
grid counts (widened to include ``count``). This is a sensitivity range, not a statistical
guarantee, so it is labelled ``method="sensitivity"``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from countonce.types import Interval

K_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
THRESHOLD_OFFSETS = (-0.1, -0.05, 0.0, 0.05, 0.1)
LOW_PCT, HIGH_PCT = 10.0, 90.0
# Nominal share of the grid inside [10th, 90th]; recorded as Interval.level (not a coverage claim).
NOMINAL_LEVEL = (HIGH_PCT - LOW_PCT) / 100.0


@dataclass
class SensitivityRange:
    count: int
    low: int
    high: int
    grid: dict[tuple[float, float], int] = field(default_factory=dict)  # (k, threshold) -> count

    @property
    def width(self) -> int:
        return self.high - self.low

    def to_interval(self) -> Interval:
        return Interval(self.low, self.high, NOMINAL_LEVEL, "sensitivity")


def k_grid(base: float, factors: Sequence[float] = K_FACTORS) -> list[float]:
    return [base * f for f in factors]


def threshold_grid(base: float, offsets: Sequence[float] = THRESHOLD_OFFSETS) -> list[float]:
    """Distinct thresholds around ``base``, clipped to [0.01, 0.99]; always contains ``base``."""
    vals = {round(min(0.99, max(0.01, base + o)), 6) for o in offsets}
    vals.add(round(base, 6))
    return sorted(vals)


def sensitivity_range(
    count_fn: Callable[[float, float], int],
    k_values: Sequence[float],
    thresholds: Sequence[float],
    default_k: float,
    default_threshold: float,
) -> SensitivityRange:
    """Evaluate ``count_fn(k, threshold)`` on the grid and summarise it.

    ``count_fn`` is called once per grid cell, and the default cell is always included.
    """
    ks = sorted({*map(float, k_values), float(default_k)})
    ts = sorted({*map(float, thresholds), float(default_threshold)})
    grid = {(k, t): int(count_fn(k, t)) for t in ts for k in ks}
    count = grid[(float(default_k), float(default_threshold))]
    vals = np.array(list(grid.values()), dtype=float)
    low = int(np.floor(np.percentile(vals, LOW_PCT)))
    high = int(np.ceil(np.percentile(vals, HIGH_PCT)))
    return SensitivityRange(count, min(low, count), max(high, count), grid)
