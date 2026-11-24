"""Scene-level bootstrap CIs and paired tests (PLAN §9.8). Resampling unit is the scene."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class CI:
    estimate: float
    low: float
    high: float
    level: float
    n: int


@dataclass(frozen=True)
class PairedResult:
    """Paired comparison of per-scene errors; diff = mean(a - b), so diff < 0 means A is better
    when the inputs are errors."""

    diff: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    p_wilcoxon: float
    n: int


Values = Sequence[float] | npt.NDArray[Any]  # per-scene numbers: list or ndarray


def _clean(x: Values) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def _boot_stats(
    values: np.ndarray, statistic: Statistic, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    if statistic is np.mean:
        return values[idx].mean(axis=1)
    return np.array([statistic(values[i]) for i in idx])


def bootstrap_ci(
    values: Values,
    statistic: Statistic = np.mean,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI over scenes. NaNs (e.g. undefined per-scene MAPE) are dropped."""
    v = _clean(values)
    if v.size == 0:
        raise ValueError("no finite values to bootstrap")
    rng = np.random.default_rng(seed)
    boots = _boot_stats(v, statistic, n_boot, rng)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [a, 1.0 - a])
    return CI(float(statistic(v)), float(lo), float(hi), level, int(v.size))


def paired_bootstrap(
    a: Values,
    b: Values,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on per-scene differences d = a - b.

    Returns (mean diff, CI low, CI high, two-sided p). The p-value is twice the smaller tail
    mass of the bootstrap distribution on either side of 0, capped at 1.
    """
    pa, pb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if pa.shape != pb.shape or pa.ndim != 1:
        raise ValueError("paired inputs must be 1-D and the same length")
    d = pa - pb
    d = d[~np.isnan(d)]
    if d.size == 0:
        raise ValueError("no finite paired differences")
    rng = np.random.default_rng(seed)
    boots = _boot_stats(d, np.mean, n_boot, rng)
    al = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [al, 1.0 - al])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def wilcoxon_p(a: Values, b: Values) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when every difference is zero."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0 or np.all(d == 0):
        return 1.0
    return float(sps.wilcoxon(d, alternative="two-sided").pvalue)


def paired_compare(
    a: Values,
    b: Values,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedResult:
    """Paired bootstrap plus Wilcoxon over the same scenes (inputs are per-scene errors)."""
    diff, lo, hi, p_boot = paired_bootstrap(a, b, n_boot, level, seed)
    n = int((~np.isnan(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))).sum())
    return PairedResult(diff, lo, hi, p_boot, wilcoxon_p(a, b), n)
