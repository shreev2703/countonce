"""Benchmark metrics (PLAN §9.4). Pure numpy; every function takes plain sequences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np


def _arrays(pred: Sequence[float], gt: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float)
    g = np.asarray(gt, dtype=float)
    if p.shape != g.shape or p.ndim != 1:
        raise ValueError(f"pred and gt must be 1-D and equal length, got {p.shape} vs {g.shape}")
    if p.size == 0:
        raise ValueError("need at least one scene")
    return p, g


def abs_errors(pred: Sequence[float], gt: Sequence[float]) -> np.ndarray:
    p, g = _arrays(pred, gt)
    return np.abs(p - g)


def pct_errors(pred: Sequence[float], gt: Sequence[float]) -> np.ndarray:
    """Per-scene |pred - gt| / gt as a fraction; NaN where gt == 0 (undefined)."""
    p, g = _arrays(pred, gt)
    out = np.full(p.shape, np.nan)
    nz = g != 0
    out[nz] = np.abs(p[nz] - g[nz]) / g[nz]
    return out


def mae(pred: Sequence[float], gt: Sequence[float]) -> float:
    return float(abs_errors(pred, gt).mean())


def mape(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Mean absolute percentage error in percent. Zero-count scenes are excluded
    (use the false-positive metrics of §9.8b for those); NaN if every scene has gt == 0."""
    e = pct_errors(pred, gt)
    e = e[~np.isnan(e)]
    return float(100.0 * e.mean()) if e.size else float("nan")


def within_tolerance(pred: Sequence[float], gt: Sequence[float], tol: float = 0.10) -> np.ndarray:
    """Per-scene indicator |pred - gt| <= tol * gt. With gt == 0 only an exact 0 passes."""
    p, g = _arrays(pred, gt)
    return np.abs(p - g) <= tol * g + 1e-12


def within_rate(pred: Sequence[float], gt: Sequence[float], tol: float = 0.10) -> float:
    """Fraction of scenes within ``tol`` (default 10%) of the true count."""
    return float(within_tolerance(pred, gt, tol).mean())


def interval_coverage(low: Sequence[float], high: Sequence[float], gt: Sequence[float]) -> float:
    """Fraction of scenes whose truth lies inside the closed interval [low, high]."""
    lo, g = _arrays(low, gt)
    hi, _ = _arrays(high, gt)
    if np.any(lo > hi):
        raise ValueError("interval low exceeds high")
    return float(((lo <= g) & (g <= hi)).mean())


# ---------------------------------------------------------------- identity-labelled metrics


def _labeled(
    clusters: Sequence[Hashable], gt_ids: Sequence[Hashable | None]
) -> list[tuple[Hashable, Hashable]]:
    if len(clusters) != len(gt_ids):
        raise ValueError("clusters and gt_ids must have equal length")
    return [(c, g) for c, g in zip(clusters, gt_ids, strict=True) if g is not None]


def _cluster_majorities(pairs: list[tuple[Hashable, Hashable]]) -> dict[Hashable, Hashable]:
    """Majority GT id of each cluster (ties broken by the smallest ``repr`` for determinism)."""
    per: dict[Hashable, Counter[Hashable]] = {}
    for c, g in pairs:
        per.setdefault(c, Counter())[g] += 1
    return {c: min(cnt, key=lambda k: (-cnt[k], repr(k))) for c, cnt in per.items()}


def duplicate_rate(clusters: Sequence[Hashable], gt_ids: Sequence[Hashable | None]) -> float:
    """Share of predicted instances that duplicate an object another instance already covers.

    Each cluster is assigned its majority GT id (detections with ``gt_id=None`` are ignored);
    clusters beyond the first for the same GT object are duplicates. NaN with no labelled
    detections.
    """
    maj = _cluster_majorities(_labeled(clusters, gt_ids))
    if not maj:
        return float("nan")
    return (len(maj) - len(set(maj.values()))) / len(maj)


def ghost_duplicate_rate(
    clusters: Sequence[Hashable],
    gt_ids: Sequence[Hashable | None],
    moving_gt: Sequence[Hashable],
) -> float:
    """Share of predicted instances that are extra copies of a *moving* GT object (PLAN 8.3b).

    Like :func:`duplicate_rate` (majority GT id per cluster) but only counts clusters beyond the
    first for GT objects in ``moving_gt``. NaN when there are no labelled detections or no
    moving objects are listed.
    """
    maj = _cluster_majorities(_labeled(clusters, gt_ids))
    moving = set(moving_gt)
    if not maj or not moving:
        return float("nan")
    ghosts = 0
    for g in moving:
        n = sum(1 for v in maj.values() if v == g)
        ghosts += max(n - 1, 0)
    return ghosts / len(maj)


def miss_rate(
    clusters: Sequence[Hashable],
    gt_ids: Sequence[Hashable | None],
    gt_universe: Sequence[Hashable] | None = None,
) -> float:
    """Share of true objects not represented by any predicted instance.

    ``gt_universe`` lists every GT object; objects never detected at all count as misses.
    Defaults to the GT ids seen in ``gt_ids`` (which then only measures association misses).
    """
    maj = _cluster_majorities(_labeled(clusters, gt_ids))
    universe = set(gt_universe) if gt_universe is not None else {g for g in gt_ids if g is not None}
    if not universe:
        return float("nan")
    return 1.0 - len(universe & set(maj.values())) / len(universe)


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def pairwise_prf(
    clusters: Sequence[Hashable], gt_ids: Sequence[Hashable | None]
) -> tuple[float, float, float]:
    """Precision, recall and F1 of the "same object" decision over detection pairs.

    A pair is predicted-positive if both detections share a cluster and truly positive if they
    share a GT id. Unlabelled detections are ignored. Precision is 1.0 when nothing is predicted
    positive and recall is 1.0 when nothing is truly positive (vacuous); F1 is 0 if p + r == 0.
    """
    pairs = _labeled(clusters, gt_ids)
    pred_pos = sum(_pairs(n) for n in Counter(c for c, _ in pairs).values())
    true_pos_all = sum(_pairs(n) for n in Counter(g for _, g in pairs).values())
    tp = sum(_pairs(n) for n in Counter(pairs).values())
    precision = tp / pred_pos if pred_pos else 1.0
    recall = tp / true_pos_all if true_pos_all else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def cluster_purity(clusters: Sequence[Hashable], gt_ids: Sequence[Hashable | None]) -> float:
    """Sum over clusters of the majority-GT detection count, over all labelled detections."""
    pairs = _labeled(clusters, gt_ids)
    if not pairs:
        return float("nan")
    per: dict[Hashable, Counter[Hashable]] = {}
    for c, g in pairs:
        per.setdefault(c, Counter())[g] += 1
    return sum(max(cnt.values()) for cnt in per.values()) / len(pairs)


# ---------------------------------------------------------------- zero-count metrics (§9.8b)


def zero_count_metrics(
    pred: Sequence[float],
    gt: Sequence[float],
    low: Sequence[float] | None = None,
    high: Sequence[float] | None = None,
) -> dict[str, float]:
    """False-positive behaviour on scenes whose true count is 0.

    Only scenes with ``gt == 0`` are used. Returns ``false_positive_rate`` (share with predicted
    count > 0), ``mean_false_count`` (mean predicted count over those scenes) and, when
    intervals are given, ``interval_includes_zero_rate`` (share with ``low <= 0``). Every value
    is NaN when there are no zero-count scenes.
    """
    p, g = _arrays(pred, gt)
    zero = g == 0
    nan = float("nan")
    out = {
        "n_zero": float(zero.sum()),
        "false_positive_rate": nan,
        "mean_false_count": nan,
    }
    if (low is None) != (high is None):
        raise ValueError("give both low and high, or neither")
    if low is not None and high is not None:
        out["interval_includes_zero_rate"] = nan
        lo, _ = _arrays(low, gt)
        hi, _ = _arrays(high, gt)
        if np.any(lo > hi):
            raise ValueError("interval low exceeds high")
    if not zero.any():
        return out
    out["false_positive_rate"] = float((p[zero] > 0).mean())
    out["mean_false_count"] = float(p[zero].mean())
    if low is not None:
        out["interval_includes_zero_rate"] = float(((lo[zero] <= 0) & (hi[zero] >= 0)).mean())
    return out


def false_positive_rate(pred: Sequence[float], gt: Sequence[float]) -> float:
    return zero_count_metrics(pred, gt)["false_positive_rate"]


def mean_false_count(pred: Sequence[float], gt: Sequence[float]) -> float:
    return zero_count_metrics(pred, gt)["mean_false_count"]


def interval_includes_zero_rate(
    low: Sequence[float], high: Sequence[float], pred: Sequence[float], gt: Sequence[float]
) -> float:
    return zero_count_metrics(pred, gt, low, high)["interval_includes_zero_rate"]
