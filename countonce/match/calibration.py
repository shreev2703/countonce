"""Probability calibration for match scores (PLAN §10.5): ECE, reliability data, temperature.

numpy only. A match probability ``p`` is recalibrated as ``sigmoid(logit(p) / T)``; ``T`` is one
scalar fitted by minimising the binary NLL on the *validation* split. ``T > 1`` softens an
overconfident model. Never fit on the test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


def _check(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels).astype(float).ravel()
    if p.shape != y.shape:
        raise ValueError("probs and labels must have the same number of elements")
    return p, y


@dataclass(frozen=True)
class Reliability:
    """Per-bin reliability-diagram data (empty bins have count 0 and NaN means)."""

    edges: np.ndarray  # (n_bins+1,)
    count: np.ndarray  # (n_bins,)
    confidence: np.ndarray  # (n_bins,) mean predicted probability
    accuracy: np.ndarray  # (n_bins,) observed positive rate

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "edges": self.edges.tolist(),
            "count": self.count.tolist(),
            "confidence": self.confidence.tolist(),
            "accuracy": self.accuracy.tolist(),
        }


def reliability_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Reliability:
    p, y = _check(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.bincount(idx, weights=p, minlength=n_bins) / count
        acc = np.bincount(idx, weights=y, minlength=n_bins) / count
    return Reliability(edges, count, conf, acc)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted mean |accuracy - confidence| over bins."""
    p, _ = _check(probs, labels)
    if p.size == 0:
        raise ValueError("ECE needs at least one sample")
    r = reliability_data(probs, labels, n_bins)
    used = r.count > 0
    gap = np.abs(r.accuracy[used] - r.confidence[used])
    return float((r.count[used] * gap).sum() / r.count.sum())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    out: np.ndarray = np.log(p) - np.log1p(-p)
    return out


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = _logit(np.asarray(probs, dtype=float)) / temperature
    return 1.0 / (1.0 + np.exp(-z))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p, y = _check(probs, labels)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def fit_temperature(
    val_probs: np.ndarray, val_labels: np.ndarray, lo: float = 0.05, hi: float = 20.0
) -> float:
    """One-parameter temperature fit on validation data (golden section on log T).

    The binary NLL is convex in 1/T, hence unimodal in log T, so a bracketed search is exact.
    """
    p, y = _check(val_probs, val_labels)
    if p.size == 0:
        raise ValueError("cannot fit a temperature on an empty validation set")
    z = _logit(p)

    def loss(log_t: float) -> float:
        q = 1.0 / (1.0 + np.exp(-z / np.exp(log_t)))
        return nll(q, y)

    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    return float(np.exp((a + b) / 2))
