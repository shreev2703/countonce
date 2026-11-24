"""Split conformal and Jackknife+ count intervals (PLAN 8.8, Strong tier).

* Split conformal uses normalized scores ``|y - yhat| / sigma`` where ``sigma`` is the
  sensitivity-range width with a floor; the interval is ``yhat +/- q * sigma`` with ``q`` the
  ``ceil((n + 1)(1 - alpha))``-th smallest calibration score. Too few scenes is a hard error.
* Jackknife+ (Barber et al. 2021) refits a tuned parameter leaving one scene out at a time and
  builds the interval from the leave-one-out residuals (worst-case coverage 1 - 2 alpha).
  Residuals are unnormalized so the stated guarantee holds.

Coverage is marginal over scenes exchangeable with the calibration set, not per scene.
A :class:`CalibrationArtifact` (JSON) stores what ``run()`` needs; it is level-agnostic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_VERSION = 1
SIGMA_FLOOR = 1.0  # counts; avoids a zero-width range giving an infinite normalized score
MIN_REAL_SCENES = 15  # PLAN 8.8: real-data split conformal needs >= 15 real calibration scenes
_EPS = 1e-9


class CalibrationError(ValueError):
    """Calibration data is too small or otherwise unusable for the requested interval."""


def sigma_hat(low: float, high: float, floor: float = SIGMA_FLOOR, scale: float = 1.0) -> float:
    """Per-scene uncertainty: sensitivity-range width (times ``scale``), at least ``floor``."""
    return max((float(high) - float(low)) * scale, floor)


def _check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    return level


def min_calibration_size(level: float) -> int:
    """Smallest n with ceil((n + 1) * level) <= n, i.e. n >= level / (1 - level) (9 for 90%)."""
    return max(1, math.ceil(_check_level(level) / (1.0 - level) - _EPS))


def conformal_rank(n: int, level: float) -> int:
    """1-based rank ceil((n + 1) * level) of the conformal quantile; raises if it exceeds n."""
    _check_level(level)
    rank = math.ceil((n + 1) * level - _EPS)
    if rank > n or n < 1:
        raise CalibrationError(
            f"{n} calibration scenes cannot give a finite {level:.0%} interval; "
            f"need n >= {min_calibration_size(level)}"
        )
    return rank


def conformal_quantile(scores: Sequence[float], level: float) -> float:
    """The ceil((n + 1) * level)-th smallest score (finite-sample corrected)."""
    s = np.sort(np.asarray(scores, dtype=float))
    return float(s[conformal_rank(len(s), level) - 1])


def normalized_scores(
    y: Sequence[float], yhat: Sequence[float], sigma: Sequence[float]
) -> np.ndarray:
    return np.abs(np.asarray(y, float) - np.asarray(yhat, float)) / np.asarray(sigma, float)


def split_conformal_interval(
    yhat: float, sigma: float, scores: Sequence[float], level: float
) -> tuple[float, float]:
    """Continuous ``(low, high)``; low is clipped at 0 (counts are non-negative)."""
    q = conformal_quantile(scores, level)
    return max(0.0, yhat - q * sigma), yhat + q * sigma


def jackknife_plus_interval(
    mu_loo: Sequence[float], residuals: Sequence[float], level: float
) -> tuple[float, float]:
    """Jackknife+ interval at a test point.

    ``mu_loo[i]`` is the prediction at the test point from the fit that left scene ``i`` out;
    ``residuals[i]`` is ``|y_i - mu_{-i}(x_i)|``. Lower end: floor(alpha (n + 1))-th smallest
    of ``mu_i - R_i``; upper end: ceil((1 - alpha)(n + 1))-th smallest of ``mu_i + R_i``.
    """
    mu = np.asarray(mu_loo, float)
    res = np.asarray(residuals, float)
    n = len(mu)
    if n != len(res):
        raise ValueError("mu_loo and residuals must have the same length")
    conformal_rank(n, level)  # same finiteness condition; raises CalibrationError
    alpha = 1.0 - level
    lo_rank = math.floor(alpha * (n + 1) + _EPS)
    hi_rank = math.ceil(level * (n + 1) - _EPS)
    lower = np.sort(mu - res)[lo_rank - 1]
    upper = np.sort(mu + res)[hi_rank - 1]
    return max(0.0, float(lower)), float(upper)


def leave_one_out(
    scenes: Sequence[Any],
    y_of: Callable[[Any], float],
    fit: Callable[[Sequence[Any]], Any],
    predict: Callable[[Any, Any], float],
) -> tuple[list[Any], list[float]]:
    """Leave-one-scene-out refit of a pluggable tuned parameter.

    ``fit(train_scenes) -> theta``, ``predict(theta, scene) -> yhat``. Returns the per-scene
    ``theta_{-i}`` and residuals ``|y_i - predict(theta_{-i}, scene_i)|``.
    """
    if len(scenes) < 2:
        raise CalibrationError("leave-one-out needs at least 2 scenes")
    thetas, resid = [], []
    for i, s in enumerate(scenes):
        theta = fit([t for j, t in enumerate(scenes) if j != i])
        thetas.append(theta)
        resid.append(abs(y_of(s) - float(predict(theta, s))))
    return thetas, resid


@dataclass
class CalibrationArtifact:
    """Stored calibration: scores for split conformal, LOO fits/residuals for Jackknife+."""

    n_calibration: int
    n_real: int
    scores: list[float]  # normalized split-conformal scores
    sigma_floor: float = SIGMA_FLOOR
    tuned: str | None = None  # name of the Jackknife+ tuned parameter ("k", "scale", ...)
    thetas: list[float] = field(default_factory=list)  # theta_{-i}, one per scene
    residuals: list[float] = field(default_factory=list)  # |y_i - mu_{-i}(x_i)|
    scene_ids: list[str] = field(default_factory=list)
    source_split: str = "val"
    version: int = ARTIFACT_VERSION

    @property
    def has_jackknife(self) -> bool:
        return self.tuned is not None and len(self.thetas) == len(self.residuals) > 1

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> CalibrationArtifact:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("version") != ARTIFACT_VERSION:
            raise CalibrationError(f"{path}: unsupported version {raw.get('version')!r}")
        return cls(**raw)


def build_artifact(
    y: Sequence[float],
    yhat: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    *,
    n_real: int | None = None,
    scene_ids: Sequence[str] = (),
    tuned: str | None = None,
    thetas: Sequence[float] = (),
    residuals: Sequence[float] = (),
    floor: float = SIGMA_FLOOR,
    source_split: str = "val",
) -> CalibrationArtifact:
    sigma = [sigma_hat(lo, hi, floor) for lo, hi in zip(low, high, strict=True)]
    scores = normalized_scores(y, yhat, sigma)
    return CalibrationArtifact(
        n_calibration=len(scores),
        n_real=len(scores) if n_real is None else n_real,
        scores=[float(s) for s in scores],
        sigma_floor=floor,
        tuned=tuned,
        thetas=[float(t) for t in thetas],
        residuals=[float(r) for r in residuals],
        scene_ids=list(scene_ids),
        source_split=source_split,
    )


def empirical_coverage(y: Sequence[float], low: Sequence[float], high: Sequence[float]) -> float:
    y_, lo, hi = (np.asarray(a, float) for a in (y, low, high))
    return float(np.mean((lo <= y_) & (y_ <= hi)))


def to_count_interval(low: float, high: float, count: int) -> tuple[int, int]:
    """Round outward to integers and make sure the reported count is inside."""
    return min(int(math.floor(low + _EPS)), count), max(int(math.ceil(high - _EPS)), count)


@dataclass
class Resolved:
    method: str  # "sensitivity" | "split_conformal" | "jackknife_plus"
    warning: dict[str, str] | None = None


def resolve_method(requested: str, art: CalibrationArtifact | None) -> Resolved:
    """Pick the interval method actually used, enforcing the >= 15 real-scene rule.

    ``requested`` is ``conformal`` or ``jackknife_plus``. Split conformal on real data with
    fewer than 15 real calibration scenes falls back to Jackknife+ when the artifact has LOO
    fits, else to the sensitivity range, with a warning either way.
    """
    if art is None:
        raise CalibrationError(f"interval_method={requested!r} needs a calibration artifact")
    if requested == "jackknife_plus":
        if not art.has_jackknife:
            raise CalibrationError("calibration artifact has no Jackknife+ fits")
        return Resolved("jackknife_plus")
    if art.n_real >= MIN_REAL_SCENES:
        return Resolved("split_conformal")
    msg = (
        f"only {art.n_real} real calibration scenes (< {MIN_REAL_SCENES}); "
        "split conformal is not reported"
    )
    fix = "Calibrate on at least 15 held-out real scenes."
    if art.has_jackknife:
        return Resolved(
            "jackknife_plus",
            {"code": "conformal_fallback", "message": msg + "; using Jackknife+", "fix": fix},
        )
    return Resolved(
        "sensitivity",
        {"code": "conformal_fallback", "message": msg + "; using sensitivity range", "fix": fix},
    )


def interval_from_artifact(
    method: str,
    art: CalibrationArtifact,
    count: int,
    low: float,
    high: float,
    level: float,
    mu_at: Callable[[float], float] | None = None,
) -> tuple[int, int]:
    """Integer ``(low, high)`` for one scene. ``mu_at(theta)`` re-predicts the count under the
    tuned parameter ``theta`` (only used for Jackknife+)."""
    if method == "split_conformal":
        sig = sigma_hat(low, high, art.sigma_floor)
        lo, hi = split_conformal_interval(float(count), sig, art.scores, level)
    elif method == "jackknife_plus":
        if mu_at is None:
            raise ValueError("jackknife_plus needs mu_at")
        cache: dict[float, float] = {}
        mus = [cache.setdefault(t, float(mu_at(t))) for t in art.thetas]
        lo, hi = jackknife_plus_interval(mus, art.residuals, level)
    else:
        raise ValueError(f"unknown conformal method {method!r}")
    return to_count_interval(lo, hi, count)


def coverage_by_level(
    art: CalibrationArtifact,
    y: Sequence[float],
    yhat: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    levels: Sequence[float],
) -> dict[float, float]:
    """Empirical split-conformal coverage on held-out scenes per target level (levels the
    calibration set is too small for are omitted)."""
    out: dict[float, float] = {}
    for lv in levels:
        try:
            ivs = [
                split_conformal_interval(p, sigma_hat(lo, hi, art.sigma_floor), art.scores, lv)
                for p, lo, hi in zip(yhat, low, high, strict=True)
            ]
        except CalibrationError:
            continue
        out[float(lv)] = empirical_coverage(y, [a for a, _ in ivs], [b for _, b in ivs])
    return out
