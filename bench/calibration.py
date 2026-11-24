"""Calibration-set builder for conformal / Jackknife+ intervals (PLAN 8.8).

Only scenes from the frozen ``val`` split (``bench/splits.json``) may calibrate; the ``test``
split is refused so reported coverage stays honest. Synthetic scenes are not in the split file;
they are accepted as ``domain="synthetic"`` unless their id is recorded as ``test``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.splits import DEFAULT_PATH, load_splits
from countonce.conformal import (
    SIGMA_FLOOR,
    CalibrationArtifact,
    build_artifact,
    leave_one_out,
)


class SplitViolation(ValueError):
    """A calibration scene belongs to the frozen test split (or is an unassigned real scene)."""


@dataclass(frozen=True)
class CalibrationScene:
    scene_id: str
    y: float  # ground-truth count
    count: float  # CountOnce count at the default setting
    low: float  # sensitivity range
    high: float
    domain: str = "real"  # "real" | "synthetic"
    k_counts: dict[float, float] = field(default_factory=dict)  # k factor -> count


def check_calibration_scenes(
    scenes: Sequence[CalibrationScene], splits_path: str | Path = DEFAULT_PATH, split: str = "val"
) -> None:
    """Raise :class:`SplitViolation` unless every scene may be used for calibration."""
    if split == "test":
        raise SplitViolation("the test split is never used for calibration")
    splits = load_splits(splits_path)
    allowed = set(splits.get(split, {}))
    tests = set(splits.get("test", {}))
    for s in scenes:
        if s.scene_id in tests:
            raise SplitViolation(f"{s.scene_id}: in the frozen test split")
        if s.domain == "real" and s.scene_id not in allowed:
            raise SplitViolation(f"{s.scene_id}: real scene not in the '{split}' split")


def _k_fit(train: Sequence[CalibrationScene]) -> float:
    """Tuned k factor: minimises total absolute error over the training scenes."""
    factors = sorted(set.intersection(*(set(s.k_counts) for s in train)))
    if not factors:
        raise ValueError("scenes share no k factors")
    best = min(factors, key=lambda f: (sum(abs(s.k_counts[f] - s.y) for s in train), abs(f - 1)))
    return float(best)


def _scale_fit(train: Sequence[CalibrationScene]) -> float:
    """Global count multiplier (ratio of sums)."""
    den = sum(s.count for s in train)
    return float(sum(s.y for s in train) / den) if den > 0 else 1.0


TUNERS: dict[str, tuple[Callable[..., float], Callable[..., float]]] = {
    "k": (_k_fit, lambda theta, s: s.k_counts[theta]),
    "scale": (_scale_fit, lambda theta, s: theta * s.count),
}


def build_calibration_set(
    scenes: Sequence[CalibrationScene],
    splits_path: str | Path = DEFAULT_PATH,
    *,
    split: str = "val",
    tuned: str | None = None,
    floor: float = SIGMA_FLOOR,
) -> CalibrationArtifact:
    """Validate scenes against the frozen splits and fit the calibration artifact.

    ``tuned`` ("k" or "scale") additionally stores the leave-one-scene-out fits and residuals
    for Jackknife+; ``None`` stores split-conformal scores only.
    """
    check_calibration_scenes(scenes, splits_path, split)
    if not scenes:
        raise ValueError("no calibration scenes")
    ids = [s.scene_id for s in scenes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate calibration scene ids")
    thetas: list[float] = []
    resid: list[float] = []
    if tuned is not None:
        if tuned not in TUNERS:
            raise ValueError(f"unknown tuned parameter {tuned!r}; choose from {sorted(TUNERS)}")
        fit, predict = TUNERS[tuned]
        thetas, resid = leave_one_out(scenes, lambda s: s.y, fit, predict)
    return build_artifact(
        [s.y for s in scenes],
        [s.count for s in scenes],
        [s.low for s in scenes],
        [s.high for s in scenes],
        n_real=sum(s.domain == "real" for s in scenes),
        scene_ids=ids,
        tuned=tuned,
        thetas=thetas,
        residuals=resid,
        floor=floor,
        source_split=split,
    )


def scenes_from_rows(
    rows: Sequence[Any], domain: str = "real", method: str | None = None
) -> list[CalibrationScene]:
    """Scene-level calibration records from ``bench.figures.Row`` objects (rows with a range,
    averaged over a scene's samples). No k grid, so only ``tuned="scale"`` is available."""
    by: dict[str, list[Any]] = {}
    for r in rows:
        if r.low is None or r.high is None or (method and r.method != method):
            continue
        by.setdefault(r.scene, []).append(r)
    return [
        CalibrationScene(
            sid,
            float(np.mean([r.gt for r in rs])),
            float(np.mean([r.pred for r in rs])),
            float(np.mean([r.low for r in rs])),
            float(np.mean([r.high for r in rs])),
            domain,
        )
        for sid, rs in sorted(by.items())
    ]
