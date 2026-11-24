"""Detector interface (PLAN section 8.2), box helpers and a deterministic fake for tests.

Detectors return *all* raw detections with scores; thresholds are applied later with
``filter_by_score`` so they can be swept without rerunning the model. Nothing here
imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from countonce.types import Detection, Frame


@dataclass(frozen=True)
class Exemplar:
    """A visual exemplar: an xyxy pixel box on frame ``frame_id``."""

    frame_id: int
    box_xyxy: tuple[float, float, float, float]


@runtime_checkable
class Detector(Protocol):
    name: str
    version: str

    def config(self) -> dict[str, object]:
        """Settings that change the raw output (part of the cache key)."""
        ...

    def detect(
        self,
        frames: Sequence[Frame],
        prompt: str,
        exemplars: Sequence[Exemplar] | None = None,
    ) -> list[Detection]:
        """Raw detections for all frames, unthresholded beyond a low model floor."""
        ...


def filter_by_score(dets: Sequence[Detection], threshold: float) -> list[Detection]:
    return [d for d in dets if d.score >= threshold]


def cxcywh_norm_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalized (cx, cy, w, h) -> pixel xyxy, clipped to the image."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * width, b[:, 1] * height, b[:, 2] * width, b[:, 3] * height
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def default_box_size(
    width: int,
    height: int,
    exemplars: Sequence[Exemplar] | None = None,
    frame_id: int | None = None,
) -> tuple[float, float]:
    """(w, h) for point->box: median exemplar size, else 5% of the shorter image side."""
    boxes = [e.box_xyxy for e in exemplars or [] if frame_id is None or e.frame_id == frame_id]
    if not boxes:
        boxes = [e.box_xyxy for e in exemplars or []]
    if boxes:
        a = np.asarray(boxes, dtype=np.float64)
        return float(np.median(a[:, 2] - a[:, 0])), float(np.median(a[:, 3] - a[:, 1]))
    s = 0.05 * min(width, height)
    return s, s


def points_to_boxes(
    points_xy: np.ndarray, size_wh: tuple[float, float], width: int, height: int
) -> np.ndarray:
    """Centre fixed-size boxes on points, clipped to the image. Returns (N,4) xyxy."""
    p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    w, h = size_wh
    out = np.stack([p[:, 0] - w / 2, p[:, 1] - h / 2, p[:, 0] + w / 2, p[:, 1] + h / 2], axis=1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def density_to_points(
    density: np.ndarray, min_value: float = 0.0, window: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of an (H,W) density map -> ((N,2) xy points, (N,) peak values).

    A pixel is a peak if it is the maximum of its ``window`` x ``window`` neighbourhood
    (ties broken toward the first in raster order) and exceeds ``min_value``.
    """
    d = np.asarray(density, dtype=np.float64)
    if d.ndim != 2 or window < 1 or window % 2 == 0:
        raise ValueError("density must be 2D and window a positive odd int")
    r = window // 2
    padded = np.pad(d, r, constant_values=-np.inf)
    is_max = np.ones(d.shape, dtype=bool)
    h, w = d.shape
    for dy in range(window):
        for dx in range(window):
            if dy == r and dx == r:
                continue
            nb = padded[dy : dy + h, dx : dx + w]
            before = (dy, dx) < (r, r)  # earlier in raster order wins ties
            is_max &= (d > nb) if before else (d >= nb)
    is_max &= d > min_value
    yx = np.argwhere(is_max)  # row-major, same order as nonzero
    ys, xs = yx[:, 0], yx[:, 1]
    return np.stack([xs, ys], axis=1).astype(np.float64), d[ys, xs]


def points_to_detections(
    frame: Frame,
    points_xy: np.ndarray,
    scores: np.ndarray,
    exemplars: Sequence[Exemplar] | None = None,
    start_k: int = 0,
) -> list[Detection]:
    """Points-only output -> detections with both the point and an estimated box."""
    size = default_box_size(frame.width, frame.height, exemplars, frame.id)
    boxes = points_to_boxes(points_xy, size, frame.width, frame.height)
    return [
        Detection(
            id=Detection.make_id(frame.id, start_k + k),
            frame_id=frame.id,
            score=float(s),
            box_xyxy=boxes[k],
            point_xy=np.asarray(points_xy[k], dtype=np.float64),
        )
        for k, s in enumerate(scores)
    ]


class FakeDetector:
    """Deterministic detector for tests. ``script`` maps frame_id -> [(x0,y0,x1,y1,score)]."""

    name = "fake"
    version = "0"

    def __init__(
        self, script: dict[int, list[tuple[float, float, float, float, float]]] | None = None
    ) -> None:
        self.script = script or {}
        self.calls = 0

    def config(self) -> dict[str, object]:
        return {"script": {str(k): v for k, v in sorted(self.script.items())}}

    def detect(
        self,
        frames: Sequence[Frame],
        prompt: str,
        exemplars: Sequence[Exemplar] | None = None,
    ) -> list[Detection]:
        self.calls += 1
        out: list[Detection] = []
        for f in frames:
            for k, (x0, y0, x1, y1, s) in enumerate(self.script.get(f.id, [])):
                out.append(
                    Detection(
                        id=Detection.make_id(f.id, k),
                        frame_id=f.id,
                        score=s,
                        box_xyxy=np.array([x0, y0, x1, y1], dtype=np.float64),
                    )
                )
        return out
