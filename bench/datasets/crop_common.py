"""Shared single-image type and YOLO/mask helpers for the crop loaders (PLAN §9.1 Track D).

``CropImage`` is duck-compatible with ``bench.datasets.fsc147.FscImage`` (``name``, ``path``,
``class_name``, ``points_xy``, ``exemplar_boxes_xyxy``, ``split``, ``count``) so the detector
sanity harness and the shared validation treat both alike. Points are box / mask centres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class CropImage:
    name: str
    path: Path
    class_name: str  # text prompt for the detector
    points_xy: np.ndarray  # (N,2) instance centres in pixels
    boxes_xyxy: np.ndarray  # (N,4) pixels; zeros((0,4)) when the set has no boxes
    split: str
    dataset: str
    mask_path: Path | None = None
    exemplar_boxes_xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    @property
    def count(self) -> int:
        return int(len(self.points_xy))


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the header only."""
    with Image.open(path) as im:
        return im.size


def read_yolo(label_file: Path, size: tuple[int, int]) -> np.ndarray:
    """YOLO text (``class cx cy w h``, normalised) -> (N,4) xyxy pixel boxes.

    TODO: verify column order and that coordinates are normalised for each dataset.
    A missing file means no annotation (empty array).
    """
    if not label_file.exists():
        return np.zeros((0, 4))
    rows = [ln.split() for ln in label_file.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.asarray([[float(v) for v in r[1:5]] for r in rows])
    w, h = size
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def box_centres(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros((0, 2))
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def instance_centroids(mask_path: Path) -> np.ndarray:
    """Centroids of the instances in an instance-id mask (0 = background, one id per instance).

    TODO: verify that MinneApple masks encode one id per instance (not one value per class).
    """
    with Image.open(mask_path) as im:
        m = np.asarray(im)
    if m.ndim == 3:
        m = m[..., 0]
    pts = []
    for v in np.unique(m):
        if v == 0:
            continue
        ys, xs = np.nonzero(m == v)
        pts.append([xs.mean(), ys.mean()])
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def check_root(root: Path, script: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/{script} first")


def finish(
    dataset: str,
    items: list[CropImage],
    split: str | None,
    limit: int | None,
    validate: bool,
    quarantine_path: str | Path | None,
) -> list[CropImage]:
    if split is not None:
        items = [i for i in items if i.split == split]
    if validate:
        from bench.datasets.validate import filter_scenes

        items, _ = filter_scenes(dataset, items, quarantine_path)
    items.sort(key=lambda i: i.name)
    return items[:limit] if limit is not None else items
