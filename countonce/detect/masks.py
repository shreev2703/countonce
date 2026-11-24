"""Instance-mask producers for mask lifting (PLAN 8.2 optional ``masks.py``, 9.6 ablation).

``lift.py`` uses ``Detection.mask`` when ``Config.lift_region == "mask"``; this module fills it.
``BoxFillMaskProducer`` is deterministic and dependency-free (every pixel in the box). The SAM
adapter imports ``sam2`` lazily and needs a checkpoint path from config; nothing is downloaded.
Masks are cached per frame, keyed by frame hash, detector, mask model, version and the boxes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from countonce.errors import CountOnceError
from countonce.types import Detection, Frame


class MaskModelUnavailable(CountOnceError):
    """The mask model's package or weights are missing."""


@runtime_checkable
class MaskProducer(Protocol):
    name: str
    version: str

    def masks(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> list[np.ndarray]:
        """One HxW bool mask per xyxy pixel box, in order."""
        ...


def box_to_mask(box: Sequence[float], height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    m = np.zeros((height, width), bool)
    xa, xb = max(int(np.floor(x0)), 0), min(int(np.ceil(x1)), width)
    ya, yb = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)), height)
    if xb > xa and yb > ya:
        m[ya:yb, xa:xb] = True
    return m


class BoxFillMaskProducer:
    """Deterministic producer: the mask is the box rectangle. Used for tests and the ablation."""

    name = "box_fill"
    version = "1"

    def masks(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> list[np.ndarray]:
        h, w = image.shape[:2]
        return [box_to_mask(b, h, w) for b in boxes]


class SamMaskProducer:
    """SAM 2 box-prompted masks. ``sam2`` and the checkpoint are required, loaded lazily."""

    name = "sam2"

    def __init__(self, checkpoint: Path | str | None, model_cfg: str = "sam2_hiera_s.yaml") -> None:
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.model_cfg = model_cfg
        self.version = model_cfg
        self._predictor: object | None = None

    def _load(self) -> object:
        if self._predictor is not None:
            return self._predictor
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise MaskModelUnavailable(
                f"SAM weights not found at {self.checkpoint!r}; set Config.mask_checkpoint to a "
                "SAM 2 checkpoint such as sam2_hiera_small.pt (see docs/GATES.md)"
            )
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise MaskModelUnavailable("the 'sam2' package is not installed") from e
        self._predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, str(self.checkpoint)))
        return self._predictor

    def masks(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> list[np.ndarray]:
        if not len(boxes):
            return []
        pred = self._load()
        pred.set_image(image)  # type: ignore[attr-defined]
        out: list[np.ndarray] = []
        for b in boxes:
            m, _, _ = pred.predict(  # type: ignore[attr-defined]
                box=np.asarray(b, dtype=np.float32), multimask_output=False
            )
            out.append(np.asarray(m[0]).astype(bool))
        return out


def build_mask_producer(mode: str, checkpoint: Path | str | None = None) -> MaskProducer | None:
    if mode == "none":
        return None
    if mode == "box_fill":
        return BoxFillMaskProducer()
    if mode == "sam":
        return SamMaskProducer(checkpoint)
    raise ValueError(f"unknown masks mode {mode!r}")


def mask_cache_key(
    frame: Frame,
    detector: str,
    producer: MaskProducer,
    boxes: Sequence[Sequence[float]],
) -> str:
    payload = json.dumps(
        {
            "frame": frame.sha256,
            "detector": detector,
            "mask_model": producer.name,
            "mask_version": producer.version,
            "boxes": [[round(float(v), 3) for v in b] for b in boxes],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class MaskCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str) -> list[np.ndarray] | None:
        try:
            with np.load(self.path(key)) as z:
                h, w = (int(v) for v in z["shape"])
                n = int(z["n"])
                bits = z["bits"]
                return [
                    np.unpackbits(bits[i], count=h * w).reshape(h, w).astype(bool) for i in range(n)
                ]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: str, masks: Sequence[np.ndarray], shape: tuple[int, int]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        bits = np.stack([np.packbits(m.reshape(-1)) for m in masks]) if masks else np.zeros((0, 0))
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, shape=np.array(shape), n=len(masks), bits=bits)
        os.replace(tmp, self.path(key))


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def attach_masks(
    detections: Sequence[Detection],
    frames: Sequence[Frame],
    producer: MaskProducer,
    detector: str,
    cache: MaskCache | None = None,
    stats: dict[str, int] | None = None,
) -> list[Detection]:
    """Return detections with ``mask`` set (boxed detections that lack one), order preserved."""
    by_frame = {f.id: f for f in frames}
    new: dict[str, np.ndarray] = {}
    groups: dict[int, list[Detection]] = {}
    for d in detections:
        if d.mask is None and d.box_xyxy is not None and d.frame_id in by_frame:
            groups.setdefault(d.frame_id, []).append(d)
    hits = misses = 0
    for fid, ds in groups.items():
        frame = by_frame[fid]
        boxes = [d.box_xyxy.tolist() for d in ds if d.box_xyxy is not None]
        key = mask_cache_key(frame, detector, producer, boxes)
        ms = cache.get(key) if cache is not None else None
        if ms is not None and len(ms) == len(ds):
            hits += 1
        else:
            misses += 1
            ms = producer.masks(_load_rgb(frame.path), boxes)
            if len(ms) != len(ds):
                raise ValueError(f"mask producer returned {len(ms)} masks for {len(ds)} boxes")
            if cache is not None:
                cache.put(key, ms, (frame.height, frame.width))
        for d, m in zip(ds, ms, strict=True):
            new[d.id] = m
    if stats is not None:
        stats["hits"] = stats.get("hits", 0) + hits
        stats["misses"] = stats.get("misses", 0) + misses
    return [replace(d, mask=new[d.id]) if d.id in new else d for d in detections]
