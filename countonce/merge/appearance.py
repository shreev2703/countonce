"""Pluggable appearance features for the v2 associator (docs/PLAN.md §8.6 step 2).

``FeatureExtractor`` is the interface; ``DinoV2Extractor`` imports torch lazily and is
never needed by the tests (they use in-memory fakes).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from countonce.types import Detection, Frame


@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, frame: Frame | None, dets: Sequence[Detection]) -> np.ndarray:
        """Return an ``(len(dets), d)`` array of appearance embeddings."""
        ...


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class DinoV2Extractor:
    """DINOv2 CLS embedding of each detection crop (needs torch + weights; lazy import)."""

    def __init__(self, model: str = "dinov2_vits14", device: str = "cpu", crop_size: int = 224):
        self.model_name = model
        self.device = device
        self.crop_size = crop_size
        self._model: Any = None

    def _load(self) -> Any:  # pragma: no cover - needs torch and downloaded weights
        import torch

        if self._model is None:
            self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self._model.eval().to(self.device)
        return self._model

    def extract(  # pragma: no cover - needs torch and weights
        self, frame: Frame | None, dets: Sequence[Detection]
    ) -> np.ndarray:
        import torch
        from PIL import Image

        if frame is None:
            raise ValueError("DinoV2Extractor needs the Frame (image path) for each detection")
        model = self._load()
        img = Image.open(frame.path).convert("RGB")
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crops = []
        for d in dets:
            if d.box_xyxy is not None:
                x0, y0, x1, y1 = (float(v) for v in d.box_xyxy)
            elif d.point_xy is not None:
                px, py = (float(v) for v in d.point_xy)
                x0, y0, x1, y1 = px - 16, py - 16, px + 16, py + 16
            elif d.mask is not None:
                ys, xs = np.nonzero(d.mask)
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            else:
                raise ValueError(f"detection {d.id} has no box, point or mask")
            box = (
                max(0, int(x0)),
                max(0, int(y0)),
                min(img.width, int(x1) + 1),
                min(img.height, int(y1) + 1),
            )
            crop = img.crop(box).resize((self.crop_size, self.crop_size))
            arr = (np.asarray(crop, dtype=np.float32) / 255.0 - mean) / std
            crops.append(arr.transpose(2, 0, 1))
        with torch.no_grad():
            x = torch.from_numpy(np.stack(crops)).to(self.device)
            out: np.ndarray = model(x).cpu().numpy()
            return out
