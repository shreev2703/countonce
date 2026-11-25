"""GroundingDINO adapter (fallback detector) with two backends.

``backend="hf"`` (default when no ``config_path``/``checkpoint_path`` is given) runs
``IDEA-Research/grounding-dino-tiny`` (public, Apache-2.0) through Hugging Face ``transformers``
(``AutoProcessor`` + ``AutoModelForZeroShotObjectDetection``).
``post_process_grounded_object_detection`` returns xyxy pixel boxes, the max-over-tokens sigmoid
score per query and text labels; we call it with a low ``threshold=floor`` so the raw score
distribution is stored.

``backend="groundingdino"`` is the original IDEA-Research/GroundingDINO package path
(``load_model``, ``load_image``, ``predict`` returning normalized cxcywh boxes and logits).

Exemplars are unsupported (text only). torch, transformers and groundingdino are imported only
inside ``_load``/``_infer``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from countonce.detect._grounding import (
    ModelUnavailableError,
    ensure_caption,
    logits_boxes_to_detections,
)
from countonce.detect.base import Exemplar
from countonce.optim import CompileMode
from countonce.types import Detection, Frame

HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
HF_SIZE_NAMES = {"hf", "groundingdino"}


def xyxy_px_to_cxcywh_norm(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel xyxy (n,4) -> normalized cxcywh (n,4), the input convention of the shared mapper."""
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    cx = (b[:, 0] + b[:, 2]) / 2 / width
    cy = (b[:, 1] + b[:, 3]) / 2 / height
    return np.stack([cx, cy, (b[:, 2] - b[:, 0]) / width, (b[:, 3] - b[:, 1]) / height], axis=1)


def cached_revision(model_id: str) -> str | None:
    """Commit hash of the locally cached snapshot (``refs/main``), without network access."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    ref = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


class GroundingDinoDetector:
    name = "groundingdino"
    supports_negatives = True  # phrase-conditioned scores: see countonce.detect.prompts

    def __init__(
        self,
        config_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        version: str = "swint_ogc",
        floor: float = 0.05,
        text_threshold: float = 0.25,
        device: str = "cuda",
        backend: str | None = None,
        model_id: str = HF_MODEL_ID,
        revision: str | None = None,
    ) -> None:
        if backend is None:
            backend = "groundingdino" if (config_path or checkpoint_path) else "hf"
        if backend not in HF_SIZE_NAMES:
            raise ValueError(f"backend must be 'hf' or 'groundingdino', got {backend!r}")
        self.backend = backend
        self.model_id = model_id
        self.revision = revision
        self._processor: Any = None
        self.config_path = None if config_path is None else Path(config_path)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._version = version
        self._version_override: str | None = None
        self.floor = floor
        self.text_threshold = text_threshold
        self.device = device
        self._model: Any = None
        # set by optim.configure_inference; applied with maybe_compile in _load
        self.compile_mode: CompileMode = "off"

    @property
    def version(self) -> str:
        """Detector version for the cache key; the HF backend pins model id plus revision."""
        if self.backend != "hf" or self._version_override is not None:
            return self._version_override or self._version
        rev = self.revision or cached_revision(self.model_id) or self._loaded_revision()
        return f"hf:{self.model_id}@{rev or 'unresolved'}"

    @version.setter
    def version(self, value: str) -> None:
        self._version_override = value

    def _loaded_revision(self) -> str | None:
        if self._model is None:
            return None
        return getattr(getattr(self._model, "config", None), "_commit_hash", None)

    def config(self) -> dict[str, object]:
        cfg: dict[str, object] = {"floor": self.floor, "text_threshold": self.text_threshold}
        if self.backend == "hf":
            cfg["backend"] = "hf"
        return cfg

    def _load_hf(self) -> Any:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        rev = self.revision or cached_revision(self.model_id)
        kw: dict[str, Any] = {} if rev is None else {"revision": rev}
        self._processor = AutoProcessor.from_pretrained(self.model_id, **kw)  # type: ignore[no-untyped-call]
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, **kw)
        model = model.to(self.device).eval()
        from countonce.optim import maybe_compile

        return maybe_compile(model, self.compile_mode)

    def _load(self) -> Any:
        """Model-loading call (needs weights; not exercised in tests)."""
        if self._model is None and self.backend == "hf":
            self._model = self._load_hf()
        if self._model is None:
            if not (
                self.config_path
                and self.checkpoint_path
                and self.config_path.exists()
                and self.checkpoint_path.exists()
            ):
                raise ModelUnavailableError(
                    "GroundingDINO config/checkpoint missing; see docs/GATES.md"
                )
            from groundingdino.util.inference import load_model

            self._model = load_model(str(self.config_path), str(self.checkpoint_path))
        return self._model

    def _infer_hf(self, frame: Frame, caption: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Returns (scores (n,), boxes xyxy pixels (n,4), text labels)."""
        import torch
        from PIL import Image

        model = self._load()
        with Image.open(frame.path) as im:
            image = im.convert("RGB")
        inputs = self._processor(images=image, text=caption, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        res = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.floor,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        scores = np.asarray(res["scores"].detach().float().cpu().numpy(), dtype=np.float64)
        boxes = np.asarray(res["boxes"].detach().float().cpu().numpy(), dtype=np.float64)
        labels = [str(t) for t in res.get("text_labels", [""] * len(scores))]
        # The real model regresses boxes that overshoot the border by a fraction of a pixel.
        boxes = boxes.reshape(-1, 4)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.height)
        ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        scores, boxes = scores[ok], boxes[ok]
        labels = [lb for lb, k in zip(labels, ok, strict=True) if k]
        return scores, boxes, labels

    def _infer(self, frame: Frame, caption: str) -> tuple[np.ndarray, np.ndarray]:
        """Package backend. Returns (scores (n,), boxes cxcywh normalized (n,4))."""
        from groundingdino.util.inference import (
            load_image,
            predict,
        )

        _, image = load_image(str(frame.path))
        boxes, logits, _phrases = predict(
            self._load(),
            image,
            caption,
            box_threshold=self.floor,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        return logits.cpu().numpy(), boxes.cpu().numpy()

    def detect(
        self,
        frames: Sequence[Frame],
        prompt: str,
        exemplars: Sequence[Exemplar] | None = None,
    ) -> list[Detection]:
        caption = ensure_caption(prompt)
        out: list[Detection] = []
        for f in frames:
            if self.backend == "hf":
                hs, hb, labels = self._infer_hf(f, caption)
                cx = xyxy_px_to_cxcywh_norm(hb, f.width, f.height)
                dets = logits_boxes_to_detections(hs, cx, f, self.floor)
                keep = np.nonzero(hs >= self.floor)[0]
                for d, i in zip(dets, keep, strict=True):
                    d.label = labels[i] or None
                    d.box_xyxy = np.asarray(hb[i], dtype=np.float64)  # exact pixels, no round trip
                out.extend(dets)
                continue
            scores, boxes = self._infer(f, caption)
            out.extend(logits_boxes_to_detections(scores, boxes, f, self.floor))
        return out
