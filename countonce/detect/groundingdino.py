"""GroundingDINO adapter (fallback detector), written against IDEA-Research/GroundingDINO.

Real API (groundingdino.util.inference): ``load_model(config_path, checkpoint_path)``,
``load_image(path) -> (image_source, image_tensor)``, and
``predict(model, image, caption, box_threshold, text_threshold, device)`` returning
``(boxes, logits, phrases)`` with boxes as normalized cxcywh and logits as the max
sigmoid token score per kept query. We call ``predict`` with a low ``box_threshold`` so
the raw score distribution is stored. Exemplars are unsupported (text only).

torch and groundingdino are imported only inside ``_load``/``_infer``.
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
from countonce.types import Detection, Frame


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
    ) -> None:
        self.config_path = None if config_path is None else Path(config_path)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.version = version
        self.floor = floor
        self.text_threshold = text_threshold
        self.device = device
        self._model: Any = None
        self.compile_mode = (
            "off"  # set by optim.configure_inference; apply with maybe_compile in _load
        )

    def config(self) -> dict[str, object]:
        return {"floor": self.floor, "text_threshold": self.text_threshold}

    def _load(self) -> Any:
        """Model-loading call (needs weights; not exercised in tests)."""
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

    def _infer(self, frame: Frame, caption: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores (n,), boxes cxcywh normalized (n,4))."""
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
            scores, boxes = self._infer(f, caption)
            out.extend(logits_boxes_to_detections(scores, boxes, f, self.floor))
        return out
