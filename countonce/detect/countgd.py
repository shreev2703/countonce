"""CountGD adapter (default detector), written against niki-amini-naieni/CountGD.

Real output format (from the repo's app/inference code, `TODO: verify` against a
checkout once weights are available): the model takes a normalized image, a caption
(lowercase, period-terminated) and optional exemplar boxes, and returns a dict with
``pred_logits`` (1, nq, T) and ``pred_boxes`` (1, nq, 4, normalized cxcywh). A query's
score is ``pred_logits.sigmoid().max(-1)``; the repo's demo keeps queries above 0.23.
We keep everything above a lower floor and let ``filter_by_score`` sweep the rest.
Exemplars are xyxy pixel boxes on the same image the model sees.

If a build returns only points (density-style), ``points_to_detections`` converts them
using exemplar size or a default.

torch and the CountGD repo are imported only inside ``_load``/``_infer``.
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


class CountGDDetector:
    name = "countgd"

    def __init__(
        self,
        repo_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        version: str = "fsc147_best",
        floor: float = 0.05,
        device: str = "cuda",
    ) -> None:
        self.repo_path = None if repo_path is None else Path(repo_path)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.version = version
        self.floor = floor
        self.device = device
        self._model: Any = None

    def config(self) -> dict[str, object]:
        return {"floor": self.floor}

    def _load(self) -> Any:
        """Model-loading call (needs repo checkout + weights; not exercised in tests)."""
        if self._model is None:
            if not (
                self.repo_path
                and self.checkpoint_path
                and self.repo_path.exists()
                and self.checkpoint_path.exists()
            ):
                raise ModelUnavailableError("CountGD repo/checkpoint missing; see docs/GATES.md")
            raise NotImplementedError(
                "CountGD model construction is stubbed until weights are available "
                "(build via the repo's model builder, load checkpoint, move to device)"
            )
        return self._model

    def _infer(
        self, frame: Frame, caption: str, exemplars: Sequence[Exemplar] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (sigmoid logits (nq,T), boxes cxcywh normalized (nq,4))."""
        import torch  # noqa: F401  (real call path needs the repo's transforms)

        model = self._load()
        raise NotImplementedError(f"forward pass for {type(model).__name__} pending weights")

    def detect(
        self,
        frames: Sequence[Frame],
        prompt: str,
        exemplars: Sequence[Exemplar] | None = None,
    ) -> list[Detection]:
        caption = ensure_caption(prompt)
        out: list[Detection] = []
        for f in frames:
            logits, boxes = self._infer(f, caption, exemplars)
            out.extend(logits_boxes_to_detections(logits, boxes, f, self.floor))
        return out
