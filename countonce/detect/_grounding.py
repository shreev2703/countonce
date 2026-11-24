"""Shared post-processing for GroundingDINO-family models (numpy only)."""

from __future__ import annotations

import numpy as np

from countonce.detect.base import cxcywh_norm_to_xyxy
from countonce.types import Detection, Frame


class ModelUnavailableError(RuntimeError):
    """Weights/config for a real detector are not present (see docs/GATES.md)."""


def ensure_caption(prompt: str) -> str:
    """GroundingDINO-style captions are lowercase and end with a period."""
    p = prompt.strip().lower()
    return p if p.endswith(".") else p + " ."


def logits_boxes_to_detections(
    logits: np.ndarray, boxes_cxcywh: np.ndarray, frame: Frame, floor: float
) -> list[Detection]:
    """Per-query token logits (nq, T) post-sigmoid + normalized boxes (nq, 4) -> detections.

    The score of a query is its max over text tokens (the rule both CountGD's app and
    GroundingDINO's ``predict`` use). Everything with score >= ``floor`` is kept so
    thresholds can be swept later.
    """
    lg = np.asarray(logits, dtype=np.float64)
    if lg.ndim == 1:
        lg = lg[:, None]
    scores = lg.max(axis=1)
    keep = np.nonzero(scores >= floor)[0]
    xyxy = cxcywh_norm_to_xyxy(np.asarray(boxes_cxcywh)[keep], frame.width, frame.height)
    return [
        Detection(
            id=Detection.make_id(frame.id, k),
            frame_id=frame.id,
            score=float(scores[i]),
            box_xyxy=xyxy[k],
        )
        for k, i in enumerate(keep)
    ]
