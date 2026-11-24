"""Natural-language prompts with negatives (PLAN 15, 9.8b): "red apples, not leaves".

``parse_prompt`` splits a prompt into positive and negative phrases. Detectors that report
phrase-conditioned scores (``supports_negatives = True``, e.g. GroundingDINO) get a
suppression step: the negative phrases are run as a second caption and a positive box is
dropped when a negative box on the same object scores higher by more than a margin.
Numpy only; nothing here imports torch.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from countonce.detect._grounding import ensure_caption
from countonce.types import Detection, Frame

log = logging.getLogger(__name__)

_MARKER = re.compile(
    r"(?:^|[\s,;])(?:but\s+not|and\s+not|not|excluding|except|without)\s+", re.IGNORECASE
)
_LIST_SPLIT = re.compile(r"\s*(?:,|;|\bor\b|\band\b)\s*", re.IGNORECASE)
NEG_IOU = 0.5  # a negative box is "the same object" above this IoU


@dataclass(frozen=True)
class ParsedPrompt:
    positives: list[str]
    negatives: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def has_negatives(self) -> bool:
        return bool(self.negatives)

    @property
    def positive_prompt(self) -> str:
        """Text sent to the detector: the original prompt unless negatives were stripped."""
        return " . ".join(self.positives) if self.negatives else self.raw

    @property
    def negative_prompt(self) -> str:
        return " . ".join(self.negatives)


def _phrases(text: str) -> list[str]:
    return [p.strip(" .") for p in _LIST_SPLIT.split(text) if p.strip(" .")]


def parse_prompt(text: str) -> ParsedPrompt:
    """Split "A, not B" / "A but not B" / "A excluding B" into positives and negatives.

    Prompts without a negation marker come back unchanged (one positive: the stripped text).
    """
    raw = text.strip()
    marks = list(_MARKER.finditer(raw))
    if not marks:
        return ParsedPrompt([raw], [], raw)
    head = raw[: marks[0].start()].strip(" ,;")
    positives = [p.strip() for p in re.split(r"\s*[,;]\s*", head) if p.strip()]
    if not positives:
        raise ValueError(f"prompt {text!r} has only negative phrases; name what to count")
    negatives: list[str] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        negatives.extend(_phrases(raw[m.end() : end]))
    return ParsedPrompt(positives, negatives, raw)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def suppress_negatives(
    positive: Sequence[Detection],
    negative: Sequence[Detection],
    margin: float,
    iou: float = NEG_IOU,
) -> list[Detection]:
    """Drop a positive box whose best overlapping negative score beats it by more than margin."""
    kept: list[Detection] = []
    for d in positive:
        best = 0.0
        if d.box_xyxy is not None:
            for n in negative:
                if (
                    n.frame_id == d.frame_id
                    and n.box_xyxy is not None
                    and _iou(d.box_xyxy, n.box_xyxy) >= iou
                ):
                    best = max(best, n.score)
        if best - d.score > margin:
            continue
        kept.append(d)
    return kept


def negatives_supported(detector: Any) -> bool:
    return bool(getattr(detector, "supports_negatives", False))


def detect_with_negatives(
    detector: Any,
    frames: Sequence[Frame],
    parsed: ParsedPrompt,
    margin: float,
) -> list[Detection]:
    """Detect the positive phrases, then suppress boxes better explained by a negative phrase."""
    pos = detector.detect(frames, parsed.positive_prompt)
    if not parsed.has_negatives:
        return list(pos)
    neg = detector.detect(frames, ensure_caption(parsed.negative_prompt))
    return suppress_negatives(pos, neg, margin)
