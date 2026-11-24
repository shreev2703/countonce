"""Quality warnings (PLAN 8.9): plain-English message plus a concrete fix for each problem.

Named ``warnings`` inside the package; the stdlib module is unaffected (absolute imports).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from countonce.types import Detection, Instance, LiftedDetection, Recon

TOO_FEW_FRAMES = "too_few_frames"
LOW_OVERLAP = "low_overlap"
LOW_DEPTH_CONFIDENCE = "low_depth_confidence"
MANY_FAILED_LIFTS = "many_failed_lifts"
WIDE_RANGE = "wide_range"
NO_OBJECTS_FOUND = "no_objects_found"
NEGATIVES_IGNORED = "negative_prompts_ignored"


@dataclass
class WarningThresholds:
    min_frames: int = 3
    min_multiview_fraction: float = 0.3  # share of instances seen in >= 2 views
    min_depth_conf: float = 1.5  # median depth confidence (VGGT confidences are >= 1)
    max_failed_lift_fraction: float = 0.3
    min_detections_for_lift_check: int = 3
    max_range_width_fraction: float = 0.25  # (high - low) / count
    min_range_width: int = 2  # ignore ranges narrower than this many objects
    small_count: int = 2  # counts up to this with an interval reaching 0 count as 'near zero'
    top_below_threshold: int = 3  # labels listed in the no-objects summary

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, float] | None) -> WarningThresholds:
        """Defaults with ``overrides`` applied; values are cast to each field's type."""
        base = cls()
        out: dict[str, float] = {}
        for k, v in (overrides or {}).items():
            if k not in cls.__dataclass_fields__:
                raise ValueError(f"unknown warning threshold {k!r}")
            out[k] = type(getattr(base, k))(v)
        return cls(**out)  # type: ignore[arg-type]


def _w(code: str, message: str, fix: str) -> dict[str, str]:
    return {"code": code, "message": message, "fix": fix}


def negatives_ignored_warning(detector: str, negatives: Sequence[str]) -> dict[str, str]:
    """Negative phrases were parsed but the detector reports no phrase scores."""
    return _w(
        NEGATIVES_IGNORED,
        f"Detector {detector!r} has no per-phrase scores, so the negative phrases "
        f"{list(negatives)} were not applied.",
        "Use detector 'gdino' for negative-prompt suppression, or review false positives "
        "with the correction tools.",
    )


def assess_quality(
    *,
    n_frames: int,
    count: int,
    instances: Sequence[Instance] = (),
    lifted: Sequence[LiftedDetection] = (),
    recon: Recon | None = None,
    interval_low: int | None = None,
    interval_high: int | None = None,
    thresholds: WarningThresholds | None = None,
) -> list[dict[str, str]]:
    """Return ``{code, message, fix}`` dicts for every quality problem found (possibly empty)."""
    th = thresholds or WarningThresholds()
    out: list[dict[str, str]] = []

    if n_frames < th.min_frames:
        out.append(
            _w(
                TOO_FEW_FRAMES,
                f"Only {n_frames} usable frame(s); objects seen from one angle can't be "
                "reliably told apart from duplicates.",
                "Take 3-5 more photos, moving sideways a little between each.",
            )
        )

    if n_frames >= 2 and len(instances) > 0:
        frac = sum(i.n_views >= 2 for i in instances) / len(instances)
        if frac < th.min_multiview_fraction:
            out.append(
                _w(
                    LOW_OVERLAP,
                    f"Only {frac:.0%} of the counted objects appear in more than one photo, "
                    "so the photos barely overlap and duplicates can't be checked.",
                    "Move less between photos so each object shows up in at least two of them.",
                )
            )

    if recon is not None and recon.depth_conf.size:
        med = float(np.median(recon.depth_conf))
        if med < th.min_depth_conf:
            out.append(
                _w(
                    LOW_DEPTH_CONFIDENCE,
                    f"The 3D reconstruction is uncertain (median depth confidence {med:.2f}); "
                    "reflective, transparent or low-texture surfaces cause this.",
                    "Retake in diffuse light, avoid glare, and include textured background.",
                )
            )

    if len(lifted) >= th.min_detections_for_lift_check:
        failed = sum(not ld.ok for ld in lifted) / len(lifted)
        if failed > th.max_failed_lift_fraction:
            out.append(
                _w(
                    MANY_FAILED_LIFTS,
                    f"{failed:.0%} of detections could not be placed in 3D and were set aside.",
                    "Retake with sharper photos, closer to the objects, with more overlap "
                    "between views.",
                )
            )

    if interval_low is not None and interval_high is not None:
        width = interval_high - interval_low
        if width >= th.min_range_width and width > th.max_range_width_fraction * max(count, 1):
            out.append(
                _w(
                    WIDE_RANGE,
                    f"The count is sensitive to settings: it ranges {interval_low}-"
                    f"{interval_high} around {count}.",
                    "Add photos from new angles, or check the overlays for missed and "
                    "duplicate objects.",
                )
            )
    return out


def below_threshold_summary(
    raw_detections: Sequence[Detection], threshold: float, prompt: str, top: int = 3
) -> list[dict[str, object]]:
    """Top (label, best score, n) among detections scored below ``threshold``.

    Detectors without class labels are reported under the prompt itself.
    """
    best: dict[str, list[float]] = {}
    for d in raw_detections:
        if d.score < threshold:
            best.setdefault(d.label or prompt, []).append(float(d.score))
    rows: list[dict[str, object]] = [
        {"label": k, "best_score": round(max(v), 3), "n": len(v)} for k, v in best.items()
    ]
    rows.sort(key=lambda r: -float(r["best_score"]))  # type: ignore[arg-type]
    return rows[:top]


def zero_count_warning(
    *,
    prompt: str,
    count: int,
    interval_low: int,
    raw_detections: Sequence[Detection] = (),
    threshold: float = 0.0,
    thresholds: WarningThresholds | None = None,
) -> tuple[dict[str, str] | None, list[dict[str, object]]]:
    """PLAN 9.8b: a ``no_objects_found`` warning when the count is 0, or small with 0 in range.

    Returns ``(warning or None, below-threshold summary)``.
    """
    th = thresholds or WarningThresholds()
    if not (count == 0 or (count <= th.small_count and interval_low <= 0)):
        return None, []
    summary = below_threshold_summary(raw_detections, threshold, prompt, th.top_below_threshold)
    msg = f"No {prompt} found."
    if summary:
        seen = ", ".join(
            f"'{r['label']}' (best score {float(r['best_score']):.2f}, {r['n']}x)"  # type: ignore[arg-type]
            for r in summary
        )
        msg += f" Only weak detections below the threshold {threshold:.2f}: {seen}."
    else:
        msg += " Nothing was detected, not even weakly."
    return _w(
        NO_OBJECTS_FOUND,
        msg,
        "Check the prompt wording and that the objects are in the photos; "
        "if they are, retake closer and sharper.",
    ), summary
