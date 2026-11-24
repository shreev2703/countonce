"""Quality warnings (PLAN 8.9): plain-English message plus a concrete fix for each problem.

Named ``warnings`` inside the package; the stdlib module is unaffected (absolute imports).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from countonce.types import Instance, LiftedDetection, Recon

TOO_FEW_FRAMES = "too_few_frames"
LOW_OVERLAP = "low_overlap"
LOW_DEPTH_CONFIDENCE = "low_depth_confidence"
MANY_FAILED_LIFTS = "many_failed_lifts"
WIDE_RANGE = "wide_range"


@dataclass
class WarningThresholds:
    min_frames: int = 3
    min_multiview_fraction: float = 0.3  # share of instances seen in >= 2 views
    min_depth_conf: float = 1.5  # median depth confidence (VGGT confidences are >= 1)
    max_failed_lift_fraction: float = 0.3
    min_detections_for_lift_check: int = 3
    max_range_width_fraction: float = 0.25  # (high - low) / count
    min_range_width: int = 2  # ignore ranges narrower than this many objects


def _w(code: str, message: str, fix: str) -> dict[str, str]:
    return {"code": code, "message": message, "fix": fix}


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
