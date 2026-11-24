"""Nearest-GT projection matching (PLAN §9.1): label each detection with a ground-truth object id.

GT 3D positions are projected into each frame with that frame's camera; a detection is assigned
the nearest projected GT point within a pixel threshold, else ``None`` (unmatched / false positive).
The output feeds ``bench.metrics`` (duplicate rate, miss rate, pairwise F1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from countonce.types import Detection


def project_points(
    xyz: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project (M,3) world points with K (3,3) and world-to-camera [R|t] (3,4).

    Returns (uv (M,2), depth (M,)); points behind the camera have depth <= 0.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    ext = np.asarray(extrinsics, dtype=np.float64)
    cam = xyz @ ext[:, :3].T + ext[:, 3]
    z = cam[:, 2]
    uvw = cam @ np.asarray(intrinsics, dtype=np.float64).T
    w = np.where(np.abs(uvw[:, 2]) < 1e-12, 1e-12, uvw[:, 2])
    return uvw[:, :2] / w[:, None], z


def detection_point(det: Detection) -> np.ndarray | None:
    """Reference pixel of a detection: its point, else the box centre."""
    if det.point_xy is not None:
        return np.asarray(det.point_xy, dtype=np.float64)
    if det.box_xyxy is not None:
        b = np.asarray(det.box_xyxy, dtype=np.float64)
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return None


def match_detections_to_gt(
    detections: Sequence[Detection],
    gt_ids: Sequence[str],
    gt_xyz: np.ndarray,
    intrinsics: dict[int, np.ndarray],
    extrinsics: dict[int, np.ndarray],
    image_size: dict[int, tuple[int, int]],
    threshold_px: float = 20.0,
) -> dict[str, str | None]:
    """Map ``Detection.id`` -> GT id (or ``None``) by nearest projected GT within ``threshold_px``.

    ``intrinsics``/``extrinsics``/``image_size`` (width, height) are keyed by ``frame_id``. GT
    points behind the camera or outside the image are not candidates. Occlusion is not modelled
    (no depth for GT); several detections in one frame may map to the same GT, which is exactly
    a within-frame duplicate. Ties break on the lower GT index; detections with no usable
    reference point or a frame without a camera map to ``None``.
    """
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    if len(gt_ids) != len(gt_xyz):
        raise ValueError("gt_ids and gt_xyz length mismatch")
    out: dict[str, str | None] = {}
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_id, []).append(d)
    for fid, dets in by_frame.items():
        has_cam = fid in intrinsics and fid in extrinsics and fid in image_size
        if not has_cam or not len(gt_xyz):
            out.update({d.id: None for d in dets})
            continue
        uv, z = project_points(gt_xyz, intrinsics[fid], extrinsics[fid])
        w, h = image_size[fid]
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        cand = np.flatnonzero(ok)
        for d in dets:
            p = detection_point(d)
            if p is None or not len(cand):
                out[d.id] = None
                continue
            dist = np.linalg.norm(uv[cand] - p, axis=1)
            j = int(np.argmin(dist))  # argmin returns the first minimum -> lowest GT index
            out[d.id] = str(gt_ids[cand[j]]) if dist[j] <= threshold_px else None
    return out
