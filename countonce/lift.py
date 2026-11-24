"""Lift 2D detections into 3D using a dense reconstruction (docs/PLAN.md §8.4).

Region: mask, else central 50% of the box, else a small disk around the point.
``xyz`` is the per-axis median of confident, unmasked ``world_points``; ``size``
is 2D extent x median depth / focal length. Sparse reconstructions (COLMAP) have zero
confidence at most pixels, so the median runs over whatever confident pixels fall in the
region; use a lower ``min_pixels`` and ``conf_percentile=0`` for them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from countonce.types import Detection, Frame, LiftedDetection, Recon


def compute_scene_scale(recon: Recon, conf_percentile: float = 50.0) -> float:
    """Median camera-to-scene distance (confident points only); 1.0 if undefined."""
    rot = recon.extrinsics[:, :, :3]
    trans = recon.extrinsics[:, :, 3]
    centers = -np.einsum("nji,nj->ni", rot, trans)  # -R^T t, world frame
    conf = recon.world_points_conf
    thr = np.percentile(conf, conf_percentile)
    dists = []
    for i in range(recon.n_views):
        sel = (conf[i] >= thr) & (conf[i] > 0) & np.isfinite(recon.world_points[i]).all(-1)
        if sel.any():
            dists.append(np.linalg.norm(recon.world_points[i][sel] - centers[i], axis=1))
    if not dists:
        return 1.0
    scale = float(np.median(np.concatenate(dists)))
    return scale if scale > 0 else 1.0


def frame_affine(frame: Frame | None, h: int, w: int) -> tuple[float, float, float, float]:
    """(sx, sy, ox, oy) taking frame pixels to recon (w x h) pixels.

    Uses the frame's exact `grid_map` (scale + crop/pad offset) when it has one; otherwise a
    plain rescale of the frame size to the recon size.
    """
    if frame is None:
        return 1.0, 1.0, 0.0, 0.0
    if frame.grid_map is not None:
        g = frame.grid_map.at_size(w, h)
        return g.sx, g.sy, g.ox, g.oy
    return w / frame.width, h / frame.height, 0.0, 0.0


def frame_valid_rect(
    frame: Frame | None, h: int, w: int, aff: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1): the part of the recon grid that shows real frame pixels (not padding)."""
    if frame is None:
        return 0.0, 0.0, float(w), float(h)
    sx, sy, ox, oy = aff
    return (
        max(0.0, ox), max(0.0, oy),
        min(float(w), ox + sx * frame.width), min(float(h), oy + sy * frame.height),
    )  # fmt: skip


def _source_index(n: int, scale: float, offset: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame-pixel index for each of ``n`` recon pixels, and which of them fall inside."""
    src = np.floor((np.arange(n) + 0.5 - offset) / scale).astype(int)
    return np.clip(src, 0, size - 1), (src >= 0) & (src < size)


def _region_mask(
    det: Detection,
    h: int,
    w: int,
    aff: tuple[float, float, float, float],
    valid: tuple[float, float, float, float],
    disk_radius: float,
):
    """Boolean HxW region in recon pixel coordinates, plus the 2D extent in px."""
    sx, sy, ox, oy = aff
    ys, xs = np.mgrid[0:h, 0:w]
    if det.mask is not None:
        m = det.mask
        yi, vy = _source_index(h, sy, oy, m.shape[0])
        xi, vx = _source_index(w, sx, ox, m.shape[1])
        m = m[np.ix_(yi, xi)] & vy[:, None] & vx[None, :]
        m = m.astype(bool)
        if not m.any():
            return m, 0.0
        r, c = np.where(m)
        return m, float(max(r.max() - r.min() + 1, c.max() - c.min() + 1))
    if det.box_xyxy is not None:
        x0, y0, x1, y1 = (float(v) for v in det.box_xyxy)
        x0, x1, y0, y1 = x0 * sx + ox, x1 * sx + ox, y0 * sy + oy, y1 * sy + oy
        x0, x1, y0, y1 = max(x0, valid[0]), min(x1, valid[2]), max(y0, valid[1]), min(y1, valid[3])
        if x1 <= x0 or y1 <= y0:
            return np.zeros((h, w), bool), 0.0
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        m = (np.abs(xs + 0.5 - cx) <= bw / 4) & (np.abs(ys + 0.5 - cy) <= bh / 4)
        return m, float(max(bw, bh))
    assert det.point_xy is not None
    px, py = float(det.point_xy[0]) * sx + ox, float(det.point_xy[1]) * sy + oy
    m = (xs + 0.5 - px) ** 2 + (ys + 0.5 - py) ** 2 <= disk_radius**2
    return m, 2.0 * disk_radius


def _outside_recon(det: Detection, aff, valid) -> bool:
    """True when the detection lies wholly in the cropped-away / padded part of the frame."""
    sx, sy, ox, oy = aff
    u0, v0, u1, v1 = valid
    if det.box_xyxy is not None:
        x0, y0, x1, y1 = (float(v) for v in det.box_xyxy)
        return x1 * sx + ox <= u0 or x0 * sx + ox >= u1 or y1 * sy + oy <= v0 or y0 * sy + oy >= v1
    if det.mask is not None:
        return False  # a mask outside the grid is empty and fails as too_few_pixels
    if det.point_xy is not None:
        u, v = float(det.point_xy[0]) * sx + ox, float(det.point_xy[1]) * sy + oy
        return not (u0 <= u < u1 and v0 <= v < v1)
    return False


def lift_detection(
    det: Detection,
    recon: Recon,
    view: int,
    frame: Frame | None = None,
    *,
    min_score: float = 0.0,
    conf_percentile: float = 30.0,
    min_point_conf: float = 0.0,
    min_pixels: int = 5,
    point_disk_radius: float = 3.0,
    depth_spread_scale: float = 0.1,
) -> LiftedDetection:
    """Lift one detection. Never raises on bad input; sets ``fail_reason`` instead."""
    h, w = recon.depth.shape[1:]
    aff = frame_affine(frame, h, w)
    sx, sy, ox, oy = aff

    if det.score < min_score:
        return LiftedDetection(det=det, fail_reason="low_conf")

    valid = frame_valid_rect(frame, h, w, aff)
    if frame is not None and _outside_recon(det, aff, valid):
        return LiftedDetection(det=det, fail_reason="outside_recon")
    region, extent_px = _region_mask(det, h, w, aff, valid, point_disk_radius)
    n_region = int(region.sum())
    if n_region == 0:
        return LiftedDetection(det=det, fail_reason="too_few_pixels")

    if frame is not None and frame.privacy_mask is not None:
        pm = frame.privacy_mask
        yi, _ = _source_index(h, sy, oy, pm.shape[0])
        xi, _ = _source_index(w, sx, ox, pm.shape[1])
        region = region & ~pm[np.ix_(yi, xi)]
        if int(region.sum()) < min_pixels <= n_region:
            return LiftedDetection(det=det, fail_reason="privacy_mask")

    pts = recon.world_points[view]
    conf = recon.world_points_conf[view]
    region = region & np.isfinite(pts).all(-1) & np.isfinite(conf)
    if not region.any():
        return LiftedDetection(det=det, fail_reason="too_few_pixels")

    region = region & (conf > min_point_conf)
    if not region.any():
        return LiftedDetection(det=det, fail_reason="low_conf")
    thr = np.percentile(conf[region], conf_percentile)
    region = region & (conf >= thr)
    if int(region.sum()) < min_pixels:
        return LiftedDetection(det=det, fail_reason="too_few_pixels")

    xyz = np.median(pts[region], axis=0)
    depth = recon.depth[view][region]
    depth = depth[np.isfinite(depth) & (depth > 0)]
    if depth.size == 0:
        return LiftedDetection(det=det, fail_reason="too_few_pixels")
    med_depth = float(np.median(depth))
    focal = float((recon.intrinsics[view, 0, 0] + recon.intrinsics[view, 1, 1]) / 2)
    size = extent_px * med_depth / focal

    spread = float(np.median(np.abs(depth - med_depth)) / med_depth)
    mean_conf = float(conf[region].mean())
    conf_term = mean_conf / (mean_conf + 1.0)  # VGGT confidences are >= 1
    spread_term = 1.0 / (1.0 + spread / depth_spread_scale)
    lift_conf = float(np.clip(det.score, 0.0, 1.0) * conf_term * spread_term)
    return LiftedDetection(det=det, xyz=xyz, size=float(size), lift_conf=lift_conf)


def lift_detections(
    detections: Sequence[Detection],
    recon: Recon,
    frames: Mapping[int, Frame] | None = None,
    frame_to_view: Mapping[int, int] | None = None,
    **kwargs,
) -> list[LiftedDetection]:
    """Lift all detections. ``frame_id`` indexes recon views unless ``frame_to_view`` is given."""
    out = []
    for det in detections:
        view = frame_to_view[det.frame_id] if frame_to_view is not None else det.frame_id
        if not 0 <= view < recon.n_views:
            raise ValueError(f"detection {det.id}: frame {det.frame_id} has no recon view")
        frame = frames.get(det.frame_id) if frames is not None else None
        out.append(lift_detection(det, recon, view, frame, **kwargs))
    return out
