"""Adapter from the ported ``recon.result.Recon`` to the dense section-7 ``types.Recon``.

torch and pycolmap are imported only inside ``load_method`` so this module (and the
tests that use it) work on a CPU-only install.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from countonce.lift import compute_scene_scale
from countonce.oom import RungUnavailable
from countonce.recon.base import ReconMethod
from countonce.recon.result import Recon as SparseRecon
from countonce.types import Frame
from countonce.types import Recon as DenseRecon

BACKBONES = ("vggt", "colmap")


class ReconAdaptError(ValueError):
    """The method result cannot be expressed as a dense reconstruction."""


def unproject_depth(
    depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> np.ndarray:
    """Depth maps (N,H,W) -> world points (N,H,W,3); extrinsics are world->camera [R|t].

    Rows with non-finite depth or camera parameters (unregistered views) become NaN.
    """
    n, h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    pix = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H,W,3)
    out = np.full((n, h, w, 3), np.nan, dtype=np.float64)
    for i in range(n):
        k, e = intrinsics[i], extrinsics[i]
        if not (np.isfinite(k).all() and np.isfinite(e).all()):
            continue
        rays = pix @ np.linalg.inv(k).T  # (H,W,3)
        cam = rays * depth[i][..., None]
        rot, trans = e[:, :3], e[:, 3]
        out[i] = (cam - trans) @ rot  # R^T (x - t), row-vector form
    return out.astype(np.float32)


def sparse_to_dense(
    res: SparseRecon, backbone: str | None = None, image_hw: tuple[int, int] | None = None
) -> DenseRecon:
    """Project a sparse point cloud into per-view sparse depth / world-point maps.

    Each triangulated point is splatted to one pixel in every registered view that observes
    it (``points_views``; all registered views if the track is unknown), keeping the nearest
    point per pixel. Confidence is ``track_length / (1 + reprojection_error)`` (1 / (1 + err)
    if the track length is unknown); it is exactly 0 wherever no point lands and for
    unregistered views, so ``lift`` ignores those pixels.
    """
    if res.points is None or res.extrinsics is None or res.intrinsics is None:
        raise ReconAdaptError(f"method {res.method!r} has no sparse points or cameras")
    hw = image_hw or res.image_hw
    if hw is None:
        raise ReconAdaptError("sparse reconstruction has no image size (image_hw)")
    h, w = hw
    n = res.extrinsics.shape[0]
    registered = np.ones(n, dtype=bool) if res.registered is None else res.registered
    pts = np.asarray(res.points, dtype=np.float64)
    m = len(pts)
    err = np.zeros(m) if res.points_error is None else np.asarray(res.points_error, np.float64)
    tl = np.ones(m) if res.points_track_len is None else np.asarray(res.points_track_len, float)
    pconf = np.maximum(tl, 1.0) / (1.0 + np.nan_to_num(err, nan=0.0))

    finite_pts = np.isfinite(pts).all(axis=1)
    per_view: list[list[int]] | None = None
    if res.points_views is not None:
        per_view = [[] for _ in range(n)]
        for j, vs in enumerate(res.points_views):
            for v in vs.tolist():
                if 0 <= v < n:
                    per_view[v].append(j)

    depth = np.zeros((n, h, w), dtype=np.float32)
    conf = np.zeros((n, h, w), dtype=np.float32)
    world = np.zeros((n, h, w, 3), dtype=np.float32)
    for i in range(n):
        k, e = res.intrinsics[i], res.extrinsics[i]
        if not registered[i] or not (np.isfinite(k).all() and np.isfinite(e).all()):
            continue
        idx = np.flatnonzero(finite_pts) if per_view is None else np.array(per_view[i], np.int64)
        idx = idx[finite_pts[idx]]
        if idx.size == 0:
            continue
        cam = pts[idx] @ e[:, :3].T + e[:, 3]
        front = cam[:, 2] > 1e-9
        idx, cam = idx[front], cam[front]
        uvw = cam @ k.T
        u = np.floor(uvw[:, 0] / uvw[:, 2]).astype(np.int64)
        v = np.floor(uvw[:, 1] / uvw[:, 2]).astype(np.int64)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx, cam, u, v = idx[ok], cam[ok], u[ok], v[ok]
        order = np.argsort(-cam[:, 2], kind="stable")  # far first, so the nearest wins
        idx, cam, u, v = idx[order], cam[order], u[order], v[order]
        depth[i, v, u] = cam[:, 2]
        conf[i, v, u] = pconf[idx]
        world[i, v, u] = pts[idx]

    dense = DenseRecon(
        backbone=backbone or res.method,
        intrinsics=np.nan_to_num(res.intrinsics).astype(np.float64),
        extrinsics=np.nan_to_num(res.extrinsics).astype(np.float64),
        depth=depth,
        depth_conf=conf,
        world_points=world,
        world_points_conf=conf.copy(),
        runtime_s=float(res.runtime_s or 0.0),
        peak_vram_mb=float(res.peak_vram_mb or 0.0),
    )
    dense.scene_scale = compute_scene_scale(dense)
    return dense


def to_dense_recon(res: SparseRecon, backbone: str | None = None) -> DenseRecon:
    """Convert a method result into the dense ``types.Recon``.

    Per-view depth (VGGT) is unprojected directly. Results with only a sparse point cloud
    (COLMAP) go through ``sparse_to_dense``: sparse per-view maps, zero confidence elsewhere.
    """
    if res.failed:
        raise ReconAdaptError(f"reconstruction failed: {res.failure_reason}")
    if res.depth is None and res.points is not None and res.extrinsics is not None:
        return sparse_to_dense(res, backbone)
    if res.depth is None or res.extrinsics is None or res.intrinsics is None:
        raise ReconAdaptError(
            f"method {res.method!r} has neither per-view depth nor a sparse point cloud"
        )
    depth = np.asarray(res.depth, dtype=np.float32)
    if depth.ndim == 4:  # (N,H,W,1)
        depth = depth[..., 0]
    conf = (
        np.ones_like(depth)
        if res.depth_conf is None
        else np.asarray(res.depth_conf, dtype=np.float32)
    )
    registered = np.ones(depth.shape[0], dtype=bool) if res.registered is None else res.registered
    world = unproject_depth(depth, res.intrinsics, res.extrinsics)
    bad = ~np.isfinite(world).all(-1) | ~np.isfinite(depth) | ~registered[:, None, None]
    world[bad] = 0.0
    conf = np.where(bad, 0.0, conf).astype(np.float32)
    dense = DenseRecon(
        backbone=backbone or res.method,
        intrinsics=np.nan_to_num(res.intrinsics).astype(np.float64),
        extrinsics=np.nan_to_num(res.extrinsics).astype(np.float64),
        depth=np.where(bad, 0.0, depth).astype(np.float32),
        depth_conf=conf,
        world_points=world,
        world_points_conf=conf.copy(),
        runtime_s=float(res.runtime_s or 0.0),
        peak_vram_mb=float(res.peak_vram_mb or 0.0),
    )
    dense.scene_scale = compute_scene_scale(dense)
    return dense


class MethodReconstructor:
    """Wraps a path-based ``ReconMethod`` as a frame-based ``Reconstructor``."""

    def __init__(self, method: ReconMethod, backbone: str | None = None) -> None:
        self.method = method
        self.name = backbone or method.name
        # Backbone-declared input preprocessing ("crop" | "pad" | None = native resolution).
        # The pipeline uses it to fill Frame.grid_map.
        self.preprocess_mode: str | None = getattr(method, "preprocess_mode", None)

    def reconstruct(self, frames: list[Frame]) -> DenseRecon:
        return to_dense_recon(self.method.reconstruct([f.path for f in frames]), self.name)

    def config(self) -> dict[str, Any]:
        return self.method.config()

    @property
    def grid_target(self) -> int | None:
        """Input-grid width the backbone currently uses (None = native/not applicable)."""
        size = getattr(self.method, "input_size", None)
        return int(size) if size is not None else None

    def _rewrap(self, method: ReconMethod) -> MethodReconstructor:
        return MethodReconstructor(method, self.name)

    def with_resolution_scale(self, scale: float) -> MethodReconstructor:
        """Same backbone at a lower input tier; RungUnavailable if the method cannot."""
        fn = getattr(self.method, "with_resolution_scale", None)
        if fn is None:
            raise RungUnavailable(f"{self.name} has no resolution control")
        return self._rewrap(fn(scale))

    def with_oom_raising(self) -> MethodReconstructor:
        """Same backbone, raising OOM instead of returning a failure (for the OOM ladder)."""
        fn = getattr(self.method, "with_oom_raising", None)
        return self if fn is None else self._rewrap(fn())


def load_method(backbone: str, **kwargs: Any) -> MethodReconstructor:
    """Build a real backbone; heavy imports (torch, pycolmap) happen here, lazily."""
    if backbone == "vggt":
        from countonce.recon.vggt import VGGTMethod

        return MethodReconstructor(VGGTMethod(**kwargs), "vggt")
    if backbone == "colmap":
        from countonce.recon.colmap import ColmapMethod

        return MethodReconstructor(ColmapMethod(**kwargs), "colmap")
    raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")
