"""Adapter from the ported ``recon.result.Recon`` to the dense section-7 ``types.Recon``.

torch and pycolmap are imported only inside ``load_method`` so this module (and the
tests that use it) work on a CPU-only install.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from countonce.lift import compute_scene_scale
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


def to_dense_recon(res: SparseRecon, backbone: str | None = None) -> DenseRecon:
    """Convert a method result with per-view depth into the dense ``types.Recon``.

    Requires depth (VGGT). COLMAP results carry only a sparse point cloud, so they raise
    ``ReconAdaptError`` rather than inventing a depth map.
    """
    if res.failed:
        raise ReconAdaptError(f"reconstruction failed: {res.failure_reason}")
    if res.depth is None or res.extrinsics is None or res.intrinsics is None:
        raise ReconAdaptError(
            f"method {res.method!r} has no per-view depth; dense Recon needs depth "
            "(VGGT). COLMAP is a baseline only."
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

    def reconstruct(self, frames: list[Frame]) -> DenseRecon:
        return to_dense_recon(self.method.reconstruct([f.path for f in frames]), self.name)

    def config(self) -> dict[str, Any]:
        return self.method.config()


def load_method(backbone: str, **kwargs: Any) -> MethodReconstructor:
    """Build a real backbone; heavy imports (torch, pycolmap) happen here, lazily."""
    if backbone == "vggt":
        from countonce.recon.vggt import VGGTMethod

        return MethodReconstructor(VGGTMethod(**kwargs), "vggt")
    if backbone == "colmap":
        from countonce.recon.colmap import ColmapMethod

        return MethodReconstructor(ColmapMethod(**kwargs), "colmap")
    raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")
