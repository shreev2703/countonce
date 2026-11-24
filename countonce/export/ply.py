"""``scene.ply`` export: a downsampled, confidence-filtered point cloud for the viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from countonce.types import Recon


def sample_points(
    recon: Recon,
    max_points: int = 200_000,
    conf_percentile: float = 50.0,
    colors: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Confident, finite world points, randomly downsampled (seeded) to ``max_points``.

    ``colors`` is an optional (N,H,W,3) uint8 array aligned with ``world_points``;
    otherwise points are mid-gray. Returns (xyz float32 (M,3), rgb uint8 (M,3)).
    """
    pts = recon.world_points.reshape(-1, 3)
    conf = recon.world_points_conf.reshape(-1)
    keep = np.isfinite(pts).all(1) & (conf > 0)
    if keep.any():
        keep &= conf >= np.percentile(conf[keep], conf_percentile)
    idx = np.flatnonzero(keep)
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_points, replace=False))
    xyz = pts[idx].astype(np.float32)
    if colors is None:
        rgb = np.full((len(idx), 3), 128, dtype=np.uint8)
    else:
        if colors.shape != (*recon.world_points.shape[:3], 3):
            raise ValueError("colors must have shape (N,H,W,3)")
        rgb = colors.reshape(-1, 3)[idx].astype(np.uint8)
    return xyz, rgb


def write_scene_ply(
    recon: Recon,
    path: Path,
    max_points: int = 200_000,
    conf_percentile: float = 50.0,
    colors: np.ndarray | None = None,
    seed: int = 0,
) -> int:
    """Write a binary little-endian PLY; returns the number of points written."""
    xyz, rgb = sample_points(recon, max_points, conf_percentile, colors, seed)
    vert = np.empty(
        len(xyz),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    vert["x"], vert["y"], vert["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vert["r"], vert["g"], vert["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vert.tobytes())
    return len(xyz)
