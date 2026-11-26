"""Sim(3) alignment (Umeyama 1991), shared by chunk stitching (PLAN §8.3) and benchmark GT matching.

Convention: ``dst ~= s * R @ src + t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sim3:
    s: float
    R: np.ndarray  # (3,3) proper rotation
    t: np.ndarray  # (3,)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Transform (...,3) points."""
        out: np.ndarray = self.s * (np.asarray(pts, dtype=np.float64) @ self.R.T) + self.t
        return out

    def inverse(self) -> Sim3:
        Ri = self.R.T
        return Sim3(1.0 / self.s, Ri, -(Ri @ self.t) / self.s)

    def compose(self, other: Sim3) -> Sim3:
        """``self.compose(other)`` applies ``other`` first, then ``self``."""
        return Sim3(self.s * other.s, self.R @ other.R, self.s * (self.R @ other.t) + self.t)

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.s * self.R
        m[:3, 3] = self.t
        return m


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim(3) mapping ``src`` onto ``dst`` (both (N,3), N >= 3, non-collinear)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching (N,3) arrays, got {src.shape} and {dst.shape}")
    n = len(src)
    if n < 3:
        raise ValueError("need at least 3 correspondences")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs**2).sum() / n
    if var_s < 1e-18:
        raise ValueError("degenerate source points (zero variance)")
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """Camera centres of world-to-camera (N,3,4) extrinsics: ``C = -R^T t``."""
    ext = np.asarray(extrinsics, dtype=np.float64)
    return np.asarray(-np.einsum("nji,nj->ni", ext[:, :3, :3], ext[:, :3, 3]))


def align_by_camera_centers(src_extrinsics: np.ndarray, dst_extrinsics: np.ndarray) -> Sim3:
    """Sim(3) taking the source world frame to the destination world frame from shared cameras
    (e.g. overlapping frames of two chunks, or recon vs dataset poses)."""
    return umeyama(camera_centers(src_extrinsics), camera_centers(dst_extrinsics))


def rotation_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between paired (N,3,3) rotations ``a`` and ``b``."""
    tr = np.einsum("nij,nij->n", np.asarray(a, np.float64), np.asarray(b, np.float64))
    return np.asarray(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def align_by_poses(src_extrinsics: np.ndarray, dst_extrinsics: np.ndarray) -> Sim3:
    """Sim(3) from shared cameras using their orientations, not only their centres.

    Centres alone leave the roll about the camera path free when they are collinear (a camera
    walking along a row). So the rotation is the chordal mean of the per-view orientation
    differences ``R_dst^T R_src``; scale and translation then come from the centres.
    """
    src = np.asarray(src_extrinsics, dtype=np.float64)
    dst = np.asarray(dst_extrinsics, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 3 or len(src) < 2:
        raise ValueError(f"expected matching (N>=2,3,4) arrays, got {src.shape} and {dst.shape}")
    U, _, Vt = np.linalg.svd((np.transpose(dst[:, :3, :3], (0, 2, 1)) @ src[:, :3, :3]).sum(0))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    c_src, c_dst = camera_centers(src), camera_centers(dst)
    a = (c_src - c_src.mean(0)) @ R.T
    b = c_dst - c_dst.mean(0)
    var = float((a**2).sum())
    if var < 1e-18:
        raise ValueError("degenerate source cameras (centres coincide)")
    s = float((a * b).sum()) / var
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"cannot estimate a positive scale from camera centres ({s})")
    return Sim3(s, R, c_dst.mean(0) - s * R @ c_src.mean(0))


def transform_extrinsics(extrinsics: np.ndarray, sim: Sim3) -> np.ndarray:
    """Express world-to-camera extrinsics of the source world in the destination world.

    With ``X_dst = s R X_src + t``, camera coordinates in destination units are
    ``s * x_c = (R_c R^T) X_dst + (s t_c - R_c R^T t)``.
    """
    ext = np.asarray(extrinsics, dtype=np.float64)
    Rc, tc = ext[:, :3, :3], ext[:, :3, 3]
    Rn = Rc @ sim.R.T
    tn = sim.s * tc - np.einsum("nij,j->ni", Rn, sim.t)
    return np.concatenate([Rn, tn[:, :, None]], axis=2)
