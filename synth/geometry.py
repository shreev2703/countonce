"""Seeded numpy generator of multi-view matching episodes (PLAN §10.2, CPU only, no renderer).

An episode is a set of 3D instances seen by several random cameras. Each camera yields noisy 2D
detections with drops and injected false positives. Ground-truth identity labels say which
instance produced each detection (-1 for false positives).

Compact `.npz` layout (flat arrays, detections concatenated across views):
    seed, n_views, n_instances            scalars
    K (V,3,3), R (V,3,3), t (V,3)         intrinsics and world->camera extrinsics
    points3d (N,3)                        instance positions
    view (D,), uv (D,2), depth (D,), score (D,), identity (D,)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SIZE = 640.0


@dataclass(frozen=True)
class GeometryConfig:
    n_views: tuple[int, int] = (3, 8)
    n_instances: tuple[int, int] = (5, 60)
    drop_prob: float = 0.15
    false_positive_rate: float = 0.1  # expected false positives per view, as fraction of N
    pixel_noise: float = 2.0
    scene_radius: float = 1.0
    camera_distance: float = 3.5


@dataclass
class Episode:
    seed: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    points3d: np.ndarray
    view: np.ndarray
    uv: np.ndarray
    depth: np.ndarray
    score: np.ndarray
    identity: np.ndarray

    @property
    def n_views(self) -> int:
        return int(self.K.shape[0])

    @property
    def n_instances(self) -> int:
        return int(self.points3d.shape[0])

    @property
    def n_detections(self) -> int:
        return int(self.view.shape[0])

    def view_slice(self, v: int) -> np.ndarray:
        return np.flatnonzero(self.view == v)

    def match_matrix(self, a: int, b: int) -> np.ndarray:
        """Boolean GT matrix between detections of views a and b (rows a, cols b)."""
        ia, ib = self.identity[self.view_slice(a)], self.identity[self.view_slice(b)]
        return (ia[:, None] == ib[None, :]) & (ia[:, None] >= 0)

    def pair_targets(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GT targets: (match matrix, unmatched-in-a mask, unmatched-in-b mask)."""
        m = self.match_matrix(a, b)
        return m, ~m.any(axis=1), ~m.any(axis=0)

    def true_count(self) -> int:
        return int(np.unique(self.identity[self.identity >= 0]).size)


def _look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = target - center
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(z @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # world -> camera rows
    return R, -R @ center


def generate_episode(seed: int, cfg: GeometryConfig | None = None) -> Episode:
    cfg = cfg or GeometryConfig()
    rng = np.random.default_rng(seed)
    V = int(rng.integers(cfg.n_views[0], cfg.n_views[1] + 1))
    N = int(rng.integers(cfg.n_instances[0], cfg.n_instances[1] + 1))
    pts = rng.uniform(-1, 1, size=(N, 3)) * cfg.scene_radius
    pts[:, 2] = np.abs(pts[:, 2]) * 0.5

    Ks, Rs, ts = [], [], []
    for _ in range(V):
        az = rng.uniform(0, 2 * np.pi)
        el = rng.uniform(0.2, 1.2)
        dist = cfg.camera_distance * rng.uniform(0.8, 1.3)
        c = dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        R, t = _look_at(c, rng.normal(0, 0.1, 3))
        f = IMAGE_SIZE * rng.uniform(0.9, 1.6)
        Ks.append([[f, 0, IMAGE_SIZE / 2], [0, f, IMAGE_SIZE / 2], [0, 0, 1]])
        Rs.append(R)
        ts.append(t)
    K, R, t = np.array(Ks), np.array(Rs), np.array(ts)

    view, uv, depth, score, ident = [], [], [], [], []
    for v in range(V):
        pc = pts @ R[v].T + t[v]
        proj = pc @ K[v].T
        z = pc[:, 2]
        xy = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
        vis = (z > 0.1) & np.all((xy >= 0) & (xy < IMAGE_SIZE), axis=1)
        keep = vis & (rng.random(N) >= cfg.drop_prob)
        idx = np.flatnonzero(keep)
        noisy = xy[idx] + rng.normal(0, cfg.pixel_noise, (idx.size, 2))
        view.append(np.full(idx.size, v))
        uv.append(noisy)
        depth.append(z[idx] * (1 + rng.normal(0, 0.02, idx.size)))
        score.append(np.clip(rng.normal(0.8, 0.1, idx.size), 0.05, 1.0))
        ident.append(idx)
        n_fp = int(rng.poisson(cfg.false_positive_rate * N))
        view.append(np.full(n_fp, v))
        uv.append(rng.uniform(0, IMAGE_SIZE, (n_fp, 2)))
        depth.append(rng.uniform(2.0, 5.0, n_fp))
        score.append(np.clip(rng.normal(0.4, 0.1, n_fp), 0.05, 1.0))
        ident.append(np.full(n_fp, -1))

    return Episode(
        seed=seed,
        K=K,
        R=R,
        t=t,
        points3d=pts,
        view=np.concatenate(view).astype(np.int64),
        uv=np.concatenate(uv).reshape(-1, 2),
        depth=np.concatenate(depth),
        score=np.concatenate(score),
        identity=np.concatenate(ident).astype(np.int64),
    )


def save_episode(ep: Episode, path: str | Path) -> None:
    np.savez_compressed(
        path,
        seed=ep.seed,
        n_views=ep.n_views,
        n_instances=ep.n_instances,
        K=ep.K.astype(np.float32),
        R=ep.R.astype(np.float32),
        t=ep.t.astype(np.float32),
        points3d=ep.points3d.astype(np.float32),
        view=ep.view.astype(np.int16),
        uv=ep.uv.astype(np.float32),
        depth=ep.depth.astype(np.float32),
        score=ep.score.astype(np.float32),
        identity=ep.identity.astype(np.int32),
    )


def load_episode(path: str | Path) -> Episode:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    return Episode(
        seed=int(d["seed"]),
        K=d["K"],
        R=d["R"],
        t=d["t"],
        points3d=d["points3d"],
        view=d["view"].astype(np.int64),
        uv=d["uv"],
        depth=d["depth"],
        score=d["score"],
        identity=d["identity"].astype(np.int64),
    )


def write_dataset(
    out_dir: str | Path, n_episodes: int, seed0: int = 0, cfg: GeometryConfig | None = None
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_episodes):
        p = out / f"ep_{seed0 + i:06d}.npz"
        save_episode(generate_episode(seed0 + i, cfg), p)
        paths.append(p)
    return paths


class EpisodeDataset:
    """Loads `.npz` episodes from a directory and yields per-view-pair training samples.

    Each item is a dict of numpy arrays: features for detections in views a and b
    (`feat_a`, `feat_b`: [n, 6] = u, v, depth, score normalised) and GT `match` (na, nb) bool
    plus `unmatched_a`, `unmatched_b` dustbin targets.
    """

    def __init__(self, root: str | Path):
        self.paths = sorted(Path(root).glob("*.npz"))

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def features(ep: Episode, idx: np.ndarray) -> np.ndarray:
        uv = ep.uv[idx] / IMAGE_SIZE
        return np.concatenate([uv, ep.depth[idx, None] / 5.0, ep.score[idx, None]], axis=1).astype(
            np.float32
        )

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        ep = load_episode(self.paths[i])
        rng = np.random.default_rng(ep.seed)
        a, b = rng.choice(ep.n_views, size=2, replace=False)
        m, ua, ub = ep.pair_targets(int(a), int(b))
        return {
            "feat_a": self.features(ep, ep.view_slice(int(a))),
            "feat_b": self.features(ep, ep.view_slice(int(b))),
            "match": m,
            "unmatched_a": ua,
            "unmatched_b": ub,
        }
