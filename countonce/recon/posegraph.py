"""Global Sim(3) synchronization of reconstruction chunks (PLAN section 8.3).

Each chunk ``i`` lives in its own world frame. Chunks that share frames give a pairwise
measurement ``Z_ij`` (Sim(3) taking chunk ``j``'s frame to chunk ``i``'s, Umeyama on the shared
cameras). We solve for ``T_i`` (chunk ``i`` -> global frame, chunk 0 fixed to identity) minimising

    sum_ij w_ij * || log( T_j^-1 T_i Z_ij ) ||^2

with the residual split into rotation (rotation vector, rad), log-scale and translation (divided
by the shared cameras' spread in chunk ``j``, so it is dimensionless), weight ``w_ij`` = shared
frame count. Initialised from the spanning-tree chain, refined by ``scipy`` least squares.
A tree (no redundant edge) has nothing to average, so the chain result is returned as is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from countonce.recon.align import Sim3, camera_centers
from countonce.recon.chunked import MIN_OVERLAP, ChunkAlignmentError, align_chunk


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    z: Sim3  # chunk j frame -> chunk i frame
    weight: float
    ref_len: float  # spread of the shared cameras in chunk j units


@dataclass(frozen=True)
class PoseGraphResult:
    sims: list[Sim3]  # chunk frame -> global frame (chunk 0 = identity)
    n_edges: int
    is_tree: bool


def build_edges(exts: list[np.ndarray], indices: list[np.ndarray]) -> list[Edge]:
    """One edge per chunk pair sharing >= ``MIN_OVERLAP`` frame ids."""
    edges: list[Edge] = []
    pos = [{int(f): k for k, f in enumerate(ix)} for ix in indices]
    for i in range(len(exts)):
        for j in range(i + 1, len(exts)):
            shared = sorted(set(pos[i]) & set(pos[j]))
            if len(shared) < MIN_OVERLAP:
                continue
            ei = exts[i][[pos[i][f] for f in shared]]
            ej = exts[j][[pos[j][f] for f in shared]]
            try:
                z = align_chunk(ei, ej)
            except ChunkAlignmentError as e:
                raise ChunkAlignmentError(f"chunks {i}-{j}: {e}") from e
            ref = float(np.linalg.norm(np.ptp(camera_centers(ej), axis=0)))
            edges.append(Edge(i, j, z, float(len(shared)), ref))
    return edges


def _chain_init(n: int, edges: list[Edge]) -> list[Sim3]:
    """Spanning tree from chunk 0 (strongest edges first); raises if disconnected."""
    sims: list[Sim3 | None] = [None] * n
    sims[0] = Sim3(1.0, np.eye(3), np.zeros(3))
    todo = sorted(edges, key=lambda e: -e.weight)
    progress = True
    while progress:
        progress = False
        for e in todo:
            a, b = sims[e.i], sims[e.j]
            if a is not None and b is None:
                sims[e.j] = a.compose(e.z)  # T_j = T_i Z_ij
                progress = True
            elif b is not None and a is None:
                sims[e.i] = b.compose(e.z.inverse())
                progress = True
    missing = [k for k, s in enumerate(sims) if s is None]
    if missing:
        raise ChunkAlignmentError(f"chunk graph is disconnected: chunks {missing} share no frames")
    return [s for s in sims if s is not None]


def _unpack(x: np.ndarray) -> Sim3:
    return Sim3(float(np.exp(x[3])), Rotation.from_rotvec(x[:3]).as_matrix(), x[4:7])


def _pack(s: Sim3) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(s.R).as_rotvec(), [np.log(s.s)], s.t])


def solve_pose_graph(exts: list[np.ndarray], indices: list[np.ndarray]) -> PoseGraphResult:
    """Chunk -> global Sim(3)s from per-chunk extrinsics and their global frame ids."""
    n = len(exts)
    edges = build_edges(exts, indices)
    init = _chain_init(n, edges)
    is_tree = len(edges) == n - 1
    if is_tree or n == 1:
        return PoseGraphResult(init, len(edges), True)

    def residuals(p: np.ndarray) -> np.ndarray:
        sims = [init[0]] + [_unpack(p[7 * k : 7 * k + 7]) for k in range(n - 1)]
        out = []
        for e in edges:
            err = sims[e.j].inverse().compose(sims[e.i].compose(e.z))
            w = np.sqrt(e.weight)
            out.append(
                w
                * np.concatenate(
                    [
                        Rotation.from_matrix(err.R).as_rotvec(),
                        [np.log(err.s)],
                        err.t / max(e.ref_len, 1e-12),
                    ]
                )
            )
        return np.concatenate(out)

    x0 = np.concatenate([_pack(s) for s in init[1:]])
    sol = least_squares(residuals, x0, method="trf", x_scale="jac")
    sims = [init[0]] + [_unpack(sol.x[7 * k : 7 * k + 7]) for k in range(n - 1)]
    return PoseGraphResult(sims, len(edges), False)
