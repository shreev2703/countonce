"""C-DOG association baseline (PLAN §9.3 baseline 5, ``docs/RELATED_WORK.md`` §4 E4).

Re-implemented from the paper, not from its repository (the repo has no licence):

    Y.-H. Sun, T.-H. Lin, J. Chen, H. Jiang, Y. H. Hu. "C-DOG: Multi-View Multi-instance
    Feature Association Using Connected delta-Overlap Graphs." arXiv:2507.14095 (v2, Aug 2025).

The method is featureless: it uses only 2D detection centres and the cameras (K, R, t).

1. **Graph initialisation** (§IV-C, Alg. 1). Every detection is a node. For each ordered view pair
   (m, m') and each node in m, take the node of m' closest to the epipolar line of the m node;
   keep that directed edge when the point-to-line distance is below ``tau_px``. Edges are used
   undirected.
2. **Weak edge pruning** (§IV-D, Alg. 2). An edge survives only if the closed-neighbourhood overlap
   ``|N[a] & N[b]| / max(|N[a]|, |N[b]|)`` exceeds ``delta``. Connected components of what is left
   are the groups.
3. **Group-level outlier removal** (§IV-E, Alg. 3). In each group of >= 3 nodes, triangulate
   every pair, back-project into the remaining views, average the back-projection error (BPE) per
   node, and drop nodes above ``Q3 + alpha * IQR`` (list padded with a zero, lower bound zero).
   Repeat until nothing is dropped.

Every point where the paper is silent or ambiguous is listed in ``docs/DEVIATIONS.md``
("C-DOG re-implementation"). In short: the paper's §IV-F (whole-group removal on a "sudden
increase" of BPE) is not implemented; groups keep >1 node per view only until a same-view
tie-break splits them; dropped nodes stay as singleton instances so every detection is counted.

Nothing here is imported by the deployed package except lazily for ``assoc: cdog``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from countonce.merge.base import ClusterResult
from countonce.merge.match import V2Associator, V2Config, _det_center_scale, _size, _Views
from countonce.merge.order import frame_rank
from countonce.types import Frame, LiftedDetection, MatchEvidence, Recon

BIG_BPE = 1e6  # back-projection error (px) of a triangulation behind a camera
MAX_PAIRS = 4000  # deterministic cap on triangulated pairs per group (paper has none)


@dataclass
class CDOGConfig:
    tau_px: float = 4.0  # epipolar distance threshold (paper: alpha * sqrt(2) * sigma)
    delta: float = 0.5  # neighbourhood-overlap threshold (paper: 0.5-0.55)
    iqr_alpha: float = 2.0  # IQR multiplier of the outlier filter (paper: 2)
    bpe_floor_px: float = 0.5  # a node under this mean BPE is never an outlier (not in the paper)


# ----------------------------------------------------------------------------- geometry


def fundamental(k1: np.ndarray, ext1: np.ndarray, k2: np.ndarray, ext2: np.ndarray) -> np.ndarray:
    """F with ``x2^T F x1 = 0`` for the same 3D point seen at pixel x1 (view 1) and x2 (view 2).

    ``ext`` is world-to-camera [R | t]. The epipolar line of x1 in view 2 is ``F @ x1``."""
    r1, t1 = ext1[:, :3], ext1[:, 3]
    r2, t2 = ext2[:, :3], ext2[:, 3]
    r = r2 @ r1.T
    t = t2 - r @ t1
    tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    return np.asarray(np.linalg.inv(k2).T @ tx @ r @ np.linalg.inv(k1))


def _baseline(ext1: np.ndarray, ext2: np.ndarray) -> float:
    """Distance between the two camera centres, in the extrinsics' units."""
    c1 = -ext1[:, :3].T @ ext1[:, 3]
    c2 = -ext2[:, :3].T @ ext2[:, 3]
    return float(np.linalg.norm(c1 - c2))


def epipolar_distances(f: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """(n1, n2) distances in px from each ``p2`` point to the epipolar line of each ``p1`` point."""
    h1 = np.c_[p1, np.ones(len(p1))]
    h2 = np.c_[p2, np.ones(len(p2))]
    lines = h1 @ f.T  # (n1, 3), line in view 2
    norm = np.linalg.norm(lines[:, :2], axis=1)
    num = np.abs(lines @ h2.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = num / norm[:, None]
    d[~np.isfinite(d)] = np.inf  # zero baseline: the line is undefined, so no edge
    return np.asarray(d)


def projection_matrices(k: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """(n, 3, 4) ``K [R | t]`` for stacked intrinsics (n,3,3) and extrinsics (n,3,4)."""
    return np.asarray(np.einsum("nij,njk->nik", k, ext))


def triangulate_pairs(pa: np.ndarray, pb: np.ndarray, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of ``n`` point pairs. ``pa``/``pb`` are (n,3,4), ``xa``/``xb``
    (n,2) pixels. Returns (n,4) homogeneous points."""
    rows = np.stack(
        [
            xa[:, 0:1] * pa[:, 2] - pa[:, 0],
            xa[:, 1:2] * pa[:, 2] - pa[:, 1],
            xb[:, 0:1] * pb[:, 2] - pb[:, 0],
            xb[:, 1:2] * pb[:, 2] - pb[:, 1],
        ],
        axis=1,
    )  # (n, 4, 4)
    _, _, vt = np.linalg.svd(rows)
    return np.asarray(vt[:, -1, :])


def back_projection_error(p: np.ndarray, x_h: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Pixel distance of homogeneous 3D points ``x_h`` (n,4) projected by ``p`` (m,3,4) to the
    observations ``obs`` (m,2). Returns (n, m); a point behind a camera counts as ``BIG_BPE``."""
    proj = np.einsum("mij,nj->nmi", p, x_h)  # (n, m, 3)
    w = proj[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = proj[..., :2] / w[..., None]
    err = np.linalg.norm(uv - obs[None], axis=-1)
    # sign of w relative to the homogeneous scale: the point is in front when w and the 4th
    # coordinate's depth agree; with a free sign in the DLT null vector, use w * x_h[3] > 0
    front = (w * x_h[:, 3:4]) > 0
    err = np.where(front & np.isfinite(err), err, BIG_BPE)
    return np.minimum(err, BIG_BPE)


# ----------------------------------------------------------------------------- the algorithm


def initial_edges(
    pts: np.ndarray, views: np.ndarray, k: np.ndarray, ext: np.ndarray, tau_px: float
) -> set[tuple[int, int]]:
    """Alg. 1: undirected edge set from per-view nearest-to-epipolar-line candidates under ``tau``.

    ``views`` indexes ``k`` / ``ext`` (one camera per view)."""
    edges: set[tuple[int, int]] = set()
    vids = sorted({int(v) for v in views})
    members = {v: np.flatnonzero(views == v) for v in vids}
    for m in vids:
        for m2 in vids:
            if m == m2:
                continue
            if _baseline(ext[m], ext[m2]) < 1e-9:
                continue  # same camera centre: no epipolar geometry, so no edge
            f = fundamental(k[m], ext[m], k[m2], ext[m2])
            d = epipolar_distances(f, pts[members[m]], pts[members[m2]])
            best = np.argmin(d, axis=1)
            for a, b in enumerate(best):
                if d[a, b] < tau_px:
                    i, j = int(members[m][a]), int(members[m2][b])
                    edges.add((min(i, j), max(i, j)))
    return edges


def prune_weak_edges(n: int, edges: set[tuple[int, int]], delta: float) -> list[tuple[int, int]]:
    """Alg. 2: keep an edge only when its closed-neighbourhood overlap score exceeds ``delta``.

    Scores are computed on the initial graph for all edges at once (order independent)."""
    nbr: list[set[int]] = [{i} for i in range(n)]
    for i, j in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    keep = []
    for i, j in sorted(edges):
        theta = len(nbr[i] & nbr[j]) / max(len(nbr[i]), len(nbr[j]))
        if theta > delta:
            keep.append((i, j))
    return keep


def components(n: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the graph on ``n`` nodes, each sorted, ordered by smallest node."""
    if not edges:
        return [[i] for i in range(n)]
    r = np.array([e[0] for e in edges])
    c = np.array([e[1] for e in edges])
    g = coo_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    out: dict[int, list[int]] = {}
    for i, lb in enumerate(lab):
        out.setdefault(int(lb), []).append(i)
    return sorted(out.values(), key=lambda g_: g_[0])


def node_bpe(idx: Sequence[int], pts: np.ndarray, views: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mean back-projection error per node of a group (Alg. 3 lines 3-10, §IV-E).

    Every pair of nodes from two different views is triangulated and back-projected into every
    other node of the group whose view differs from both pair views; a node's score is the mean
    over all reconstructions that reach it. Nodes no reconstruction reaches score 0."""
    ids = np.asarray(idx)
    n = len(ids)
    ii, jj = np.triu_indices(n, k=1)
    ok = views[ids[ii]] != views[ids[jj]]
    ii, jj = ii[ok], jj[ok]
    if len(ii) > MAX_PAIRS:
        sel = np.random.default_rng(0).choice(len(ii), MAX_PAIRS, replace=False)
        ii, jj = ii[np.sort(sel)], jj[np.sort(sel)]
    tot = np.zeros(n)
    cnt = np.zeros(n)
    if len(ii) == 0:
        return tot
    pv = p[views[ids]]  # (n, 3, 4) projection matrix of each node's view
    obs = pts[ids]
    xh = triangulate_pairs(pv[ii], pv[jj], obs[ii], obs[jj])
    err = back_projection_error(pv, xh, obs)  # (npairs, n)
    vi, vj = views[ids[ii]], views[ids[jj]]
    vn = views[ids]
    reach = (vn[None, :] != vi[:, None]) & (vn[None, :] != vj[:, None])
    tot = (err * reach).sum(0)
    cnt = reach.sum(0).astype(float)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)


def iqr_outliers(bpe: np.ndarray, alpha: float, floor_px: float = 0.0) -> np.ndarray:
    """Boolean mask of nodes above ``Q3 + alpha * IQR``; the list is padded with a zero first and
    the lower bound is zero (§IV-E). ``floor_px`` additionally protects nodes with a negligible
    BPE: with near-exact observations the IQR collapses to rounding noise and the paper's rule
    alone would flag arbitrary nodes (DEVIATIONS)."""
    padded = np.r_[bpe, 0.0]
    q1, q3 = np.percentile(padded, [25, 75])
    return np.asarray((bpe > q3 + alpha * (q3 - q1)) & (bpe > floor_px))


def remove_group_outliers(
    group: list[int],
    pts: np.ndarray,
    views: np.ndarray,
    p: np.ndarray,
    alpha: float,
    floor_px: float = 0.0,
) -> tuple[list[int], list[int]]:
    """Alg. 3 on one group: returns (kept nodes, dropped nodes)."""
    cur = list(group)
    dropped: list[int] = []
    while len(cur) >= 3:
        bad = iqr_outliers(node_bpe(cur, pts, views, p), alpha, floor_px)
        if not bad.any():
            break
        dropped += [n for n, b in zip(cur, bad, strict=True) if b]
        cur = [n for n, b in zip(cur, bad, strict=True) if not b]
    return cur, dropped


def split_same_view(
    group: list[int], pts: np.ndarray, views: np.ndarray, p: np.ndarray
) -> tuple[list[int], list[int]]:
    """Enforce one node per view (§IV-A constraint 2), which the paper leaves unspecified: keep,
    per view, the node with the lowest mean BPE of the group; the others are returned as split."""
    if len({int(views[n]) for n in group}) == len(group):
        return group, []
    bpe = node_bpe(group, pts, views, p) if len(group) >= 3 else np.zeros(len(group))
    best: dict[int, tuple[float, int]] = {}
    for n, b in zip(group, bpe, strict=True):
        v = int(views[n])
        if v not in best or (b, n) < best[v]:
            best[v] = (float(b), n)
    keep = sorted(n for _, n in best.values())
    return keep, [n for n in group if n not in set(keep)]


def cdog_groups(
    pts: np.ndarray,
    views: np.ndarray,
    k: np.ndarray,
    ext: np.ndarray,
    cfg: CDOGConfig | None = None,
) -> list[list[int]]:
    """Partition of the nodes into groups, every node in exactly one (outliers as singletons).

    ``pts`` (n,2) pixel centres, ``views`` (n,) camera index per node, ``k`` (m,3,3) intrinsics,
    ``ext`` (m,3,4) world-to-camera extrinsics."""
    cfg = cfg or CDOGConfig()
    pts = np.asarray(pts, float).reshape(-1, 2)
    views = np.asarray(views, int)
    n = len(pts)
    if n == 0:
        return []
    p = projection_matrices(np.asarray(k, float), np.asarray(ext, float))
    edges = initial_edges(pts, views, np.asarray(k, float), np.asarray(ext, float), cfg.tau_px)
    strong = prune_weak_edges(n, edges, cfg.delta)
    out: list[list[int]] = []
    for g in components(n, strong):
        if len(g) == 1:
            out.append(g)
            continue
        kept, dropped = remove_group_outliers(g, pts, views, p, cfg.iqr_alpha, cfg.bpe_floor_px)
        kept, split = split_same_view(kept, pts, views, p)
        if kept:
            out.append(sorted(kept))
        out += [[d] for d in dropped + split]
    return sorted(out, key=lambda g_: g_[0])


# ----------------------------------------------------------------------------- associator


class CDOGAssociator:
    """``Associator`` running C-DOG on the detections that lifted (same inputs as v1/v2/sync)."""

    name = "cdog"

    def __init__(
        self,
        config: CDOGConfig | None = None,
        v2_config: V2Config | None = None,
        frame_to_view: Mapping[int, int] | None = None,
    ):
        self.config = config or CDOGConfig()
        self._v2 = V2Associator(v2_config, None, frame_to_view)
        self.frame_to_view = frame_to_view

    def associate(
        self,
        lifted: Sequence[LiftedDetection],
        recon: Recon | None = None,
        frames: Mapping[int, Frame] | None = None,
    ) -> ClusterResult:
        if recon is None:
            raise ValueError("CDOGAssociator needs the Recon for the cameras")
        res = ClusterResult(instances=[])
        rank = frame_rank({m.det.frame_id for m in lifted}, recon, self.frame_to_view)
        ok: list[LiftedDetection] = []
        for ld in sorted(lifted, key=lambda m: (rank(m.det.frame_id), m.det.id)):
            if ld.ok:
                ok.append(ld)
            else:
                res.detection_status[ld.det.id] = "rejected"
                res.reject_reasons[ld.det.id] = ld.fail_reason or "no_xyz"
        if not ok:
            return res
        geo = _Views(recon, frames, self.frame_to_view)
        # detection centres in recon-grid pixels, the frame the intrinsics live in
        pts = np.empty((len(ok), 2))
        views = np.empty(len(ok), int)
        for i, ld in enumerate(ok):
            centre, _ = _det_center_scale(ld.det)
            sx, sy, ox, oy = geo.affine(ld.det.frame_id)
            pts[i] = (centre[0] * sx + ox, centre[1] * sy + oy)
            views[i] = geo.view(ld.det.frame_id)
        groups = cdog_groups(pts, views, recon.intrinsics, recon.extrinsics, self.config)
        for g in groups:
            rep = ok[g[0]]
            for j in g[1:]:
                other = ok[j]
                assert rep.xyz is not None and other.xyz is not None
                res.evidence.append(
                    MatchEvidence(
                        det_a=rep.det.id,
                        det_b=other.det.id,
                        dist_3d_norm=float(np.linalg.norm(rep.xyz - other.xyz))
                        / ((_size(rep) + _size(other)) / 2),
                        reproj_error_px=0.0,
                        appearance_sim=0.0,
                        match_prob=1.0,
                        merged=True,
                        reason="cdog_group",
                    )
                )
        self._v2._build([[ok[i] for i in g] for g in groups], geo, res)
        return res
