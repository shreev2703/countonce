"""Instance formation from match probabilities (PLAN §10.3 "Instance formation").

Threshold the (N,N) match probabilities, take connected components, then split every component
that holds two detections of the same view (frame) with an s-t min-cut on the match scores, with
the two conflicting detections as source and sink (cannot-link). Splitting recurses until no
group has a same-view pair, so the cannot-link constraint holds by construction.

Pure numpy/scipy; no torch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_flow

_CAP_SCALE = 10_000  # probabilities -> integer capacities for scipy's max-flow


def _min_cut_split(w: np.ndarray, s: int, t: int) -> np.ndarray:
    """Boolean mask of the source side of a min s-t cut on the symmetric weights ``w``."""
    n = w.shape[0]
    cap = np.rint(w * _CAP_SCALE).astype(np.int32)
    flow = maximum_flow(csr_matrix(cap), s, t).flow.toarray()
    resid = cap - flow
    side = np.zeros(n, dtype=bool)
    side[s] = True
    stack = [s]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero((resid[u] > 0) & ~side):
            side[v] = True
            stack.append(int(v))
    if side[t]:  # cannot happen after max-flow; guard against an infinite recursion anyway
        side[:] = False
        side[s] = True
    return side


def _split(idx: np.ndarray, w: np.ndarray, views: np.ndarray, out: list[np.ndarray]) -> None:
    v = views[idx]
    order = np.argsort(v, kind="stable")
    dup = np.flatnonzero(v[order][1:] == v[order][:-1])
    if dup.size == 0:
        out.append(idx)
        return
    s, t = int(order[dup[0]]), int(order[dup[0] + 1])
    sub = w[np.ix_(idx, idx)]
    side = _min_cut_split(sub, s, t)
    _split(idx[side], w, views, out)
    _split(idx[~side], w, views, out)


def form_instances(
    probs: np.ndarray, views: Sequence[int] | np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Instance label per detection, shape (N,), labels 0..K-1 in order of first member.

    ``probs`` is the (N,N) match probability matrix (it is symmetrised; same-view entries are
    ignored). No two detections with equal ``views`` ever share a label.
    """
    p = np.asarray(probs, dtype=float)
    vw = np.asarray(views)
    n = vw.shape[0]
    if p.shape != (n, n):
        raise ValueError(f"probs must be ({n},{n}), got {p.shape}")
    if n == 0:
        return np.zeros(0, dtype=int)
    w = np.clip((p + p.T) / 2.0, 0.0, 1.0)
    w[vw[:, None] == vw[None, :]] = 0.0
    np.fill_diagonal(w, 0.0)
    adj = csr_matrix(w > threshold)
    _, comp = connected_components(adj, directed=False)

    groups: list[np.ndarray] = []
    for c in np.unique(comp):
        # Conflicts are split on the full weights inside the component, thresholded edges only
        # decide connectivity, so weak (sub-threshold) evidence cannot hold a group together.
        idx = np.flatnonzero(comp == c)
        _split(idx, np.where(w > threshold, w, 0.0), vw, groups)

    labels = np.full(n, -1, dtype=int)
    for k, g in enumerate(sorted(groups, key=lambda g: int(g.min()))):
        labels[g] = k
    return labels
