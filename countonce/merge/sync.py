"""Cycle-consistent synchronization baseline (docs/PLAN.md §9.3 item 9, §10).

Pairwise matches are built per pair of frames from v2's geometric cost (symmetric
reprojection error, optional appearance gate) and a per-pair Hungarian assignment. These
pairwise matches are noisy and not transitively consistent. Spectral permutation
synchronization (Pachauri et al., NeurIPS 2013) fixes that: the full block match matrix
(zero diagonal blocks = cannot-link) is eigendecomposed, and detections are rounded into a
universe of instances from the top eigenvectors, with a Hungarian assignment per view so
two detections of the same view are never merged. The output is a partition, hence
transitively consistent by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from countonce.merge.appearance import cosine_sim
from countonce.merge.base import ClusterResult
from countonce.merge.match import V2Associator, V2Config, _det_center_scale, _size, _Views
from countonce.types import Frame, LiftedDetection, MatchEvidence, Recon


@dataclass
class SyncConfig:
    min_pair_affinity: float = 0.05  # pairwise matches weaker than this are dropped
    eig_threshold: float = 0.5  # eigenvalue cut deciding the universe dimension
    assign_threshold: float = 0.5  # max (1 - cosine) between a detection and an instance


def synchronize(
    affinity: np.ndarray, views: Sequence[int], cfg: SyncConfig | None = None
) -> np.ndarray:
    """Round a noisy symmetric match matrix into per-detection instance labels.

    ``affinity`` is (N, N) in [0, 1]; same-view entries are ignored (cannot-link).
    Returns int labels (N,); no two detections with equal ``views`` share a label, and
    the induced "same label" relation is an equivalence relation.
    """
    cfg = cfg or SyncConfig()
    n = len(views)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    vw = np.asarray(views)
    a = np.asarray(affinity, float).copy()
    a = np.clip((a + a.T) / 2.0, 0.0, 1.0)
    a[vw[:, None] == vw[None, :]] = 0.0
    np.fill_diagonal(a, 0.0)

    evals, evecs = np.linalg.eigh(a)
    keep = evals > cfg.eig_threshold
    emb = evecs[:, keep] * np.sqrt(evals[keep])
    norms = np.linalg.norm(emb, axis=1) if emb.shape[1] else np.zeros(n)
    live = norms > 1e-6  # rows with no match evidence become their own instances
    unit = np.zeros_like(emb)
    unit[live] = emb[live] / norms[live, None]

    centers: list[np.ndarray] = []  # running sum of unit rows per instance
    for v in sorted(set(vw.tolist())):
        idx = [i for i in np.flatnonzero(vw == v) if live[i]]
        taken: set[int] = set()
        if idx and centers:
            c = np.stack([s / max(np.linalg.norm(s), 1e-12) for s in centers])
            cost = 1.0 - unit[idx] @ c.T
            rows, cols = linear_sum_assignment(cost)
            for r, k in zip(rows, cols, strict=True):
                if cost[r, k] <= cfg.assign_threshold:
                    labels[idx[r]] = k
                    centers[k] = centers[k] + unit[idx[r]]
                    taken.add(r)
        for r, i in enumerate(idx):
            if r not in taken:
                labels[i] = len(centers)
                centers.append(unit[i].copy())
    next_label = len(centers)
    for i in np.flatnonzero(~live):
        labels[i] = next_label
        next_label += 1
    return labels


class SyncAssociator:
    """Spectral permutation synchronization over v2's pairwise matches (``assoc: sync``)."""

    name = "sync"

    def __init__(
        self,
        config: SyncConfig | None = None,
        v2_config: V2Config | None = None,
        extractor=None,
        frame_to_view: Mapping[int, int] | None = None,
    ):
        self.config = config or SyncConfig()
        self._v2 = V2Associator(v2_config, extractor, frame_to_view)
        self.frame_to_view = frame_to_view

    def _pair_affinity(self, a, b, geo: _Views, feats) -> float:
        """Symmetric match score in [0, 1] from mutual reprojection of the two detections."""
        gate = self._v2.config.max_reproj_norm
        cfg = self._v2.config
        errs = []
        for src, dst in ((a, b), (b, a)):
            view = geo.view(dst.det.frame_id)
            sx, sy = geo.scale(dst.det.frame_id)
            vis, uv, z = geo.project(src.xyz, _size(src), view, cfg.depth_tol)
            if not vis:
                return 0.0
            center, scale = _det_center_scale(dst.det)
            if scale is None:
                focal = float(
                    (geo.recon.intrinsics[view, 0, 0] + geo.recon.intrinsics[view, 1, 1]) / 2
                )
                scale = _size(src) * focal / max(z, 1e-9) / ((sx + sy) / 2)
            err = float(np.linalg.norm(uv / np.array([sx, sy]) - center))
            errs.append(err / max(scale, 1.0))
        rn = float(np.mean(errs))
        if rn > gate:
            return 0.0
        score = 1.0 - rn / gate
        fa, fb = feats.get(a.det.id), feats.get(b.det.id)
        if fa is not None and fb is not None:
            sim = cosine_sim(fa, fb)
            if sim < cfg.min_appearance_sim:
                return 0.0
            score *= 1.0 - cfg.appearance_weight * (1.0 - sim) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def pairwise_matrix(self, ok, geo, feats) -> np.ndarray:
        """Block match matrix: per frame pair, Hungarian-assigned affinities (else 0)."""
        n = len(ok)
        mat = np.zeros((n, n))
        frames = sorted({ld.det.frame_id for ld in ok})
        idx = {f: [i for i, ld in enumerate(ok) if ld.det.frame_id == f] for f in frames}
        for x, fi in enumerate(frames):
            for fj in frames[x + 1 :]:
                ii, jj = idx[fi], idx[fj]
                aff = np.array(
                    [[self._pair_affinity(ok[i], ok[j], geo, feats) for j in jj] for i in ii]
                )
                rows, cols = linear_sum_assignment(-aff)
                for r, c in zip(rows, cols, strict=True):
                    if aff[r, c] > self.config.min_pair_affinity:
                        mat[ii[r], jj[c]] = mat[jj[c], ii[r]] = aff[r, c]
        return mat

    def associate(
        self,
        lifted: Sequence[LiftedDetection],
        recon: Recon | None = None,
        frames: Mapping[int, Frame] | None = None,
    ) -> ClusterResult:
        if recon is None:
            raise ValueError("SyncAssociator needs the Recon for projection")
        res = ClusterResult(instances=[])
        ok: list[LiftedDetection] = []
        for ld in sorted(lifted, key=lambda m: (m.det.frame_id, m.det.id)):
            if ld.ok:
                ok.append(ld)
            else:
                res.detection_status[ld.det.id] = "rejected"
                res.reject_reasons[ld.det.id] = ld.fail_reason or "no_xyz"
        if not ok:
            return res

        geo = _Views(recon, frames, self.frame_to_view)
        feats = self._v2._features(ok, frames)
        mat = self.pairwise_matrix(ok, geo, feats)
        labels = synchronize(mat, [ld.det.frame_id for ld in ok], self.config)

        groups: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(i)
        for members in groups.values():
            rep = members[0]
            for j in members[1:]:
                a, b = ok[rep], ok[j]
                res.evidence.append(
                    MatchEvidence(
                        det_a=a.det.id,
                        det_b=b.det.id,
                        dist_3d_norm=float(np.linalg.norm(a.xyz - b.xyz))
                        / ((_size(a) + _size(b)) / 2),
                        reproj_error_px=0.0,
                        appearance_sim=0.0,
                        match_prob=float(mat[rep, j]),
                        merged=True,
                        reason="synchronized",
                    )
                )
        self._v2._build([[ok[i] for i in m] for m in groups.values()], geo, res)
        return res
