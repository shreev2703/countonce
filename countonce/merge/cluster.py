"""Associate v1: constrained 3D clustering (docs/PLAN.md §8.5).

Greedy agglomerative (single-linkage) merging of lifted detections. Distance is
Euclidean between ``xyz`` normalized by the pair's mean ``size``; pairs merge
below ``k``. Cannot-link: two clusters never merge if they share a frame, so
detections from the same frame are never merged, even transitively.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from countonce.merge.base import ClusterResult
from countonce.types import Frame, Instance, LiftedDetection, MatchEvidence, Recon

DEFAULT_K = 0.5


def _evidence(
    a: LiftedDetection, b: LiftedDetection, dist: float, k: float, merged: bool, reason: str
) -> MatchEvidence:
    return MatchEvidence(
        det_a=a.det.id,
        det_b=b.det.id,
        dist_3d_norm=float(dist),
        reproj_error_px=0.0,  # not computed in v1
        appearance_sim=0.0,  # not computed in v1
        match_prob=float(np.clip(1.0 - dist / k, 0.0, 1.0)) if merged else 0.0,
        merged=merged,
        reason=reason,
    )


def cluster_detections(
    lifted: Sequence[LiftedDetection], k: float = DEFAULT_K, cannot_link: bool = True
) -> ClusterResult:
    """Cluster lifted detections into instances. Failed lifts are marked rejected.

    ``cannot_link=False`` is the ablation: same-frame detections may merge."""
    if k <= 0:
        raise ValueError("k must be positive")
    res = ClusterResult(instances=[])
    ok: list[LiftedDetection] = []
    for ld in lifted:
        if ld.ok:
            ok.append(ld)
        else:
            res.detection_status[ld.det.id] = "rejected"
            res.reject_reasons[ld.det.id] = ld.fail_reason or "no_xyz"

    n = len(ok)
    if n == 0:
        return res

    xyz = np.stack([np.asarray(ld.xyz) for ld in ok]).astype(float)
    known = [ld.size for ld in ok if ld.size is not None and ld.size > 0]
    fallback = float(np.median(known)) if known else 1.0
    sizes = np.array(
        [ld.size if ld.size is not None and ld.size > 0 else fallback for ld in ok], dtype=float
    )
    frames = [ld.det.frame_id for ld in ok]

    # Candidate pairs below k, closest first (stable tie-break on index).
    diff = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    norm = diff / ((sizes[:, None] + sizes[None, :]) / 2.0)
    ii, jj = np.where(np.triu(norm < k, 1))

    def tie_key(p: tuple[int, int]) -> tuple[float | str, ...]:
        # order-independent: distance, then detection ids
        return (norm[p], *sorted((ok[p[0]].det.id, ok[p[1]].det.id)))

    order = sorted(zip(ii.tolist(), jj.tolist(), strict=True), key=tie_key)

    parent = list(range(n))
    cluster_frames: list[set[int]] = [{f} for f in frames]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in order:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if cannot_link and frames[i] == frames[j]:
            res.evidence.append(
                _evidence(ok[i], ok[j], norm[i, j], k, False, "same_frame_cannot_link")
            )
        elif cannot_link and cluster_frames[ri] & cluster_frames[rj]:
            res.evidence.append(
                _evidence(ok[i], ok[j], norm[i, j], k, False, "cluster_cannot_link")
            )
        else:
            parent[rj] = ri
            cluster_frames[ri] |= cluster_frames[rj]
            res.evidence.append(_evidence(ok[i], ok[j], norm[i, j], k, True, "merged"))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    for inst_id, idxs in enumerate(sorted(groups.values(), key=lambda g: g[0])):
        members = [ok[i] for i in idxs]
        n_views = len({m.det.frame_id for m in members})
        mean_conf = float(np.mean([m.lift_conf for m in members]))
        # Heuristic: more independent views -> more confidence (1 view: x0.5, 2: x0.75, ...)
        conf = float(np.clip(mean_conf * (1.0 - 0.5**n_views), 0.0, 1.0))
        res.instances.append(
            Instance(
                id=inst_id,
                xyz=np.median(xyz[idxs], axis=0),
                size=float(np.mean(sizes[idxs])),
                members=[m.det.id for m in members],
                n_views=n_views,
                conf=conf,
                possibly_moving=any(m.possibly_moving for m in members),
            )
        )
        best = max(members, key=lambda m: (m.lift_conf, m.det.score, m.det.id))
        for m in members:
            res.detection_status[m.det.id] = "new" if m is best else "duplicate"
    return res


@dataclass
class V1Associator:
    """``Associator`` wrapper around :func:`cluster_detections` (geometry only; ignores recon)."""

    k: float = DEFAULT_K
    cannot_link: bool = True
    name: str = "v1"

    def associate(
        self,
        lifted: Sequence[LiftedDetection],
        recon: Recon | None = None,
        frames: Mapping[int, Frame] | None = None,
    ) -> ClusterResult:
        return cluster_detections(lifted, k=self.k, cannot_link=self.cannot_link)
