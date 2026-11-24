"""Dynamic-object residual filter (docs/PLAN.md §8.3b, §9.6).

VGGT assumes a static scene, so an object that moves between views is lifted to a different
3D position per view and is counted once per position (a "ghost" duplicate).

``flag_moving`` projects every lifted detection into the other views where the depth test says
it should be visible and measures the residual to the nearest detection there, normalized by
box size. When too few of those views agree, the detection is marked ``possibly_moving``.
``MotionAwareAssociator`` wraps any associator (v1/v2/v3/sync): instances made only of flagged
detections are downweighted, and flagged instances that explain each other's disagreement (the
nearest detection in a disagreeing view is a flagged detection of another instance) are merged
as one moving object, subject to cannot-link on frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from countonce.merge.base import ClusterResult
from countonce.merge.match import _det_center_scale, _Views
from countonce.types import Frame, Instance, LiftedDetection, MatchEvidence, Recon


@dataclass
class MotionConfig:
    residual_norm: float = 0.5  # a view agrees if residual / box size <= this
    min_agree: float = 0.5  # flag when the agreeing share of checked views is below this
    min_views: int = 2  # need at least this many checked views to judge at all
    size_floor_frac: float = 0.02  # box-size floor, as a fraction of scene_scale (in pixels)
    depth_tol: float = 1.0  # depth test tolerance, in object sizes
    conf_factor: float = 0.5  # instance conf multiplier for fully flagged instances


def _links(
    lifted: Sequence[LiftedDetection],
    recon: Recon,
    frames: Mapping[int, Frame] | None,
    frame_to_view: Mapping[int, int] | None,
    cfg: MotionConfig,
) -> dict[str, tuple[float | None, bool, list[str]]]:
    """Per id: (median residual, flagged, nearest detection ids in disagreeing views)."""
    geo = _Views(recon, frames, frame_to_view)
    ok = [ld for ld in lifted if ld.ok]
    by_frame: dict[int, list[LiftedDetection]] = {}
    for ld in ok:
        by_frame.setdefault(ld.det.frame_id, []).append(ld)
    centers = {
        fid: [(ld, *_det_center_scale(ld.det)) for ld in sorted(g, key=lambda m: m.det.id)]
        for fid, g in by_frame.items()
    }
    out: dict[str, tuple[float | None, bool, list[str]]] = {}
    floor_depth = max(float(recon.scene_scale), 1e-9)
    for ld in ok:
        assert ld.xyz is not None
        size = ld.size if ld.size is not None and ld.size > 0 else 1.0
        residuals: list[float] = []
        partners: list[str] = []
        n_agree = 0
        for fid in sorted(by_frame):
            if fid == ld.det.frame_id:
                continue
            view = geo.view(fid)
            vis, uv, z = geo.project(ld.xyz, size, view, cfg.depth_tol)
            if not vis:
                continue
            uv_f = geo.to_frame(fid, uv)
            best: tuple[float, str, float | None] | None = None
            for other, c, scale in centers[fid]:
                d = float(np.linalg.norm(uv_f - c))
                if best is None or d < best[0]:
                    best = (d, other.det.id, scale)
            assert best is not None
            err, pid, scale = best
            focal = float((recon.intrinsics[view, 0, 0] + recon.intrinsics[view, 1, 1]) / 2)
            if scale is None:
                scale = size * focal / max(z, 1e-9)
            scale = max(scale, cfg.size_floor_frac * floor_depth * focal / max(z, 1e-9), 1.0)
            r = err / scale
            residuals.append(r)
            if r <= cfg.residual_norm:
                n_agree += 1
            else:
                partners.append(pid)
        n = len(residuals)
        flagged = n >= cfg.min_views and n_agree / n < cfg.min_agree
        out[ld.det.id] = (float(np.median(residuals)) if n else None, flagged, partners)
    return out


def flag_moving(
    lifted: Sequence[LiftedDetection],
    recon: Recon,
    frames: Mapping[int, Frame] | None = None,
    cfg: MotionConfig | None = None,
    frame_to_view: Mapping[int, int] | None = None,
) -> list[LiftedDetection]:
    """Copies of ``lifted`` with ``motion_residual`` and ``possibly_moving`` filled in."""
    info = _links(lifted, recon, frames, frame_to_view, cfg or MotionConfig())
    res = []
    for ld in lifted:
        got = info.get(ld.det.id)
        res.append(
            ld if got is None else replace(ld, motion_residual=got[0], possibly_moving=got[1])
        )
    return res


class MotionAwareAssociator:
    """Wraps an associator: downweights flagged detections and merges moving-object ghosts."""

    def __init__(
        self,
        inner: object,
        cfg: MotionConfig | None = None,
        frame_to_view: Mapping[int, int] | None = None,
    ):
        self.inner = inner
        self.cfg = cfg or MotionConfig()
        self.frame_to_view = frame_to_view
        self.name = getattr(inner, "name", "assoc")

    def associate(
        self,
        lifted: Sequence[LiftedDetection],
        recon: Recon | None = None,
        frames: Mapping[int, Frame] | None = None,
    ) -> ClusterResult:
        if recon is None:
            return self.inner.associate(lifted, recon, frames)  # type: ignore[attr-defined,no-any-return]
        info = _links(lifted, recon, frames, self.frame_to_view, self.cfg)
        flagged_lifted = [
            replace(ld, motion_residual=info[ld.det.id][0], possibly_moving=info[ld.det.id][1])
            if ld.det.id in info
            else ld
            for ld in lifted
        ]
        res: ClusterResult = self.inner.associate(flagged_lifted, recon, frames)  # type: ignore[attr-defined]
        return self._merge_ghosts(res, flagged_lifted, info)

    def _merge_ghosts(
        self,
        res: ClusterResult,
        lifted: Sequence[LiftedDetection],
        info: dict[str, tuple[float | None, bool, list[str]]],
    ) -> ClusterResult:
        by_id = {ld.det.id: ld for ld in lifted}
        moving = [
            i
            for i, inst in enumerate(res.instances)
            if all(by_id[m].possibly_moving for m in inst.members)
        ]
        member_of = {m: i for i in moving for m in res.instances[i].members}
        frames_of = {i: {by_id[m].det.frame_id for m in res.instances[i].members} for i in moving}
        parent = {i: i for i in moving}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged_pairs: list[tuple[str, str]] = []
        for i in moving:
            for m in sorted(res.instances[i].members):
                for pid in info.get(m, (None, False, []))[2]:
                    j = member_of.get(pid)
                    if j is None:
                        continue
                    ri, rj = find(i), find(j)
                    if ri == rj or frames_of[ri] & frames_of[rj]:
                        continue  # cannot-link: one object is never twice in the same frame
                    parent[rj] = ri
                    frames_of[ri] |= frames_of[rj]
                    merged_pairs.append((m, pid))

        groups: dict[int, list[int]] = {}
        for i in moving:
            groups.setdefault(find(i), []).append(i)
        absorbed = {i for g in groups.values() for i in g[1:]}
        out = ClusterResult(
            instances=[],
            detection_status=dict(res.detection_status),
            reject_reasons=dict(res.reject_reasons),
            evidence=list(res.evidence),
        )
        for a, b in merged_pairs:
            out.evidence.append(MatchEvidence(a, b, 0.0, 0.0, 0.0, 0.0, True, "motion_ghost_merge"))
        for i, inst in enumerate(res.instances):
            if i in absorbed:
                continue
            if i in groups:
                members = sorted(m for j in groups[i] for m in res.instances[j].members)
                parts = [res.instances[j] for j in groups[i]]
                n_views = len(frames_of[find(i)])
                conf = float(np.mean([p.conf for p in parts]))
                xyz = np.median(np.stack([p.xyz for p in parts]), axis=0)
                inst = Instance(
                    id=0,
                    xyz=xyz,
                    size=float(np.mean([p.size for p in parts])),
                    members=members,
                    n_views=n_views,
                    conf=float(np.clip(conf * self.cfg.conf_factor, 0.0, 1.0)),
                    possibly_moving=True,
                )
                best = max(
                    (by_id[m] for m in members), key=lambda m: (m.lift_conf, m.det.score, m.det.id)
                )
                for m in members:
                    out.detection_status[m] = "new" if m == best.det.id else "duplicate"
            out.instances.append(inst)
        out.instances = [replace(x, id=k) for k, x in enumerate(out.instances)]
        return out
