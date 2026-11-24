"""Associate v2: geometric cross-view matching (docs/PLAN.md §8.6).

Frames are processed in increasing frame id. Each frame's detections are matched to the
running instance tracks with the Hungarian algorithm; the cost is reprojection error
normalized by box size plus an optional appearance distance. Only tracks that *should be
visible* (project inside the image and pass a depth test) are candidates. One detection
per track per frame makes cannot-link hold by construction. Afterwards a split check and
a hallucination filter run. Output is independent of input order of views and detections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from countonce.lift import frame_affine
from countonce.merge.appearance import FeatureExtractor, cosine_sim
from countonce.merge.base import ClusterResult
from countonce.types import Detection, Frame, Instance, LiftedDetection, MatchEvidence, Recon

_BIG = 1e9


@dataclass
class V2Config:
    max_reproj_norm: float = 0.5  # reject if reprojection error / box size exceeds this
    appearance_weight: float = 0.5  # weight of (1 - cos) / 2 in the cost
    min_appearance_sim: float = 0.0  # gate when both features exist
    depth_tol: float = 1.0  # depth test tolerance, in object sizes
    hallucination: Literal["off", "downweight", "drop"] = "downweight"
    hallucination_max_views: int = 1  # only instances seen in <= this many views
    hallucination_min_expected: int = 2  # expected-but-unmatched in >= this many other views
    hallucination_factor: float = 0.5  # conf multiplier for "downweight"
    split_extent_factor: float = 2.0  # split if member spread > factor * size
    split_min_sim: float = 0.3  # split if two members' appearance sim is below this
    split_check: bool = True
    cannot_link: bool = True  # False (ablation): extra same-frame detections may join a track
    use_appearance: bool = True  # False (ablation): ignore all appearance features


class _Views:
    """Camera geometry per frame id, in frame-pixel coordinates."""

    def __init__(
        self,
        recon: Recon,
        frames: Mapping[int, Frame] | None,
        frame_to_view: Mapping[int, int] | None,
    ):
        self.recon = recon
        self.h, self.w = recon.depth.shape[1:]
        self.frames = frames or {}
        self.f2v = frame_to_view
        self.v2f = (
            {v: f for f, v in frame_to_view.items()}
            if frame_to_view is not None
            else {v: v for v in range(recon.n_views)}
        )

    def view(self, frame_id: int) -> int:
        v = self.f2v[frame_id] if self.f2v is not None else frame_id
        if not 0 <= v < self.recon.n_views:
            raise ValueError(f"frame {frame_id} has no recon view")
        return v

    def affine(self, frame_id: int) -> tuple[float, float, float, float]:
        """(sx, sy, ox, oy): frame pixels -> recon pixels, exact when the frame has a grid_map."""
        return frame_affine(self.frames.get(frame_id), self.h, self.w)

    def to_frame(self, frame_id: int, uv: np.ndarray) -> np.ndarray:
        """Recon pixels -> frame pixels (inverse of `affine`)."""
        sx, sy, ox, oy = self.affine(frame_id)
        out: np.ndarray = (uv - np.array([ox, oy])) / np.array([sx, sy])
        return out

    def project(
        self, xyz: np.ndarray, size: float, view: int, depth_tol: float
    ) -> tuple[bool, np.ndarray, float]:
        """Return (visible, uv_in_depth_grid, z). Visible = inside image and passes depth test."""
        ext, k = self.recon.extrinsics[view], self.recon.intrinsics[view]
        cam = ext[:, :3] @ xyz + ext[:, 3]
        z = float(cam[2])
        if z <= 1e-9:
            return False, np.zeros(2), z
        uv = (k @ cam)[:2] / z
        u, v = float(uv[0]), float(uv[1])
        if not (0 <= u < self.w and 0 <= v < self.h):
            return False, uv, z
        d = float(self.recon.depth[view, int(v), int(u)])
        if np.isfinite(d) and d > 0 and abs(z - d) > depth_tol * max(size, 1e-9):
            return False, uv, z  # occluded (or behind the surface)
        return True, uv, z


def _det_center_scale(det: Detection) -> tuple[np.ndarray, float | None]:
    """Center (frame px) and box size (frame px; None for bare points)."""
    if det.box_xyxy is not None:
        b = np.asarray(det.box_xyxy, float)
        return (b[:2] + b[2:]) / 2, float(max(b[2] - b[0], b[3] - b[1]))
    if det.mask is not None:
        ys, xs = np.nonzero(det.mask)
        if xs.size:
            return np.array([xs.mean() + 0.5, ys.mean() + 0.5]), float(
                max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            )
    return np.asarray(det.point_xy, float), None


@dataclass
class _Track:
    members: list[LiftedDetection] = field(default_factory=list)
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: float = 1.0

    def refresh(self) -> None:
        self.xyz = np.median(np.stack([m.xyz for m in self.members]), axis=0)
        self.size = float(np.mean([_size(m) for m in self.members]))


def _size(ld: LiftedDetection) -> float:
    return float(ld.size) if ld.size is not None and ld.size > 0 else 1.0


class V2Associator:
    """Geometric cross-view matcher. Implements the ``Associator`` protocol."""

    name = "v2"

    def __init__(
        self,
        config: V2Config | None = None,
        extractor: FeatureExtractor | None = None,
        frame_to_view: Mapping[int, int] | None = None,
    ):
        self.config = config or V2Config()
        self.extractor = extractor
        self.frame_to_view = frame_to_view

    # ------------------------------------------------------------------ features
    def _features(
        self, ok: Sequence[LiftedDetection], frames: Mapping[int, Frame] | None
    ) -> dict[str, np.ndarray]:
        feats: dict[str, np.ndarray] = {}
        if not self.config.use_appearance:
            return feats
        missing: dict[int, list[Detection]] = {}
        for ld in ok:
            if ld.det.feature is not None:
                feats[ld.det.id] = np.asarray(ld.det.feature, float)
            elif self.extractor is not None:
                missing.setdefault(ld.det.frame_id, []).append(ld.det)
        extractor = self.extractor
        for fid in sorted(missing):
            assert extractor is not None  # `missing` is only filled when an extractor is set
            dets = sorted(missing[fid], key=lambda d: d.id)
            out = np.asarray(extractor.extract(frames.get(fid) if frames else None, dets))
            for d, f in zip(dets, out, strict=True):
                feats[d.id] = np.asarray(f, float)
        return feats

    # ------------------------------------------------------------------ main
    def associate(
        self,
        lifted: Sequence[LiftedDetection],
        recon: Recon | None = None,
        frames: Mapping[int, Frame] | None = None,
    ) -> ClusterResult:
        if recon is None:
            raise ValueError("V2Associator needs the Recon for projection")
        cfg = self.config
        res = ClusterResult(instances=[])
        ok: list[LiftedDetection] = []
        for ld in lifted:
            if ld.ok:
                ok.append(ld)
            else:
                res.detection_status[ld.det.id] = "rejected"
                res.reject_reasons[ld.det.id] = ld.fail_reason or "no_xyz"
        if not ok:
            return res

        geo = _Views(recon, frames, self.frame_to_view)
        feats = self._features(ok, frames)
        by_frame: dict[int, list[LiftedDetection]] = {}
        for ld in ok:
            by_frame.setdefault(ld.det.frame_id, []).append(ld)

        tracks: list[_Track] = []
        for fid in sorted(by_frame):
            dets = sorted(by_frame[fid], key=lambda m: m.det.id)
            self._match_frame(fid, dets, tracks, geo, feats, res)

        groups = [t.members for t in tracks]
        if cfg.split_check:
            groups = [g for m in groups for g in _split(m, feats, cfg)]
        self._build(groups, geo, res)
        return res

    # ------------------------------------------------------------------ matching
    def _match_frame(
        self,
        fid: int,
        dets: list[LiftedDetection],
        tracks: list[_Track],
        geo: _Views,
        feats: dict[str, np.ndarray],
        res: ClusterResult,
    ) -> None:
        cfg = self.config
        view = geo.view(fid)
        sx, sy, _, _ = geo.affine(fid)
        focal = float((geo.recon.intrinsics[view, 0, 0] + geo.recon.intrinsics[view, 1, 1]) / 2)
        n_t, n_d = len(tracks), len(dets)
        cost = np.full((n_t, n_d), _BIG)
        info: dict[tuple[int, int], tuple[float, float, float, bool]] = {}
        for ti, tr in enumerate(tracks):
            vis, uv, z = geo.project(tr.xyz, tr.size, view, cfg.depth_tol)
            if not vis:
                continue
            uv_f = geo.to_frame(fid, uv)
            for di, ld in enumerate(dets):
                center, scale = _det_center_scale(ld.det)
                if scale is None:
                    scale = tr.size * focal / max(z, 1e-9) / ((sx + sy) / 2)
                err = float(np.linalg.norm(uv_f - center))
                rn = err / max(scale, 1.0)
                sim, have = 0.0, False
                fa = [feats[m.det.id] for m in tr.members if m.det.id in feats]
                fb = feats.get(ld.det.id)
                if fa and fb is not None:
                    sim, have = float(np.mean([cosine_sim(a, fb) for a in fa])), True
                gated = rn > cfg.max_reproj_norm or (have and sim < cfg.min_appearance_sim)
                c = rn + (cfg.appearance_weight * (1.0 - sim) / 2.0 if have else 0.0)
                info[(ti, di)] = (err, sim, rn, have)
                if not gated:
                    cost[ti, di] = c

        matched: dict[int, int] = {}
        if n_t and n_d:
            rows, cols = linear_sum_assignment(cost)
            matched = {int(c): int(r) for r, c in zip(rows, cols, strict=True) if cost[r, c] < _BIG}

        if not cfg.cannot_link and n_t:
            # Ablation: unassigned detections may also join their cheapest admissible track,
            # even though that track already took a detection from this frame.
            for di in range(n_d):
                if di not in matched and (cost[:, di] < _BIG).any():
                    matched[di] = int(np.argmin(cost[:, di]))

        new_tracks: list[_Track] = []
        for di, ld in enumerate(dets):
            if di in matched:
                ti = matched[di]
                tr = tracks[ti]
                res.evidence.append(
                    self._evidence(tr, ld, info[(ti, di)], True, "matched", cost[ti, di])
                )
                tr.members.append(ld)
                continue
            # Unmatched: record the closest visible candidate (if any) as non-merged evidence.
            cands = [(info[(ti, di)][2], ti) for ti in range(n_t) if (ti, di) in info]
            if cands:
                _, ti = min(cands)
                res.evidence.append(
                    self._evidence(tracks[ti], ld, info[(ti, di)], False, "above_threshold", 0.0)
                )
            new_tracks.append(_Track(members=[ld]))
        for tr in tracks:
            tr.refresh()
        for tr in new_tracks:
            tr.refresh()
        tracks.extend(new_tracks)

    def _evidence(
        self,
        tr: _Track,
        ld: LiftedDetection,
        info: tuple[float, float, float, bool],
        merged: bool,
        reason: str,
        cost: float,
    ) -> MatchEvidence:
        err, sim, rn, have = info
        rep = min(tr.members, key=lambda m: (m.det.frame_id, m.det.id))
        d3 = float(np.linalg.norm(tr.xyz - ld.xyz)) / ((tr.size + _size(ld)) / 2)
        prob = float(np.clip(1.0 - cost / (self.config.max_reproj_norm + 1e-9), 0.0, 1.0))
        return MatchEvidence(
            det_a=rep.det.id,
            det_b=ld.det.id,
            dist_3d_norm=d3,
            reproj_error_px=err,
            appearance_sim=sim if have else 0.0,
            match_prob=prob if merged else 0.0,
            merged=merged,
            reason=reason,
        )

    # ------------------------------------------------------------------ output
    def _build(self, groups: list[list[LiftedDetection]], geo: _Views, res: ClusterResult) -> None:
        cfg = self.config
        groups = sorted(groups, key=lambda g: min(m.det.id for m in g))
        n_views = geo.recon.n_views
        next_id = 0
        for members in groups:
            xyz = np.median(np.stack([m.xyz for m in members]), axis=0)
            size = float(np.mean([_size(m) for m in members]))
            frame_ids = {m.det.frame_id for m in members}
            mean_conf = float(np.mean([m.lift_conf for m in members]))
            conf = float(np.clip(mean_conf * (1.0 - 0.5 ** len(frame_ids)), 0.0, 1.0))
            if cfg.hallucination != "off" and len(frame_ids) <= cfg.hallucination_max_views:
                expected = 0
                for v in range(n_views):
                    if geo.v2f.get(v) in frame_ids:
                        continue
                    if geo.project(xyz, size, v, cfg.depth_tol)[0]:
                        expected += 1
                if expected >= cfg.hallucination_min_expected:
                    if cfg.hallucination == "drop":
                        for m in members:
                            res.detection_status[m.det.id] = "rejected"
                            res.reject_reasons[m.det.id] = "hallucination"
                        continue
                    conf *= cfg.hallucination_factor
            res.instances.append(
                Instance(
                    id=next_id,
                    xyz=xyz,
                    size=size,
                    members=sorted(m.det.id for m in members),
                    n_views=len(frame_ids),
                    conf=conf,
                    possibly_moving=any(m.possibly_moving for m in members),
                )
            )
            next_id += 1
            best = max(members, key=lambda m: (m.lift_conf, m.det.score, m.det.id))
            for m in members:
                res.detection_status[m.det.id] = "new" if m is best else "duplicate"


def _split(
    members: list[LiftedDetection], feats: dict[str, np.ndarray], cfg: V2Config
) -> list[list[LiftedDetection]]:
    """Recursively split a track whose members are too spread out or look too different."""
    if len(members) < 2:
        return [members]
    ms = sorted(members, key=lambda m: m.det.id)
    xyz = np.stack([m.xyz for m in ms])
    size = float(np.mean([_size(m) for m in ms]))
    n = len(ms)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    sim = np.full((n, n), 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = feats.get(ms[i].det.id), feats.get(ms[j].det.id)
            if fi is not None and fj is not None:
                sim[i, j] = sim[j, i] = cosine_sim(fi, fj)

    if dist.max() > cfg.split_extent_factor * size:
        a, b = (int(v) for v in np.unravel_index(np.argmax(dist), dist.shape))
        left = [i for i in range(n) if dist[i, a] <= dist[i, b]]
    elif sim.min() < cfg.split_min_sim:
        a, b = (int(v) for v in np.unravel_index(np.argmin(sim), sim.shape))
        left = [i for i in range(n) if sim[i, a] >= sim[i, b]]
    else:
        return [members]
    a, b = min(a, b), max(a, b)
    left_set = {i for i in left if i != b} | {a}
    part_a = [ms[i] for i in range(n) if i in left_set]
    part_b = [ms[i] for i in range(n) if i not in left_set]
    return _split(part_a, feats, cfg) + _split(part_b, feats, cfg)
