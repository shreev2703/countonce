"""'Explain this match' backend (PLAN 14.8): why were two detections merged, or not?

Works from a stored ``result.json`` (schema 1.1; 1.0 documents are upgraded on load and simply
have less to report). Everything is recomputed from the persisted per-detection geometry, so any
pair can be explained, not only the pairs the associator logged evidence for.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from countonce.lift import frame_affine
from countonce.schemas import ResultV11, load_result
from countonce.types import CameraRecord, DetectionRecord, Frame, LiftedDetection, Recon

DEFAULT_K = 0.5


@dataclass
class PairExplanation:
    det_a: str
    det_b: str
    merged: bool
    same_frame: bool
    cannot_link: str  # "blocked" | "not_applicable" | "disabled"
    k: float
    associator: str
    dist_3d_norm: float | None  # 3D distance / mean object size
    reproj_a_to_b_px: float | None  # a's 3D point projected into b's view vs b's detection
    reproj_b_to_a_px: float | None
    appearance_cosine: float | None  # only when features were stored
    match_prob: float | None  # associator-logged probability (v3: calibrated)
    reason: str
    instance_a: int | None = None
    instance_b: int | None = None
    missing: list[str] = field(default_factory=list)  # what the document could not provide

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        def f(v: float | None, fmt: str = "{:.3f}") -> str:
            return "n/a" if v is None else fmt.format(v)

        lines = [
            f"{self.det_a} vs {self.det_b}: {'MERGED' if self.merged else 'NOT merged'}",
            f"  reason: {self.reason}",
            f"  3D distance / object size: {f(self.dist_3d_norm)} (k={self.k:g})",
            f"  reprojection error a->b: {f(self.reproj_a_to_b_px, '{:.1f}px')}, "
            f"b->a: {f(self.reproj_b_to_a_px, '{:.1f}px')}",
            f"  appearance cosine: {f(self.appearance_cosine)}",
            f"  match probability ({self.associator}): {f(self.match_prob)}",
            f"  cannot-link: {self.cannot_link}",
        ]
        if self.missing:
            lines.append(f"  not available: {', '.join(self.missing)}")
        return "\n".join(lines)


# ------------------------------------------------------------------ recording (pipeline side)
def build_detection_records(
    lifted: Sequence[LiftedDetection],
    recon: Recon,
    frames: Mapping[int, Frame] | None = None,
    store_features: bool = False,
) -> list[DetectionRecord]:
    """Per-detection geometry for result.json. ``frame_id`` indexes recon views (pipeline)."""
    from countonce.merge.match import _det_center_scale

    h, w = recon.depth.shape[1:]
    out = []
    for ld in lifted:
        det = ld.det
        centre, _ = _det_center_scale(det)
        sx, sy, ox, oy = frame_affine(frames.get(det.frame_id) if frames else None, h, w)
        uv = [float(centre[0] * sx + ox), float(centre[1] * sy + oy)]
        feat = None
        if store_features and det.feature is not None:
            feat = [float(x) for x in np.asarray(det.feature, float).ravel()]
        out.append(
            DetectionRecord(
                id=det.id,
                frame_id=det.frame_id,
                view=det.frame_id,
                xyz=[float(x) for x in ld.xyz] if ld.xyz is not None else None,
                size=float(ld.size) if ld.size is not None else None,
                uv=uv,
                feature=feat,
            )
        )
    return out


def build_camera_records(recon: Recon) -> list[CameraRecord]:
    return [
        CameraRecord(i, recon.intrinsics[i].tolist(), recon.extrinsics[i].tolist())
        for i in range(recon.n_views)
    ]


# ------------------------------------------------------------------ explanation
def _project(cam: Any, xyz: Sequence[float]) -> np.ndarray | None:
    k = np.asarray(cam.intrinsics, float)
    e = np.asarray(cam.extrinsics, float)
    p = e[:, :3] @ np.asarray(xyz, float) + e[:, 3]
    if p[2] <= 1e-9:
        return None
    return np.asarray((k @ p)[:2] / p[2])


def explain_pair(result: ResultV11 | dict[str, Any], det_a: str, det_b: str) -> PairExplanation:
    """Explain why ``det_a`` and ``det_b`` were (or were not) counted as one object."""
    if isinstance(result, dict):
        result = load_result(result)
    if det_a == det_b:
        raise ValueError("det_a and det_b must differ")
    known = set(result.detection_status) | {d.id for d in result.detections}
    for d in (det_a, det_b):
        if d not in known:
            raise ValueError(f"unknown detection id {d!r}")

    cfg = result.config.get("config", {}) if isinstance(result.config, dict) else {}
    k = float(cfg.get("merge_k", DEFAULT_K))
    assoc = str(cfg.get("assoc", result.model_versions.associator))
    cl_on = bool(cfg.get("cannot_link", True))
    recs = {d.id: d for d in result.detections}
    ra, rb = recs.get(det_a), recs.get(det_b)
    missing: list[str] = []

    fa, fb = (int(x.split(":")[0]) for x in (det_a, det_b))
    same_frame = fa == fb
    cannot_link = "not_applicable" if not same_frame else ("blocked" if cl_on else "disabled")

    inst_of = {m: i.id for i in result.instances for m in i.members}
    ia, ib = inst_of.get(det_a), inst_of.get(det_b)
    merged = ia is not None and ia == ib

    ev = next(
        (e for e in result.match_evidence if {e.det_a, e.det_b} == {det_a, det_b}),
        None,
    )

    dist: float | None = None
    if ra and rb and ra.xyz and rb.xyz and ra.size and rb.size:
        mean_size = (ra.size + rb.size) / 2
        dist = float(np.linalg.norm(np.subtract(ra.xyz, rb.xyz)) / max(mean_size, 1e-12))
    elif ev is not None:
        dist = ev.dist_3d_norm
    else:
        missing.append("3d_distance")

    cams = {c.view: c for c in result.cameras}
    r_ab = r_ba = None
    if ra and rb and ra.xyz and rb.xyz and rb.uv and ra.uv and ra.view in cams and rb.view in cams:
        pa = _project(cams[rb.view], ra.xyz)
        pb = _project(cams[ra.view], rb.xyz)
        if pa is not None:
            r_ab = float(np.linalg.norm(pa - np.asarray(rb.uv)))
        if pb is not None:
            r_ba = float(np.linalg.norm(pb - np.asarray(ra.uv)))
    if r_ab is None and r_ba is None:
        missing.append("reprojection_error")

    cos: float | None = None
    if ra and rb and ra.feature and rb.feature:
        fa_, fb_ = np.asarray(ra.feature, float), np.asarray(rb.feature, float)
        denom = float(np.linalg.norm(fa_) * np.linalg.norm(fb_))
        cos = float(fa_ @ fb_ / denom) if denom > 0 else None
    if cos is None:
        missing.append("appearance_cosine")

    prob = ev.match_prob if ev is not None else None
    if prob is None:
        missing.append("match_probability")

    reason = _reason(
        merged, same_frame, cl_on, dist, k, ia, det_a, det_b, result, cos, ev.reason if ev else None
    )
    return PairExplanation(
        det_a=det_a,
        det_b=det_b,
        merged=merged,
        same_frame=same_frame,
        cannot_link=cannot_link,
        k=k,
        associator=assoc,
        dist_3d_norm=dist,
        reproj_a_to_b_px=r_ab,
        reproj_b_to_a_px=r_ba,
        appearance_cosine=cos,
        match_prob=prob,
        reason=reason,
        instance_a=ia,
        instance_b=ib,
        missing=missing,
    )


def _reason(
    merged: bool,
    same_frame: bool,
    cl_on: bool,
    dist: float | None,
    k: float,
    inst: int | None,
    det_a: str,
    det_b: str,
    result: ResultV11,
    cos: float | None,
    ev_reason: str | None,
) -> str:
    if same_frame and cl_on and not merged:
        return "same frame: cannot-link"
    if merged:
        if dist is not None and math.isfinite(dist):
            return f"merged: 3D distance {dist:.1f}x object size <= k={k:g} (instance {inst})"
        return f"merged into instance {inst}"
    for d in (det_a, det_b):
        st = result.detection_status.get(d)
        if st in ("rejected", "failed"):
            why = result.reject_reasons.get(d, "unknown")
            return f"{d} was {st} ({why}), so it cannot match"
    if dist is not None and dist > k:
        return f"3D distance {dist:.1f}x object size > k={k:g}"
    if cos is not None and ev_reason:
        return f"kept separate: {ev_reason} (appearance cosine {cos:.2f})"
    if ev_reason:
        return f"kept separate: {ev_reason}"
    return "kept separate by the associator (within k, but assigned to different instances)"


def load_result_dir(out_dir: Path | str) -> ResultV11:
    path = Path(out_dir) / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return load_result(json.loads(path.read_text(encoding="utf-8")))
