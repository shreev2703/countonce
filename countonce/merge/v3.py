"""Associate v3: learned multi-view matcher (``assoc: v3``, PLAN §10.3).

Builds per-detection features from the lifted detections and the reconstruction, runs the
attentional GNN (``match/model.py``), and forms instances with cannot-link-safe components
(``match/instances.py``). torch is imported only when a model is loaded or run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from countonce.match.instances import form_instances
from countonce.match.model import DetInputs, load_checkpoint, mask_inputs
from countonce.merge.base import ClusterResult
from countonce.merge.sync import SyncConfig
from countonce.merge.sync import synchronize as spectral_sync
from countonce.types import Detection, Frame, Instance, LiftedDetection, MatchEvidence, Recon


def require_checkpoint(path: Path | str | None) -> Path:
    """Resolve the v3 checkpoint, with a clear error when it is unset or missing."""
    if path is None:
        raise FileNotFoundError(
            "assoc: v3 needs a trained matcher checkpoint; set Config.v3_checkpoint "
            "(YAML key `v3_checkpoint`)"
        )
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"v3 matcher checkpoint not found: {p}")
    return p


@lru_cache(maxsize=2)
def _load_resolved(path: str) -> Any:
    return load_checkpoint(path)


def load_cached(path: Path | str | None) -> Any:
    """Load the checkpoint once per path (the sensitivity grid builds many associators)."""
    return _load_resolved(str(require_checkpoint(path).resolve()))


def _box_cxcywh(det: Detection, w: float, h: float) -> np.ndarray:
    if det.box_xyxy is not None:
        x0, y0, x1, y1 = (float(x) for x in det.box_xyxy)
    elif det.point_xy is not None:
        x0 = x1 = float(det.point_xy[0])
        y0 = y1 = float(det.point_xy[1])
    else:
        assert det.mask is not None  # guaranteed by Detection.__post_init__
        ys, xs = np.nonzero(det.mask)
        x0, x1, y0, y1 = float(xs.min()), float(xs.max() + 1), float(ys.min()), float(ys.max() + 1)
    return np.array([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])


def build_inputs(
    ok: Sequence[LiftedDetection],
    recon: Recon,
    frames: Mapping[int, Frame] | None,
    frame_to_view: Mapping[int, int] | None = None,
) -> DetInputs:
    """Model inputs from lifted detections (xyz / scene_scale, rays and cameras from ``recon``)."""
    from countonce.merge.match import _det_center_scale, _Views

    geo = _Views(recon, frames, frame_to_view)
    scale = float(recon.scene_scale) if recon.scene_scale and recon.scene_scale > 0 else 1.0
    xyz = np.stack([ld.xyz for ld in ok]).astype(float)
    shift = xyz.mean(axis=0)
    rot, trans = recon.extrinsics[:, :, :3], recon.extrinsics[:, :, 3]
    centres = -np.einsum("nji,nj->ni", rot, trans)

    rays, cams, boxes, views = [], [], [], []
    for ld in ok:
        det = ld.det
        view = geo.view(det.frame_id)
        center, _ = _det_center_scale(det)
        sx, sy, ox, oy = geo.affine(det.frame_id)
        u, v = center[0] * sx + ox, center[1] * sy + oy
        d_cam = np.linalg.inv(recon.intrinsics[view]) @ np.array([u, v, 1.0])
        ray = rot[view].T @ d_cam
        rays.append(ray / max(np.linalg.norm(ray), 1e-12))
        cams.append((centres[view] - shift) / scale)
        fr = geo.frames.get(det.frame_id)
        w, h = (fr.width, fr.height) if fr is not None else (geo.w, geo.h)
        boxes.append(_box_cxcywh(det, float(w), float(h)))
        views.append(view)

    known = [ld.size for ld in ok if ld.size is not None and ld.size > 0]
    fallback = float(np.median(known)) if known else 1.0
    size = np.array([ld.size if ld.size is not None and ld.size > 0 else fallback for ld in ok])
    app = None
    if all(ld.det.feature is not None for ld in ok):
        app = np.stack([np.asarray(ld.det.feature, float) for ld in ok])
    return DetInputs(
        xyz=(xyz - shift) / scale,
        size=size / scale,
        box=np.stack(boxes),
        score=np.array([ld.det.score for ld in ok], float),
        lift_conf=np.array([ld.lift_conf for ld in ok], float),
        ray=np.stack(rays),
        cam=np.stack(cams),
        view=np.array(views, dtype=np.int64),
        appearance=app,
    )


class V3Associator:
    """Learned associator (``assoc: v3``). Pass a loaded ``model`` or a ``checkpoint`` path."""

    name = "v3"

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        *,
        model: Any = None,
        threshold: float = 0.5,
        frame_to_view: Mapping[int, int] | None = None,
        device: str = "cpu",
        synchronize: bool = False,
        feature_group: str = "full",
    ):
        if model is None:
            model = load_checkpoint(require_checkpoint(checkpoint), device)
        self.model = model
        self.threshold = threshold
        self.frame_to_view = frame_to_view
        self.synchronize = synchronize  # spectral sync of the match matrix (§10.3 fix 2)
        self.feature_group = feature_group

    def associate(
        self,
        lifted: Sequence[LiftedDetection],
        recon: Recon | None = None,
        frames: Mapping[int, Frame] | None = None,
    ) -> ClusterResult:
        import torch

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
        if recon is None:
            raise ValueError("assoc: v3 needs the reconstruction (cameras) to build features")

        inputs = mask_inputs(
            build_inputs(ok, recon, frames, self.frame_to_view),
            self.feature_group,
            int(self.model.hparams.get("appearance_dim", 0)),
        )
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            probs = self.model.match_probs(inputs).cpu().numpy()
        self.model.train(was_training)
        if self.synchronize:
            labels = spectral_sync(probs, inputs.view, SyncConfig())
        else:
            labels = form_instances(probs, inputs.view, self.threshold)

        xyz = np.stack([ld.xyz for ld in ok]).astype(float)
        known = [ld.size for ld in ok if ld.size is not None and ld.size > 0]
        fallback = float(np.median(known)) if known else 1.0
        sizes = np.array([ld.size if ld.size else fallback for ld in ok], dtype=float)
        for k, lab in enumerate(np.unique(labels)):
            idxs = np.flatnonzero(labels == lab)
            members = [ok[i] for i in idxs]
            n_views = len({m.det.frame_id for m in members})
            mean_conf = float(np.mean([m.lift_conf for m in members]))
            conf = float(np.clip(mean_conf * (1.0 - 0.5**n_views), 0.0, 1.0))
            res.instances.append(
                Instance(
                    id=k,
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
        ii, jj = np.where(np.triu(probs > self.threshold, 1))
        for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
            res.evidence.append(
                MatchEvidence(
                    det_a=ok[i].det.id,
                    det_b=ok[j].det.id,
                    dist_3d_norm=float(
                        np.linalg.norm(xyz[i] - xyz[j]) / ((sizes[i] + sizes[j]) / 2.0)
                    ),
                    reproj_error_px=0.0,
                    appearance_sim=0.0,
                    match_prob=float(probs[i, j]),
                    merged=bool(labels[i] == labels[j]),
                    reason="merged" if labels[i] == labels[j] else "cannot_link_split",
                )
            )
        return res
