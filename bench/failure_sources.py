"""Controlled failure-source harness (PLAN §9.6, §0.2 M6).

Which stage causes the count error: detection, geometry, or association? The harness answers by
swapping each upstream stage for an oracle, one at a time, on the same scene:

=====================  ==================  ===================
condition              detections          geometry
=====================  ==================  ===================
``oracle_det__est_geom``   GT objects projected    estimated (reconstruction)
``real_det__oracle_geom``  real detector            oracle (dataset cameras / depth)
``full``                   real detector            estimated
``oracle_det__oracle_geom``  GT objects projected    oracle (ceiling: association only)
=====================  ==================  ===================

Oracle detections are the GT 3D objects projected into every photo with the oracle cameras and kept
only where the oracle depth map shows them (not occluded). Detections are lifted **once per
condition**; every associator then receives that same ``lifted`` list and ``recon`` object, and the
rows carry an ``input_hash`` of it so identical upstream inputs are checkable. Associator settings
come from the caller (tuned on a validation split with :func:`tune_associators`, not defaults).

A scene is described by a :class:`SceneBundle`; builders exist for the bundled tiny scene
(:func:`tiny_scene_bundle`) and for a synthetic episode (:func:`synth_bundle`). Fuji-SfM and own
captures only need to fill a bundle (GT positions + radius, oracle recon, estimated recon, real
detections); no harness code changes.

Count-first metrics: signed/absolute error and the exact decomposition of
``bench.metrics.count_decomposition`` (missed, duplicates, false positives, wrong merges).
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.gt_matching import match_detections_to_gt, project_points
from bench.metrics import count_decomposition, signed_bias
from countonce.lift import LiftKwargs, lift_detections
from countonce.merge.base import Associator
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator, V2Config
from countonce.merge.sync import SyncAssociator, SyncConfig
from countonce.recon.adapter import unproject_depth
from countonce.types import Detection, Frame, LiftedDetection, Recon

CONDITIONS: dict[str, tuple[str, str]] = {  # name -> (detections, geometry)
    "oracle_det__oracle_geom": ("oracle", "oracle"),
    "oracle_det__est_geom": ("oracle", "estimated"),
    "real_det__oracle_geom": ("real", "oracle"),
    "full": ("real", "estimated"),
}
REQUIRED_CONDITIONS = ("oracle_det__est_geom", "real_det__oracle_geom", "full")

V1_GRID = {"k": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]}
V2_GRID = {"max_reproj_norm": [0.25, 0.5, 1.0, 2.0], "depth_tol": [0.5, 1.0, 2.0, 4.0]}
SYNC_GRID = {"eig_threshold": [0.3, 0.5, 0.7], "assign_threshold": [0.3, 0.5, 0.7]}


@dataclass
class SceneBundle:
    """Everything the harness needs about one scene.

    ``cam_K`` / ``cam_E`` are the oracle cameras in **frame pixels** (``(V,3,3)``, ``(V,3,4)``
    world->camera). ``oracle_recon`` / ``estimated_recon`` live on their own (possibly smaller)
    depth grid; the estimated one may be in a different world frame and scale. ``surface_offset``
    says how far in front of an object's centre its visible surface lies, in radii (1 for a sphere,
    0 for a flat disc); it is only used for the oracle visibility test.
    """

    name: str
    prompt: str
    frames: dict[int, Frame]
    gt_ids: list[str]
    gt_xyz: np.ndarray
    gt_radius: float
    cam_K: np.ndarray
    cam_E: np.ndarray
    oracle_recon: Recon
    estimated_recon: Recon
    real_dets: list[Detection]
    real_identity: dict[str, str | None] | None = None  # None: nearest projected GT within match_px
    surface_offset: float = 1.0
    match_px: float = 20.0
    lift_kw: LiftKwargs = field(default_factory=dict)

    @property
    def true_count(self) -> int:
        return len(self.gt_ids)


# --------------------------------------------------------------------------- oracle detections


def oracle_detections(b: SceneBundle) -> tuple[list[Detection], dict[str, str | None]]:
    """GT objects projected into every photo, kept where the oracle depth shows them."""
    dets: list[Detection] = []
    ident: dict[str, str | None] = {}
    rec = b.oracle_recon
    gh, gw = rec.depth.shape[1:]
    for fid, frame in sorted(b.frames.items()):
        uv, z = project_points(b.gt_xyz, b.cam_K[fid], b.cam_E[fid])
        f = float((b.cam_K[fid][0, 0] + b.cam_K[fid][1, 1]) / 2)
        sx, sy = gw / frame.width, gh / frame.height
        k = 0
        for j in range(len(b.gt_ids)):
            if z[j] <= 0:
                continue
            half = f * b.gt_radius / z[j]
            u, v = uv[j]
            if not (0 <= u < frame.width and 0 <= v < frame.height):
                continue
            d_at = float(rec.depth[fid, min(int(v * sy), gh - 1), min(int(u * sx), gw - 1)])
            if not np.isfinite(d_at) or d_at <= 0:
                continue
            if d_at < z[j] - (b.surface_offset + 0.5) * b.gt_radius:
                continue  # something nearer hides the object's centre
            did = Detection.make_id(fid, k)
            dets.append(
                Detection(
                    id=did,
                    frame_id=fid,
                    score=1.0,
                    box_xyxy=np.array([u - half, v - half, u + half, v + half], dtype=np.float64),
                )
            )
            ident[did] = b.gt_ids[j]
            k += 1
    return dets, ident


def real_identities(b: SceneBundle) -> dict[str, str | None]:
    if b.real_identity is not None:
        return b.real_identity
    return match_detections_to_gt(
        b.real_dets,
        b.gt_ids,
        b.gt_xyz,
        {i: b.cam_K[i] for i in b.frames},
        {i: b.cam_E[i] for i in b.frames},
        {i: (f.width, f.height) for i, f in b.frames.items()},
        threshold_px=b.match_px,
    )


# ------------------------------------------------------------------------------ lifting, scoring


@dataclass
class Upstream:
    """One condition's identical associator input."""

    condition: str
    lifted: list[LiftedDetection]
    recon: Recon
    identity: dict[str, str | None]
    n_detections: int
    n_lift_failed: int
    input_hash: str


def _hash_inputs(lifted: Sequence[LiftedDetection], recon: Recon) -> str:
    h = hashlib.sha256()
    for ld in lifted:
        h.update(ld.det.id.encode())
        if ld.xyz is not None:
            h.update(np.ascontiguousarray(ld.xyz, dtype=np.float64).tobytes())
        h.update(repr((ld.size, ld.fail_reason)).encode())
    h.update(np.ascontiguousarray(recon.extrinsics, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def build_upstream(b: SceneBundle, condition: str) -> Upstream:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}")
    det_src, geom_src = CONDITIONS[condition]
    if det_src == "oracle":
        dets, ident = oracle_detections(b)
    else:
        dets, ident = list(b.real_dets), real_identities(b)
    recon = b.oracle_recon if geom_src == "oracle" else b.estimated_recon
    lifted = lift_detections(dets, recon, b.frames, **b.lift_kw)
    return Upstream(
        condition=condition,
        lifted=lifted,
        recon=recon,
        identity=ident,
        n_detections=len(dets),
        n_lift_failed=sum(1 for ld in lifted if not ld.ok),
        input_hash=_hash_inputs(lifted, recon),
    )


def score(b: SceneBundle, up: Upstream, res: Any) -> dict[str, float]:
    cluster = {m: k for k, inst in enumerate(res.instances) for m in inst.members}
    # Lift failures never reach an associator; they are not clustered, so leave them out of the
    # decomposition (an object seen only through failed lifts counts as missed).
    ids = [ld.det.id for ld in up.lifted if ld.ok]
    labels = [cluster.get(i, -1 - n) for n, i in enumerate(ids)]  # unclustered = own cluster
    gt = [up.identity.get(i) for i in ids]
    dec = count_decomposition(labels, gt, b.gt_ids)
    pred, truth = float(len(res.instances)), float(b.true_count)
    return {
        "pred": pred,
        "gt": truth,
        "abs_err": abs(pred - truth),
        "signed_err": pred - truth,
        **{k: float(v) for k, v in dec.items()},
    }


def run_scene(
    b: SceneBundle,
    associators: Mapping[str, Associator],
    conditions: Sequence[str] = tuple(CONDITIONS),
) -> list[dict[str, Any]]:
    """Rows for one scene: every condition x every associator, one shared upstream per condition."""
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        up = build_upstream(b, cond)
        for name, assoc in associators.items():
            res = assoc.associate(up.lifted, up.recon, b.frames)
            rows.append(
                {
                    "scene": b.name,
                    "condition": cond,
                    "method": name,
                    "n_detections": up.n_detections,
                    "n_lift_failed": up.n_lift_failed,
                    "input_hash": up.input_hash,
                    **score(b, up, res),
                }
            )
    return rows


def run_bundles(
    bundles: Sequence[SceneBundle],
    associators: Mapping[str, Associator],
    conditions: Sequence[str] = tuple(CONDITIONS),
) -> list[dict[str, Any]]:
    return [r for b in bundles for r in run_scene(b, associators, conditions)]


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (condition, method): count-first means over scenes."""
    out = []
    keys = sorted({(r["condition"], r["method"]) for r in rows})
    for cond, method in keys:
        sel = [r for r in rows if r["condition"] == cond and r["method"] == method]
        row: dict[str, Any] = {"condition": cond, "method": method, "n_scenes": len(sel)}
        row["mae"] = float(np.mean([r["abs_err"] for r in sel]))
        row["signed_bias"] = signed_bias([r["pred"] for r in sel], [r["gt"] for r in sel])
        for k in ("missed", "duplicates", "false_positives", "wrong_merges", "n_lift_failed"):
            row[k] = float(np.mean([r[k] for r in sel]))
        out.append(row)
    return out


def attribution(summary: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per method: MAE under each condition, and how much each oracle removes from the full MAE.

    ``removed_by_oracle_detections = mae(full) - mae(oracle_det__est_geom)``;
    ``removed_by_oracle_geometry = mae(full) - mae(real_det__oracle_geom)``. Errors interact and
    can cancel, so the two need not add up, and a negative value means the oracle made it worse.
    """
    by = {(r["condition"], r["method"]): r["mae"] for r in summary}
    out = []
    for m in sorted({r["method"] for r in summary}):
        get = lambda c, m=m: by.get((c, m), float("nan"))  # noqa: E731
        full = get("full")
        out.append(
            {
                "method": m,
                "mae_full": full,
                "mae_oracle_det__est_geom": get("oracle_det__est_geom"),
                "mae_real_det__oracle_geom": get("real_det__oracle_geom"),
                "mae_oracle_both": get("oracle_det__oracle_geom"),
                "removed_by_oracle_detections": full - get("oracle_det__est_geom"),
                "removed_by_oracle_geometry": full - get("real_det__oracle_geom"),
            }
        )
    return out


# ----------------------------------------------------------------------------------- tuning


def _grid(space: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*space.values())]


def make_v1(p: Mapping[str, Any]) -> Associator:
    return V1Associator(**p)


def make_v2(p: Mapping[str, Any]) -> Associator:
    return V2Associator(V2Config(**p))


def make_sync(p: Mapping[str, Any]) -> Associator:
    return SyncAssociator(SyncConfig(**p), V2Config())


FACTORIES: dict[str, tuple[Callable[[Mapping[str, Any]], Associator], dict[str, list[Any]]]] = {
    "v1": (make_v1, V1_GRID),
    "v2": (make_v2, V2_GRID),
    "sync": (make_sync, SYNC_GRID),
}


def tune_associators(
    val: Sequence[SceneBundle],
    condition: str = "full",
    methods: Sequence[str] = ("v1", "v2", "sync"),
) -> dict[str, dict[str, Any]]:
    """Grid-search each method on the validation bundles (min mean |count error|, ties: earlier).

    Tuning uses one condition (default the full pipeline) with identical upstream inputs for every
    candidate. Call this on validation scenes only, then freeze the result before the test run.
    """
    ups = [(b, build_upstream(b, condition)) for b in val]
    best: dict[str, dict[str, Any]] = {}
    for m in methods:
        build, space = FACTORIES[m]
        scored = []
        for p in _grid(space):
            assoc = build(p)
            mae = float(
                np.mean(
                    [
                        score(b, up, assoc.associate(up.lifted, up.recon, b.frames))["abs_err"]
                        for b, up in ups
                    ]
                )
            )
            scored.append((p, mae))
        best[m] = dict(min(scored, key=lambda t: t[1])[0])
    return best


def build_associators(
    tuned: Mapping[str, Mapping[str, Any]] | None = None, v3_checkpoint: str | None = None
) -> dict[str, Associator]:
    """v1/v2/sync at tuned (else default) settings; v3 added when a checkpoint is given."""
    tuned = tuned or {}
    out = {m: FACTORIES[m][0](tuned.get(m, {})) for m in FACTORIES}
    if v3_checkpoint:
        from countonce.merge.v3 import V3Associator, load_cached

        out["v3"] = V3Associator(model=load_cached(v3_checkpoint))
    return out


# --------------------------------------------------------------------------- degraded geometry


def perturb_recon(
    recon: Recon,
    *,
    rot_deg: float = 1.5,
    trans: float = 0.05,
    depth_scale_sigma: float = 0.05,
    depth_noise: float = 0.01,
    seed: int = 0,
) -> Recon:
    """A stand-in 'estimated' reconstruction: per-view pose jitter and depth scale drift.

    Each camera is rotated about an arbitrary axis by up to ``rot_deg`` and its centre moved by
    ``trans`` (world units), depth gets a per-view log-normal scale and per-pixel noise, and the
    world points are re-derived from the perturbed cameras, so views disagree as they do in a real
    reconstruction. Deterministic in ``seed``. Use a real backbone instead on GPU boxes.
    """
    rng = np.random.default_rng(seed)
    ext = recon.extrinsics.astype(np.float64).copy()
    depth = recon.depth.astype(np.float64).copy()
    for v in range(recon.n_views):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = np.deg2rad(rot_deg) * rng.uniform(0.3, 1.0)
        kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        dr = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * (kx @ kx)
        r, t = ext[v, :, :3], ext[v, :, 3]
        centre = -r.T @ t + rng.normal(0, trans, 3)
        r2 = dr @ r
        ext[v, :, :3], ext[v, :, 3] = r2, -r2 @ centre
        scale = float(np.exp(rng.normal(0, depth_scale_sigma)))
        noise = 1.0 + rng.normal(0, depth_noise, depth[v].shape)
        depth[v] = np.where(depth[v] > 0, depth[v] * scale * noise, depth[v])
    return replace(
        recon,
        backbone=f"{recon.backbone}+perturbed",
        extrinsics=ext,
        depth=depth.astype(np.float32),
        world_points=unproject_depth(depth, recon.intrinsics, ext),
    )


# --------------------------------------------------------------------------------- bundles


def tiny_scene_bundle(
    scene_dir: Path | str,
    prompt: str = "red sphere",
    *,
    detections: Sequence[Detection] | None = None,
    estimated: Recon | None = None,
    seed: int = 0,
) -> SceneBundle:
    """The bundled tiny scene: oracle = its stored cameras and depth, real = colour-blob detector.

    ``estimated`` defaults to a perturbed copy of the oracle (CPU, no weights); pass a real
    reconstruction (e.g. VGGT on the five images) on a GPU box. ``detections`` likewise overrides
    the colour-blob stand-in (e.g. real GroundingDINO output).
    """
    import json

    from countonce.detect.color_blob import ColorBlobDetector
    from countonce.pipeline import make_frames
    from countonce.recon.fixture import FixtureReconstructor

    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir / "scene.json").read_text())
    paths = [scene_dir / v["image"] for v in scene["views"]]
    frame_list = make_frames(paths)
    frames = {f.id: f for f in frame_list}
    oracle = FixtureReconstructor(scene_dir / "scene.json").reconstruct(frame_list)
    colour = prompt.lower().split()[0]
    gt = [s for s in scene["spheres"] if s["colour"] == colour]
    dets = (
        list(detections)
        if detections is not None
        else ColorBlobDetector().detect(frame_list, prompt)
    )
    return SceneBundle(
        name="tiny_scene",
        prompt=prompt,
        frames=frames,
        gt_ids=[f"gt{i}" for i in range(len(gt))],
        gt_xyz=np.array([s["xyz"] for s in gt], dtype=np.float64),
        gt_radius=float(gt[0]["radius"]),
        cam_K=oracle.intrinsics,
        cam_E=oracle.extrinsics,
        oracle_recon=oracle,
        estimated_recon=estimated if estimated is not None else perturb_recon(oracle, seed=seed),
        real_dets=dets,
        surface_offset=1.0,
        lift_kw={"min_pixels": 3, "conf_percentile": 0.0},
    )


def synth_bundle(episode: Any, *, seed: int = 0) -> SceneBundle:
    """A ``synth.geometry`` episode: oracle = true cameras + exact depth, real = noisy detections.

    Detections and depth are rendered at the episode's 640 px image size with the 64 px stand-in
    recon grid of ``bench.synth_component``. Real detections carry the episode's own identity
    labels (``None`` for injected false positives).
    """
    from bench.synth_assoc import GRID
    from bench.synth_component import DepthModel, render_depth
    from countonce.match.model import NOMINAL_OBJECT_SIZE
    from synth.geometry import IMAGE_SIZE, load_episode

    ep = episode if hasattr(episode, "K") else load_episode(episode)
    s = GRID / IMAGE_SIZE
    scale_k = np.diag([s, s, 1.0])
    k_grid = np.einsum("ij,vjk->vik", scale_k, ep.K)
    ext = np.concatenate([ep.R, ep.t[:, :, None]], axis=2)

    def make(depth: np.ndarray, extr: np.ndarray) -> Recon:
        conf = np.ones_like(depth, dtype=np.float32)
        return Recon(
            backbone="synthetic",
            intrinsics=k_grid,
            extrinsics=extr,
            depth=depth.astype(np.float32),
            depth_conf=conf,
            world_points=unproject_depth(depth.astype(np.float64), k_grid, extr),
            world_points_conf=conf.copy(),
        )

    exact = render_depth(ep, DepthModel(noise=0.0))
    oracle = make(exact, ext)
    est = perturb_recon(oracle, rot_deg=1.0, trans=0.03, depth_scale_sigma=0.04, seed=seed)
    est = replace(est, depth=render_depth(ep, DepthModel(noise=0.02)))
    est = replace(
        est, world_points=unproject_depth(est.depth.astype(np.float64), k_grid, est.extrinsics)
    )

    present = sorted(int(g) for g in np.unique(ep.identity[ep.identity >= 0]))  # = ep.true_count()
    size = int(IMAGE_SIZE)
    frames = {
        v: Frame(id=v, path=Path("synthetic"), width=size, height=size, sha256="")
        for v in range(ep.n_views)
    }
    half = ep.K[ep.view, 0, 0] * NOMINAL_OBJECT_SIZE / 2 / np.maximum(ep.depth, 1e-6)
    dets, ident = [], {}
    counts: dict[int, int] = {}
    for i in range(ep.n_detections):
        v = int(ep.view[i])
        k = counts.get(v, 0)
        counts[v] = k + 1
        did = Detection.make_id(v, k)
        u, w = ep.uv[i]
        dets.append(
            Detection(
                id=did,
                frame_id=v,
                score=float(ep.score[i]),
                box_xyxy=np.array([u - half[i], w - half[i], u + half[i], w + half[i]]),
            )
        )
        ident[did] = f"gt{int(ep.identity[i])}" if ep.identity[i] >= 0 else None
    return SceneBundle(
        name=f"synth_{ep.seed}",
        prompt="object",
        frames=frames,
        gt_ids=[f"gt{i}" for i in present],
        gt_xyz=ep.points3d[present],
        gt_radius=NOMINAL_OBJECT_SIZE / 2,
        cam_K=ep.K,
        cam_E=ext,
        oracle_recon=oracle,
        estimated_recon=est,
        real_dets=dets,
        real_identity=ident,
        surface_offset=0.0,
        lift_kw={"min_pixels": 1, "conf_percentile": 0.0},
    )
