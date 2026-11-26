"""Synthetic component test (PLAN §0.2 M5): tuned baselines, finite noisy depth, count-first.

Extends ``bench.synth_assoc`` with two things the first v3 comparison lacked:

* a **finite noisy-depth variant**: the stand-in reconstruction carries a z-buffered depth map
  (instances occlude one another, a ground plane fills the background, multiplicative noise),
  so the v2/sync visibility check is active. The original variant keeps NaN depth (check inert);
* **tuning on the validation split**: v1 / v2 / sync get a small grid search minimising mean
  absolute count error on ``val``; v3 stays at its trained default (it is not tuned here).

Metrics are count-first: MAE, signed bias, within-10%, the exact error decomposition
(missed / duplicates / false-positive instances / wrong merges, plus the number of injected
false-positive detections fed in); association F1 is explanatory. ``true`` is the number of objects
with at least one detection (``Episode.true_count``), as in ``bench.synth_assoc``.
This is a component test on generated geometry, not evidence about real photos.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.metrics import (
    count_decomposition,
    count_summary,
    duplicate_rate,
    miss_rate,
    pairwise_prf,
)
from bench.synth_assoc import GRID, episode_to_lifted
from countonce.match.model import NOMINAL_OBJECT_SIZE
from countonce.merge.base import Associator
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator, V2Config
from countonce.merge.sync import SyncAssociator, SyncConfig
from countonce.types import LiftedDetection, Recon
from synth.geometry import IMAGE_SIZE, Episode, load_episode

FAR_DEPTH = 10.0

# ``score_threshold`` drops detections scoring below it before association (0.0 keeps all).
SCORE_GRID = [0.0, 0.4, 0.6, 0.8]
V1_GRID = {"k": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "score_threshold": SCORE_GRID}
V2_GRID = {
    "max_reproj_norm": [0.25, 0.5, 1.0, 2.0],
    "depth_tol": [0.5, 1.0, 2.0, 4.0],
    "score_threshold": SCORE_GRID,
}
SYNC_GRID = {
    "eig_threshold": [0.3, 0.5, 0.7],
    "assign_threshold": [0.3, 0.5, 0.7],
    "score_threshold": SCORE_GRID,
}


class ScoreGated:
    """Wrap an associator so detections scoring below ``threshold`` are never associated."""

    def __init__(self, inner: Associator, threshold: float):
        self.inner = inner
        self.threshold = threshold
        self.name = inner.name

    def associate(self, lifted: Any, recon: Any = None, frames: Any = None) -> Any:
        keep = [ld for ld in lifted if ld.det.score >= self.threshold]
        return self.inner.associate(keep, recon, frames)


@dataclass(frozen=True)
class DepthModel:
    noise: float = 0.02  # multiplicative per-pixel depth noise (matches the detection depth noise)
    object_radius: float = NOMINAL_OBJECT_SIZE / 2.0  # occluder radius in world units


def render_depth(ep: Episode, model: DepthModel | None = None) -> np.ndarray:
    """Finite (V, GRID, GRID) depth maps: z-buffered instance discs over a ground plane."""
    model = model or DepthModel()
    rng = np.random.default_rng(ep.seed + 7919)
    s = GRID / IMAGE_SIZE
    out = np.empty((ep.n_views, GRID, GRID), dtype=np.float32)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    pix = np.stack([(xx + 0.5) / s, (yy + 0.5) / s, np.ones_like(xx, dtype=float)], axis=-1)
    for v in range(ep.n_views):
        ray = pix @ np.linalg.inv(ep.K[v]).T  # camera-frame ray with z = 1
        world = ray @ ep.R[v]  # R^T applied to each ray
        centre = -ep.R[v].T @ ep.t[v]
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = -centre[2] / world[..., 2]
        depth = np.where((world[..., 2] < 0) & (hit > 0), hit, FAR_DEPTH)
        depth = np.clip(depth, 0.1, FAR_DEPTH)
        cam = ep.points3d @ ep.R[v].T + ep.t[v]
        for p in cam[cam[:, 2] > 0.1]:
            z = float(p[2])
            u, w = (ep.K[v] @ p)[:2] / z * s
            r = max(float(ep.K[v][0, 0]) * model.object_radius / z * s, 0.5)
            x0, x1 = max(int(np.floor(u - r)), 0), min(int(np.ceil(u + r)) + 1, GRID)
            y0, y1 = max(int(np.floor(w - r)), 0), min(int(np.ceil(w + r)) + 1, GRID)
            if x0 >= x1 or y0 >= y1:
                continue
            sub = ((xx[y0:y1, x0:x1] + 0.5 - u) ** 2 + (yy[y0:y1, x0:x1] + 0.5 - w) ** 2) <= r * r
            view = depth[y0:y1, x0:x1]
            view[sub] = np.minimum(view[sub], z)
        out[v] = depth * (1.0 + rng.normal(0.0, model.noise, depth.shape))
    return np.maximum(out, 0.05)


def episode_inputs_for(
    ep: Episode, finite_depth: bool, model: DepthModel | None = None
) -> tuple[list[LiftedDetection], Recon]:
    lifted, recon = episode_to_lifted(ep)
    if finite_depth:
        recon = Recon(
            backbone=recon.backbone,
            intrinsics=recon.intrinsics,
            extrinsics=recon.extrinsics,
            depth=render_depth(ep, model),
            depth_conf=recon.depth_conf,
            world_points=recon.world_points,
            world_points_conf=recon.world_points_conf,
            scene_scale=recon.scene_scale,
        )
    return lifted, recon


def score_episode(ep: Episode, lifted: Sequence[LiftedDetection], res: Any) -> dict[str, float]:
    cluster: dict[str, int] = {}
    for k, inst in enumerate(res.instances):
        for m in inst.members:
            cluster[m] = k
    labels = [cluster.get(ld.det.id, -1 - i) for i, ld in enumerate(lifted)]
    gt: list[int | None] = [int(g) if g >= 0 else None for g in ep.identity]
    universe = sorted({g for g in gt if g is not None})
    # Only detections inside a predicted instance count towards the predicted count: one the
    # associator rejected is not an instance (and an object with only rejected ones is missed).
    kept = [i for i, ld in enumerate(lifted) if ld.det.id in cluster]
    kept_labels = [labels[i] for i in kept]
    kept_gt = [gt[i] for i in kept]
    dec = count_decomposition(kept_labels, kept_gt, universe)
    pred, truth = float(len(res.instances)), float(ep.true_count())
    return {
        "pred": pred,
        "gt": truth,
        "abs_err": abs(pred - truth),
        "signed_err": pred - truth,
        "assoc_f1": pairwise_prf(labels, gt)[2],
        "duplicate_rate": duplicate_rate(kept_labels, kept_gt),
        "miss_rate": miss_rate(kept_labels, kept_gt, universe),
        "injected_fp_dets": float((ep.identity < 0).sum()),
        **{k: float(v) for k, v in dec.items()},
    }


@dataclass
class Prepared:
    """An episode with its lifted detections and recon built once for a depth variant."""

    ep: Episode
    lifted: list[LiftedDetection]
    recon: Recon


def prepare(
    files: Sequence[Path], finite_depth: bool, model: DepthModel | None = None
) -> list[Prepared]:
    out = []
    for p in files:
        ep = load_episode(p)
        lifted, recon = episode_inputs_for(ep, finite_depth, model)
        out.append(Prepared(ep, lifted, recon))
    return out


def evaluate(assoc: Associator, data: Sequence[Prepared]) -> list[dict[str, float]]:
    return [score_episode(d.ep, d.lifted, assoc.associate(d.lifted, d.recon, None)) for d in data]


def mean_abs_error(rows: Sequence[dict[str, float]]) -> float:
    return float(np.mean([r["abs_err"] for r in rows]))


@dataclass
class TuneResult:
    method: str
    best: dict[str, Any]
    val_mae: float
    default_val_mae: float
    trials: list[tuple[dict[str, Any], float]] = field(default_factory=list)


def _grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*space.values())]


def _gate(inner: Associator, p: dict[str, Any]) -> Associator:
    thr = float(p.get("score_threshold", 0.0))
    return ScoreGated(inner, thr) if thr > 0 else inner


def make_v1(p: dict[str, Any]) -> Associator:
    q = {k: v for k, v in p.items() if k != "score_threshold"}
    return _gate(V1Associator(**q), p)


def make_v2(p: dict[str, Any]) -> Associator:
    q = {k: v for k, v in p.items() if k != "score_threshold"}
    return _gate(V2Associator(V2Config(**q)), p)


def make_sync(p: dict[str, Any], v2: dict[str, Any] | None = None) -> Associator:
    """Sync over v2 pairwise matches; the score gate is sync's own (``p``), not v2's."""
    q = {k: v for k, v in p.items() if k != "score_threshold"}
    v2q = {k: v for k, v in (v2 or {}).items() if k != "score_threshold"}
    return _gate(SyncAssociator(SyncConfig(**q), V2Config(**v2q)), p)


def write_tuning_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Every grid point tried on the validation split, with the default and chosen flags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = ["variant", "method", "val_mae", "is_default", "chosen"]
    params = sorted({k for r in rows for k in r if k not in keys})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*keys, *params], restval="")
        w.writeheader()
        w.writerows(rows)


def tune(
    method: str,
    build: Callable[[dict[str, Any]], Associator],
    space: dict[str, list[Any]],
    default: dict[str, Any],
    val: Sequence[Prepared],
) -> TuneResult:
    """Grid search minimising mean |count error| on ``val``; ties go to the earlier grid point."""
    trials = [(p, mean_abs_error(evaluate(build(p), val))) for p in _grid(space)]
    best, best_mae = min(trials, key=lambda t: t[1])
    default_mae = mean_abs_error(evaluate(build(default), val))
    return TuneResult(method, best, best_mae, default_mae, trials)


COUNT_COLUMNS = ("mae", "signed_bias", "within_10pct", "zero_scene_false_count")
EXPLANATORY_COLUMNS = (
    "missed",
    "duplicates",
    "false_positives",
    "wrong_merges",
    "injected_fp_dets",
    "assoc_f1",
    "duplicate_rate",
    "miss_rate",
)
SUMMARY_COLUMNS = ("variant", "method", "n", "n_zero", *COUNT_COLUMNS, *EXPLANATORY_COLUMNS)


def summarize_counts(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Count-first summary of per-episode rows. ``zero_scene_false_count`` is NaN (with
    ``n_zero == 0``) when no episode has a true count of zero."""
    counts = count_summary([r["pred"] for r in rows], [r["gt"] for r in rows])
    out = {"n": float(len(rows)), "n_zero": counts["n_zero"]}
    out.update({k: counts[k] for k in COUNT_COLUMNS})
    out.update({k: float(np.mean([r[k] for r in rows])) for k in EXPLANATORY_COLUMNS})
    return out


def write_summary_csv(path: Path, summaries: dict[tuple[str, str], dict[str, float]]) -> None:
    """One row per (depth variant, method), count metrics before the explanatory ones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for (variant, method), s in summaries.items():
            w.writerow({"variant": variant, "method": method, **s})
