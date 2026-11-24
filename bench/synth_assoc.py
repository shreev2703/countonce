"""Synthetic association benchmark (PLAN §9.4 association metrics, §9.7 figure 4, §10.5).

Runs v1, v2, sync and v3 on held-out ``synth.geometry`` test episodes. The input is the episode's
lifted detections (unprojected from the noisy 2D detections and depths, so detector noise,
drops and false positives are all present) and the ground-truth identities are used only for
scoring. Rows go to a ``bench.csv``-compatible file with ``domain=synthetic``.

Episodes are rendered into a small stand-in reconstruction (a 64 px grid, NaN depth so the
occlusion test never fires), because no images exist for geometry-only episodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bench.metrics import duplicate_rate, miss_rate, pairwise_prf
from bench.runner import Job, RunOutput, SkipRun, run_jobs
from countonce.match.model import NOMINAL_OBJECT_SIZE, episode_inputs
from countonce.merge.base import Associator, ClusterResult
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator
from countonce.merge.sync import SyncAssociator
from countonce.types import Detection, LiftedDetection, Recon
from synth.geometry import IMAGE_SIZE, Episode, load_episode

GRID = 64  # side of the stand-in recon grid, in pixels
METHODS = ("v1", "v2", "sync", "v3")
DEFAULT_CSV = Path("results/bench_synth_assoc.csv")


@dataclass
class SynthAssocConfig:
    test_dir: str = "data/synth/test"
    csv: str = str(DEFAULT_CSV)
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    v3_checkpoint: str | None = None
    v3_synchronize: bool = False
    max_episodes: int | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> SynthAssocConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        bad = sorted(set(raw) - {f.name for f in fields(cls)})
        if bad:
            raise ValueError(f"unknown synth-assoc config keys: {bad}")
        cfg = cls(**raw)
        unknown = sorted(set(cfg.methods) - set(METHODS))
        if unknown:
            raise ValueError(f"unknown methods {unknown}; expected a subset of {METHODS}")
        return cfg


def episode_to_lifted(ep: Episode) -> tuple[list[LiftedDetection], Recon]:
    """Lifted detections and a stand-in ``Recon`` for one episode (frame id = view index).

    Positions are the unprojected ``uv`` + ``depth`` in world coordinates. The box half-extent
    is half of what ``episode_inputs`` uses so that the box width the v3 feature builder reads
    back equals the width ``episode_inputs`` gives the trainer.
    """
    s = GRID / IMAGE_SIZE
    scale_k = np.diag([s, s, 1.0])
    k = np.einsum("ij,vjk->vik", scale_k, ep.K)
    ext = np.concatenate([ep.R, ep.t[:, :, None]], axis=2)

    d = episode_inputs(ep)
    v = ep.view
    pix = np.concatenate([ep.uv, np.ones((ep.n_detections, 1))], axis=1)
    cam_xyz = np.einsum("nij,nj->ni", np.linalg.inv(ep.K[v]), pix) * ep.depth[:, None]
    xyz = np.einsum("nji,nj->ni", ep.R[v], cam_xyz - ep.t[v])

    half_px = ep.K[v, 0, 0] * NOMINAL_OBJECT_SIZE / np.maximum(ep.depth, 1e-6) * s
    lifted: list[LiftedDetection] = []
    counts: dict[int, int] = {}
    for i in range(ep.n_detections):
        f = int(v[i])
        j = counts.get(f, 0)
        counts[f] = j + 1
        c = ep.uv[i] * s
        h = float(half_px[i]) / 2.0
        det = Detection(
            id=Detection.make_id(f, j),
            frame_id=f,
            score=float(ep.score[i]),
            box_xyxy=np.array([c[0] - h, c[1] - h, c[0] + h, c[1] + h]),
        )
        lifted.append(LiftedDetection(det=det, xyz=xyz[i], size=NOMINAL_OBJECT_SIZE, lift_conf=1.0))

    n = ep.n_views
    recon = Recon(
        backbone="synthetic",
        intrinsics=k,
        extrinsics=ext,
        depth=np.broadcast_to(np.float32(np.nan), (n, GRID, GRID)),
        depth_conf=np.broadcast_to(np.float32(1.0), (n, GRID, GRID)),
        world_points=np.broadcast_to(np.float32(0.0), (n, GRID, GRID, 3)),
        world_points_conf=np.broadcast_to(np.float32(1.0), (n, GRID, GRID)),
        scene_scale=_scene_scale(ep, d),
    )
    return lifted, recon


def _scene_scale(ep: Episode, d: Any) -> float:
    """The scale ``episode_inputs`` divides by (median detection-to-camera distance)."""
    # d.size = NOMINAL_OBJECT_SIZE / scale
    return float(NOMINAL_OBJECT_SIZE / d.size[0]) if ep.n_detections else 1.0


def score_result(ep: Episode, lifted: Sequence[LiftedDetection], res: ClusterResult) -> RunOutput:
    """Association metrics of one associator output against the episode's GT identities."""
    cluster: dict[str, int] = {}
    for k, inst in enumerate(res.instances):
        for m in inst.members:
            cluster[m] = k
    labels = [cluster.get(ld.det.id, -1 - i) for i, ld in enumerate(lifted)]  # unassigned: alone
    gt: list[int | None] = [int(g) if g >= 0 else None for g in ep.identity]
    pred, truth = float(len(res.instances)), float(ep.true_count())
    return RunOutput(
        pred=pred,
        gt=truth,
        assoc_f1=pairwise_prf(labels, gt)[2],
        duplicate_rate=duplicate_rate(labels, gt),
        miss_rate=miss_rate(labels, gt, sorted({g for g in gt if g is not None})),
        count_error=abs(pred - truth),
        domain="synthetic",
    )


def evaluate_associator(assoc: Associator, ep: Episode) -> RunOutput:
    lifted, recon = episode_to_lifted(ep)
    return score_result(ep, lifted, assoc.associate(lifted, recon, None))


def make_associator(name: str, cfg: SynthAssocConfig) -> Associator:
    if name == "v1":
        return V1Associator()
    if name == "v2":
        return V2Associator()
    if name == "sync":
        return SyncAssociator()
    if name == "v3":
        from countonce.merge.v3 import V3Associator, require_checkpoint

        try:
            path = require_checkpoint(cfg.v3_checkpoint)
        except FileNotFoundError as exc:
            raise SkipRun("skipped_no_checkpoint", str(exc)) from exc
        return V3Associator(path, synchronize=cfg.v3_synchronize)
    raise ValueError(f"unknown method {name!r}")


def episode_files(cfg: SynthAssocConfig) -> list[Path]:
    files = sorted(Path(cfg.test_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz episodes in {cfg.test_dir} (run `countonce synth` first)")
    return files[: cfg.max_episodes]


def run_synth_assoc(cfg: SynthAssocConfig) -> tuple[int, int, int]:
    """Append one row per (test episode, method); resumable. Returns (ran, skipped, errored)."""
    files = episode_files(cfg)
    cache: dict[str, Associator] = {}

    def fn(job: Job) -> RunOutput:
        if job.method not in cache:
            cache[job.method] = make_associator(job.method, cfg)
        return evaluate_associator(cache[job.method], load_episode(job.config["episode"]))

    jobs: list[Job] = []
    for p in files:
        ep = load_episode(p)
        for m in cfg.methods:
            conf: dict[str, Any] = {"episode": str(p), "domain": "synthetic"}
            if m == "v3":
                conf.update(v3_checkpoint=cfg.v3_checkpoint, synchronize=cfg.v3_synchronize)
            jobs.append(
                Job(
                    scene_id=f"synth_{ep.seed:06d}",
                    method=m,
                    n_views=ep.n_views,
                    split="test",
                    config=conf,
                )
            )
    return run_jobs(jobs, fn, cfg.csv, run_config={"synth_assoc": cfg.test_dir})
