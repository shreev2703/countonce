"""Sweep definition for the benchmark (PLAN §9.5-9.6): config, job grid and a default executor.

``build_jobs`` is pure and deterministic, so the grid can be inspected with ``--dry-run`` before
any GPU time is spent. Only the default executor touches the pipeline.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bench.baselines import BASELINES
from bench.runner import DEFAULT_CSV, Job, RunOutput, SkipRun, run_jobs
from bench.slices import centers_spread
from countonce.detect.base import Detector, filter_by_score
from countonce.detect.cache import CachedDetector, DetectionCache
from countonce.ingest.frames import list_images, sample_frames
from countonce.pipeline import make_frames
from countonce.recon.base import Reconstructor

TIERS = ("easy", "medium", "hard")


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["baseline", "countonce", "vlm", "zero_suite"]
    baseline: Literal["sum", "max", "median"] | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _vlm_needs_provider(self) -> MethodSpec:
        if self.kind == "vlm" and not {"provider", "model"} <= set(self.config):
            raise ValueError(f"vlm method {self.name!r} needs config.provider and config.model")
        return self


class Ablation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    values: list[Any]


class DatasetSpec(BaseModel):
    """Opt-in external dataset (``bench.datasets``): materialised as scene dirs, then swept."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["rgbd_videocount", "videocount", "onescene"]
    root: Path
    enabled: bool = False  # opt-in: nothing is loaded unless set true
    max_frames: int | None = None  # None = loader default (20 for videos, all for onescene)
    stride: int | None = None


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes_dir: Path = Path("data/scenes")
    datasets: list[DatasetSpec] = Field(default_factory=list)
    csv: Path = DEFAULT_CSV
    runs_dir: Path = Path("results/runs")
    cache_dir: Path | None = Path("results/cache")  # shared detection/recon cache; None = off
    prompt_key: str = "prompt"
    prompt: str = "object"
    split: str = "test"
    views: list[int] = [3, 5, 10, 20, 40]
    random_samples: int = Field(3, ge=0)
    even_sample: bool = True
    seed: int = 0
    # Paid VLM baselines (kind: vlm) run only with allow_paid AND a positive cap AND an API key.
    allow_paid: bool = False
    spend_cap_usd: float | None = None
    order: list[Literal["ordered", "shuffled"]] = ["ordered", "shuffled"]
    # Viewpoint spread (9.5): narrow/wide = bottom/top tercile of angular extent; all = no filter.
    spread: list[Literal["narrow", "wide", "all"]] = ["all"]
    methods: list[MethodSpec]
    ablation_base: str | None = None
    ablations: dict[str, Ablation] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> BenchConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls(**raw)

    def all_methods(self) -> list[MethodSpec]:
        """Declared methods plus one-factor ablations of ``ablation_base``."""
        out = list(self.methods)
        if not self.ablations:
            return out
        base = next((m for m in self.methods if m.name == self.ablation_base), None)
        if base is None or base.kind != "countonce":
            raise ValueError("ablation_base must name a countonce method in `methods`")
        seen = {m.name for m in out}
        for abl_name, abl in self.ablations.items():
            for v in abl.values:
                if base.config.get(abl.field) == v:
                    continue  # identical to the base method
                name = f"{base.name}+{abl_name}={v}"
                if name not in seen:
                    seen.add(name)
                    out.append(
                        MethodSpec(
                            name=name, kind="countonce", config={**base.config, abl.field: v}
                        )
                    )
        return out


def view_samples(cfg: BenchConfig) -> list[tuple[str, int | None]]:
    """Sample labels with their sampling seed: random0..randomN-1 then ``even`` (seed None)."""
    s: list[tuple[str, int | None]] = [(f"random{i}", i) for i in range(cfg.random_samples)]
    if cfg.even_sample:
        s.append(("even", None))
    return s


def scene_centers(scene: dict[str, Any], n_images: int) -> np.ndarray | None:
    """GT camera centres (N,3) of a scene, or None when absent or not one per image."""
    c = scene.get("camera_centers")
    if c is None:
        return None
    a = np.asarray(c, dtype=np.float64)
    return a if a.shape == (n_images, 3) else None


def scene_scale(scene: dict[str, Any], centers: np.ndarray) -> float:
    """``scene_scale`` from gt.json, else the RMS spread of all the scene's camera centres."""
    s = scene.get("scene_scale")
    return float(s) if s else centers_spread(centers, 1.0)


def select_views(
    paths: list[Path],
    n_views: int,
    sample_seed: int | None,
    base_seed: int,
    ordered: bool,
    spread: str = "all",
    centers: np.ndarray | None = None,
) -> list[Path]:
    """The frames for one run. Selection is identical for ordered and shuffled runs (same
    frames, only the presentation order differs), and for every method (paired comparisons)."""
    if spread != "all" and centers is not None:
        chosen = select_spread_views(paths, centers, n_views, spread, sample_seed, base_seed)
    else:
        seed = None if sample_seed is None else base_seed * 1000 + sample_seed
        chosen = sample_frames(paths, n_views, seed)
    if not ordered:
        rng = random.Random(f"{base_seed}:{n_views}:{sample_seed}")
        rng.shuffle(chosen)
    return chosen


SPREAD_POOL = 64  # candidate subsets drawn per (scene, n_views) for tercile selection


def angular_order(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Views in angular order around the centroid, in the best-fit plane of the centres.

    Returns (order, angle): ``angle[k]`` is the monotone angle in radians of ``order[k]`` from
    the start of the arc. The arc starts after the largest angular gap, so a partial arc is not
    split at the +-pi seam."""
    c = np.asarray(centers, dtype=np.float64)
    c = c - c.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ang = np.arctan2(c @ vt[1], c @ vt[0])
    idx = np.argsort(ang, kind="stable")
    sa = ang[idx]
    gaps = np.diff(np.concatenate([sa, [sa[0] + 2 * np.pi]]))
    start = (int(np.argmax(gaps)) + 1) % len(idx)
    order = np.roll(idx, -start)
    rolled = np.roll(sa, -start)
    unwrapped = np.unwrap(rolled)
    return order, unwrapped - unwrapped[0]


def select_spread_views(
    paths: list[Path],
    centers: np.ndarray,
    n_views: int,
    spread: str,
    sample_seed: int | None,
    base_seed: int,
) -> list[Path]:
    """Frames whose spread around the centroid is in the bottom (``narrow``) or top (``wide``)
    tercile of a seeded pool of candidate sub-arcs of the angular order; ``all`` is the plain
    ``sample_frames`` rule.

    The pool is one evenly spaced full-arc set plus random sub-arcs of the angular order. A random
    sample (int ``sample_seed``) draws uniformly from the tercile; the even sample (``None``)
    takes the tercile member with the most even angular spacing. Returned in image order."""
    if spread == "all" or n_views <= 0 or n_views >= len(paths):
        seed = None if sample_seed is None else base_seed * 1000 + sample_seed
        return sample_frames(paths, n_views, seed)
    if len(centers) != len(paths):
        raise ValueError(f"{len(centers)} camera centres for {len(paths)} images")
    order, angle = angular_order(centers)
    m = len(order)
    pool_rng = random.Random(f"spread-pool:{base_seed}:{n_views}")  # same pool for every sample
    pool: list[np.ndarray] = [np.linspace(0, m - 1, n_views).round().astype(int)]
    for _ in range(SPREAD_POOL):
        w = pool_rng.randint(n_views, m)
        start = pool_rng.randint(0, m - w)
        pool.append(np.array(sorted(pool_rng.sample(range(start, start + w), n_views))))
    # Terciles are of the measured spread (the CSV column), so the label and the column agree;
    # the angular order only shapes the candidate sub-arcs.
    extent = np.array([centers_spread(centers[order[p]]) for p in pool])
    lo, hi = np.quantile(extent, [1 / 3, 2 / 3])
    keep = np.flatnonzero(extent <= lo if spread == "narrow" else extent >= hi)
    if sample_seed is None:
        gap_std = [float(np.std(np.diff(angle[pool[k]]))) for k in keep]
        pick = int(keep[int(np.argmin(gap_std))])
    else:
        pick = int(random.Random(f"spread:{base_seed}:{n_views}:{sample_seed}").choice(list(keep)))
    return [paths[i] for i in sorted(int(order[k]) for k in pool[pick])]


def build_jobs(cfg: BenchConfig, scenes: list[dict[str, Any]]) -> list[Job]:
    """Cartesian grid scene × method × views × sample × order. ``scenes`` items need ``id``
    and may carry ``tier``. Baselines run once per view set; they are order-invariant for
    sum/max/median, so shuffled baseline rows are still emitted to keep the figures aligned."""
    jobs: list[Job] = []
    for scene in scenes:
        for method in cfg.all_methods():
            if method.kind == "zero_suite":
                continue  # run by bench.zero_suite, not part of the view grid
            for n in cfg.views:
                for label, _ in view_samples(cfg):
                    for spread, order in itertools.product(cfg.spread, cfg.order):
                        jobs.append(
                            Job(
                                scene_id=scene["id"],
                                method=method.name,
                                n_views=n,
                                sample=label,
                                ordered=order == "ordered",
                                split=cfg.split,
                                spread=spread,
                                config={
                                    "kind": method.kind,
                                    "baseline": method.baseline,
                                    "options": method.config,
                                    "tier": scene.get("tier", ""),
                                    "seed": cfg.seed,
                                    # Only when the scene declares it, so older job keys hold.
                                    **({"overlap": scene["overlap"]} if "overlap" in scene else {}),
                                },
                            )
                        )
    return jobs


def discover_scenes(scenes_dir: Path) -> list[dict[str, Any]]:
    """Scene folders with ``images/`` and ``gt.json``; sorted for a deterministic grid."""
    scenes: list[dict[str, Any]] = []
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"scenes_dir not found: {scenes_dir}")
    for d in sorted(p for p in scenes_dir.iterdir() if p.is_dir()):
        gt_path = d / "gt.json"
        if not (d / "images").is_dir() or not gt_path.exists():
            continue
        gt = json.loads(gt_path.read_text())
        scene = {"id": d.name, "dir": d, "tier": gt.get("tier", ""), "gt": gt}
        if "camera_centers" in gt:  # GT camera centres, in sorted image order (spread axis)
            scene["camera_centers"] = gt["camera_centers"]
            if "scene_scale" in gt:
                scene["scene_scale"] = gt["scene_scale"]
        if "overlap" in gt:
            scene["overlap"] = gt["overlap"]  # view-overlap label or fraction, for figure 10
        scenes.append(scene)
    return scenes


def dataset_scenes(cfg: BenchConfig) -> list[dict[str, Any]]:
    """Scenes from the enabled ``datasets`` entries, in ``discover_scenes`` format. Frames are
    symlinked under ``<csv dir>/datasets/<name>/`` and only the config's ``split`` is kept."""
    from bench.datasets import load_dataset
    from bench.datasets.common import to_sweep_scenes

    out: list[dict[str, Any]] = []
    for d in cfg.datasets:
        if not d.enabled:
            continue
        kw: dict[str, Any] = {"stride": d.stride}
        if d.max_frames is not None:
            kw["max_frames"] = d.max_frames
        if d.name == "videocount":
            kw["frames_dir"] = cfg.csv.parent / "datasets" / d.name / "_frames"
        loaded = load_dataset(d.name, d.root, **kw)
        out += to_sweep_scenes(loaded, cfg.csv.parent / "datasets" / d.name, split=cfg.split)
    return out


def baseline_counts(paths: list[Path], prompt: str, conf: Any, detector: Detector) -> list[int]:
    """Per-image detection counts: the detector runs once over the selected views (through the
    detection cache when ``conf.cache_dir`` is set), then ``conf.score_threshold`` is applied.
    Views with no detections count as 0, so the list has one entry per selected view."""
    frames = make_frames(paths)
    det: Detector = detector
    if conf.cache_dir is not None:
        det = CachedDetector(detector, DetectionCache(Path(conf.cache_dir) / "detect"))
    kept = filter_by_score(det.detect(frames, prompt), conf.score_threshold)
    per_frame = Counter(d.frame_id for d in kept)
    return [per_frame.get(f.id, 0) for f in frames]


def make_executor(
    cfg: BenchConfig,
    scenes: list[dict[str, Any]],
    detector_factory: Callable[[Any], Detector] | None = None,
    reconstructor_factory: Callable[[Any], Reconstructor] | None = None,
) -> Callable[[Job], RunOutput]:
    """Default executor. Baselines (sum/max/median) run the config-selected detector on the
    selected views and reduce the per-image counts; CountOnce methods, including
    ``co_v1_colmap`` (COLMAP lifting, baseline 4), go through ``run()``. Both paths build their
    backends from the same ``countonce.Config`` (method options + the bench cache dir).
    The factories (default: ``build_detector``) let tests inject fakes."""
    from countonce.config import Config
    from countonce.pipeline import build_detector, run

    by_id = {s["id"]: s for s in scenes}
    seeds = dict(view_samples(cfg))
    make_det = detector_factory or build_detector
    vlm_fn: Callable[[Job], RunOutput] | None = None

    def fn(job: Job) -> RunOutput:
        nonlocal vlm_fn
        if job.config["kind"] == "vlm":
            if vlm_fn is None:
                from bench.vlm.harness import make_vlm_executor

                vlm_fn = make_vlm_executor(cfg, scenes)
            return vlm_fn(job)
        opts = job.config["options"]
        if job.config["kind"] == "countonce" and opts.get("assoc") == "v3":
            ckpt = opts.get("v3_checkpoint")
            if ckpt is None or not Path(ckpt).is_file():
                raise SkipRun("skipped_no_checkpoint", f"v3 checkpoint not found: {ckpt}")
        scene = by_id[job.scene_id]
        gt = scene["gt"]
        prompt = gt.get(cfg.prompt_key, cfg.prompt)
        all_paths = list_images(scene["dir"])
        centers = scene_centers(scene, len(all_paths))
        if job.spread != "all" and centers is None:
            raise SkipRun("skipped_no_cameras", "scene gt.json has no usable camera_centers")
        paths = select_views(
            all_paths, job.n_views or 0, seeds[job.sample], cfg.seed, job.ordered,
            spread=job.spread, centers=centers,
        )  # fmt: skip
        spread_val = None
        if centers is not None:
            pos = {p: i for i, p in enumerate(all_paths)}
            spread_val = centers_spread(
                centers[[pos[p] for p in paths]], scene_scale(scene, centers)
            )
        t0 = time.perf_counter()
        options = {"cache_dir": cfg.cache_dir, **job.config["options"]}
        conf = Config(**options).with_overrides(out_dir=cfg.runs_dir / job.key())
        if job.config["kind"] == "baseline":
            reduce = BASELINES[job.config["baseline"]]
            counts = baseline_counts(paths, prompt, conf, make_det(conf))
            return RunOutput(
                pred=reduce(counts),
                gt=float(gt["count"]),
                runtime_s=time.perf_counter() - t0,
                baseline_spread=spread_val,
            )
        res = run(
            paths,
            prompt,
            conf,
            detector=make_det(conf) if detector_factory else None,
            reconstructor=reconstructor_factory(conf) if reconstructor_factory else None,
        )
        return RunOutput(
            pred=float(res.count),
            gt=float(gt["count"]),
            low=float(res.interval.low),
            high=float(res.interval.high),
            runtime_s=time.perf_counter() - t0,
            peak_vram_mb=float(res.peak_vram_mb),
            timings={k: float(v) for k, v in res.timings.items()},
            run_dir=str(conf.out_dir),
            baseline_spread=spread_val,
        )

    return fn


def run_bench(
    cfg: BenchConfig,
    scenes: list[dict[str, Any]],
    fn: Callable[[Job], RunOutput] | None = None,
    *,
    detector_factory: Callable[[Any], Detector] | None = None,
    reconstructor_factory: Callable[[Any], Reconstructor] | None = None,
) -> tuple[int, int, int]:
    jobs = build_jobs(cfg, scenes)
    fn = fn or make_executor(cfg, scenes, detector_factory, reconstructor_factory)
    ran, skipped, errored = run_jobs(jobs, fn, cfg.csv, cfg.model_dump(mode="json"))
    zero = [m for m in cfg.methods if m.kind == "zero_suite"]
    if zero:
        from bench.zero_suite import run_zero_suite, write_zero_summary

        for m in zero:
            z = dict(m.config)
            r = run_zero_suite(
                cfg.csv,
                m.name,
                z.pop("options", {}),
                Path(z.pop("scenes_dir", cfg.csv.parent / "zero_suite" / "scenes")),
                scenes,
                n_synthetic=int(z.pop("n_synthetic", 12)),
                seed=int(z.pop("seed", cfg.seed)),
                split=cfg.split,
                runs_dir=cfg.runs_dir,
                cache_dir=cfg.cache_dir,
                n_views=int(z.pop("n_views", 5)),
                detector_factory=detector_factory,
                reconstructor_factory=reconstructor_factory,
            )
            ran, skipped, errored = ran + r[0], skipped + r[1], errored + r[2]
        write_zero_summary(cfg.csv, cfg.csv.parent / "zero_suite.md")
    return ran, skipped, errored
