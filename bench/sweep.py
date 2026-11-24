"""Sweep definition for the benchmark (PLAN §9.5-9.6): config, job grid and a default executor.

``build_jobs`` is pure and deterministic, so the grid can be inspected with ``--dry-run`` before
any GPU time is spent. Only the default executor touches the pipeline.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bench.baselines import BASELINES
from bench.runner import DEFAULT_CSV, Job, RunOutput, SkipRun, run_jobs
from countonce.detect.base import Detector, filter_by_score
from countonce.detect.cache import CachedDetector, DetectionCache
from countonce.ingest.frames import list_images, sample_frames
from countonce.pipeline import make_frames
from countonce.recon.base import Reconstructor

TIERS = ("easy", "medium", "hard")


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["baseline", "countonce"]
    baseline: Literal["sum", "max", "median"] | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class Ablation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    values: list[Any]


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes_dir: Path = Path("data/scenes")
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
    order: list[Literal["ordered", "shuffled"]] = ["ordered", "shuffled"]
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


def select_views(
    paths: list[Path], n_views: int, sample_seed: int | None, base_seed: int, ordered: bool
) -> list[Path]:
    """The frames for one run. Selection is identical for ordered and shuffled runs (same
    frames, only the presentation order differs), and for every method (paired comparisons)."""
    seed = None if sample_seed is None else base_seed * 1000 + sample_seed
    chosen = sample_frames(paths, n_views, seed)
    if not ordered:
        rng = random.Random(f"{base_seed}:{n_views}:{sample_seed}")
        rng.shuffle(chosen)
    return chosen


def build_jobs(cfg: BenchConfig, scenes: list[dict[str, Any]]) -> list[Job]:
    """Cartesian grid scene × method × views × sample × order. ``scenes`` items need ``id``
    and may carry ``tier``. Baselines run once per view set; they are order-invariant for
    sum/max/median, so shuffled baseline rows are still emitted to keep the figures aligned."""
    jobs: list[Job] = []
    for scene in scenes:
        for method in cfg.all_methods():
            for n in cfg.views:
                for label, _ in view_samples(cfg):
                    for order in cfg.order:
                        jobs.append(
                            Job(
                                scene_id=scene["id"],
                                method=method.name,
                                n_views=n,
                                sample=label,
                                ordered=order == "ordered",
                                split=cfg.split,
                                config={
                                    "kind": method.kind,
                                    "baseline": method.baseline,
                                    "options": method.config,
                                    "tier": scene.get("tier", ""),
                                    "seed": cfg.seed,
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
        scenes.append({"id": d.name, "dir": d, "tier": gt.get("tier", ""), "gt": gt})
    return scenes


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

    def fn(job: Job) -> RunOutput:
        opts = job.config["options"]
        if job.config["kind"] == "countonce" and opts.get("assoc") == "v3":
            ckpt = opts.get("v3_checkpoint")
            if ckpt is None or not Path(ckpt).is_file():
                raise SkipRun("skipped_no_checkpoint", f"v3 checkpoint not found: {ckpt}")
        scene = by_id[job.scene_id]
        gt = scene["gt"]
        prompt = gt.get(cfg.prompt_key, cfg.prompt)
        paths = select_views(
            list_images(scene["dir"]), job.n_views or 0, seeds[job.sample], cfg.seed, job.ordered
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
    return run_jobs(jobs, fn, cfg.csv, cfg.model_dump(mode="json"))
