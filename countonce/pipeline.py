"""End-to-end pipeline: ingest -> detect -> recon -> lift -> associate -> export (PLAN 8.11)."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from countonce.confidence import k_grid, sensitivity_range, threshold_grid
from countonce.config import Config
from countonce.conformal import CalibrationArtifact, interval_from_artifact, resolve_method
from countonce.detect.base import Detector, filter_by_score
from countonce.detect.cache import CachedDetector, DetectionCache
from countonce.export import export_all
from countonce.ingest.frames import IMAGE_EXTS, drop_blurry, extract_frames
from countonce.ingest.select import cap_frames, drop_near_duplicates
from countonce.lift import LiftKwargs, compute_scene_scale, lift_detections
from countonce.merge.base import Associator
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator, V2Config
from countonce.merge.sync import SyncAssociator
from countonce.provenance import run_metadata
from countonce.recon.base import Reconstructor
from countonce.recon.cache import CachedReconstructor, ReconCache
from countonce.recon.chunked import ChunkedReconstructor
from countonce.types import CountResult, Frame, Interval, ModelVersions
from countonce.types import Recon as DenseRecon
from countonce.warnings import assess_quality

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def collect_paths(inputs: list[Path] | Path, out_dir: Path, fps: float) -> list[Path]:
    """Expand a directory, video file or list of images into an ordered image list."""
    items = [Path(inputs)] if isinstance(inputs, (str, Path)) else [Path(p) for p in inputs]
    paths: list[Path] = []
    for item in items:
        if item.is_dir():
            paths += sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        elif item.suffix.lower() in VIDEO_EXTS:
            paths += extract_frames(item, out_dir / "frames", fps)
        elif item.suffix.lower() in IMAGE_EXTS and item.exists():
            paths.append(item)
        else:
            raise FileNotFoundError(f"unsupported or missing input: {item}")
    if not paths:
        raise ValueError("no input images found")
    return paths


def make_frames(paths: list[Path], preprocess_mode: str | None = None) -> list[Frame]:
    """Frames at native size. With a backbone ``preprocess_mode`` ("crop"/"pad") each frame also
    gets the exact `GridMap` onto that backbone's input grid (PLAN 8.1)."""
    from PIL import Image

    from countonce.ingest.frames import backbone_grid_map

    frames = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            w, h = im.size
        gm = backbone_grid_map(w, h, preprocess_mode) if preprocess_mode else None
        frames.append(Frame(id=i, path=p, width=w, height=h, sha256=_sha256(p), grid_map=gm))
    return frames


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_detector(config: Config) -> Detector:
    if config.detector == "gdino":
        from countonce.detect.groundingdino import GroundingDinoDetector

        return GroundingDinoDetector(**config.detector_options)
    if config.detector == "countgd":
        from countonce.detect.countgd import CountGDDetector

        return CountGDDetector(**config.detector_options)
    raise NotImplementedError(f"detector {config.detector!r} is not implemented yet")


def build_reconstructor(config: Config) -> Reconstructor:
    if config.backbone in ("vggt", "colmap"):
        from countonce.recon.adapter import load_method

        return load_method(config.backbone, **config.backbone_options)
    raise NotImplementedError(f"backbone {config.backbone!r} is not implemented yet")


def _chunk_cap(config: Config) -> int | None:
    """Frames per chunk: explicit ``chunk_size``, else the memory-guard calibration, else None."""
    if config.chunk_size > 0:
        return config.chunk_size
    from countonce.recon.profiling import max_frames_from_profile

    dtype = config.backbone_options.get("weights_dtype", "bf16")
    return max_frames_from_profile(dtype)


def _make_associator(config: Config, k: float) -> Associator:
    """Associator at strictness ``k``: v1 uses it directly, v2 scales its reprojection gate."""
    if config.assoc == "v1":
        return V1Associator(k=k)
    base = V2Config()
    gate = base.max_reproj_norm * k / config.merge_k
    v2cfg = V2Config(**{**base.__dict__, "max_reproj_norm": gate})
    if config.assoc == "sync":
        return SyncAssociator(v2_config=v2cfg)
    return V2Associator(v2cfg)


def lift_kwargs(config: Config, recon: DenseRecon) -> LiftKwargs:
    """Lift thresholds from the config; sparse (COLMAP) input gets a lower pixel floor."""
    sparse = recon.backbone == "colmap"
    kw: LiftKwargs = {}
    min_px = config.lift_min_pixels
    if min_px is None and sparse:
        min_px = 3
    pct = config.lift_conf_percentile
    if pct is None and sparse:
        pct = 0.0
    if min_px is not None:
        kw["min_pixels"] = min_px
    if pct is not None:
        kw["conf_percentile"] = pct
    return kw


def run(
    inputs: list[Path] | Path,
    prompt: str,
    config: Config,
    *,
    detector: Detector | None = None,
    reconstructor: Reconstructor | None = None,
    export: bool = True,
) -> CountResult:
    """Count distinct instances of ``prompt`` in the images/video at ``inputs``.

    ``detector`` / ``reconstructor`` override the config-selected backends (tests, notebooks).
    """
    if config.assoc not in ("v1", "v2", "sync"):
        raise NotImplementedError(f"associator {config.assoc!r} is not implemented yet")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # Ingest
    out_dir = Path(config.out_dir)
    paths = collect_paths(inputs, out_dir, config.fps)
    if config.blur_threshold > 0:
        paths, _ = drop_blurry(paths, config.blur_threshold)
    if config.dedup_threshold > 0:
        paths, _ = drop_near_duplicates(paths, config.dedup_threshold)
    if config.max_frames > 0:
        paths = cap_frames(paths, config.max_frames)
    if not paths:
        raise ValueError("no frames left after filtering")
    rec = reconstructor if reconstructor is not None else build_reconstructor(config)
    frames = make_frames(paths, getattr(rec, "preprocess_mode", None))
    timings["ingest"] = time.perf_counter() - t0

    # Detect
    t = time.perf_counter()
    det = detector if detector is not None else build_detector(config)
    if config.cache_dir is not None:
        det = CachedDetector(det, DetectionCache(Path(config.cache_dir) / "detect"))
    raw_detections = det.detect(frames, prompt)
    detections = filter_by_score(raw_detections, config.score_threshold)
    timings["detect"] = time.perf_counter() - t

    # Reconstruct
    t = time.perf_counter()
    if config.chunked_recon:
        cap = _chunk_cap(config)
        if cap is not None:
            rec = ChunkedReconstructor(rec, cap, config.chunk_overlap)
    cache_key: str | None = None
    cache_hit = False
    recon: DenseRecon
    if config.cache_dir is not None:
        cached = CachedReconstructor(rec, ReconCache(Path(config.cache_dir) / "recon"), "0")
        got = cached.reconstruct_cached(frames)
        recon, cache_key, cache_hit = got.recon, got.key, got.hit
    else:
        recon = rec.reconstruct(frames)
    timings["recon"] = time.perf_counter() - t

    # Lift + associate
    t = time.perf_counter()
    lift_kw = lift_kwargs(config, recon)
    lifted = lift_detections(detections, recon, {f.id: f for f in frames}, **lift_kw)
    frame_map = {f.id: f for f in frames}
    clustered = _make_associator(config, config.merge_k).associate(lifted, recon, frame_map)
    timings["associate"] = time.perf_counter() - t

    # Sensitivity range: re-count over a grid of merge strictness and detector threshold.
    t = time.perf_counter()
    lift_by_thr = {config.score_threshold: lifted}

    def count_at(k: float, thr: float) -> int:
        if thr not in lift_by_thr:
            kept = filter_by_score(raw_detections, thr)
            lift_by_thr[thr] = lift_detections(kept, recon, frame_map, **lift_kw)
        assoc = _make_associator(config, k)
        return len(assoc.associate(lift_by_thr[thr], recon, frame_map).instances)

    sens = sensitivity_range(
        count_at,
        k_grid(config.merge_k),
        threshold_grid(config.score_threshold),
        config.merge_k,
        config.score_threshold,
    )
    interval = sens.to_interval()
    interval_warnings: list[dict[str, str]] = []
    if config.interval_method != "sensitivity":
        art = _load_calibration(config)
        res = resolve_method(config.interval_method, art)
        if res.warning:
            interval_warnings.append(res.warning)
        if res.method != "sensitivity" and art is not None:
            lo, hi = interval_from_artifact(
                res.method,
                art,
                len(clustered.instances),
                sens.low,
                sens.high,
                config.interval_level,
                mu_at=lambda theta: _mu_at(art, theta, config, count_at),
            )
            interval = Interval(lo, hi, config.interval_level, res.method, art.n_calibration)
    timings["confidence"] = time.perf_counter() - t

    per_frame: dict[int, int] = {f.id: 0 for f in frames}
    for ld in lifted:
        if ld.ok:
            per_frame[ld.det.frame_id] += 1

    count = len(clustered.instances)
    quality = assess_quality(
        n_frames=len(frames),
        count=count,
        instances=clustered.instances,
        lifted=lifted,
        recon=recon,
        interval_low=sens.low,
        interval_high=sens.high,
    )
    result = CountResult(
        prompt=prompt,
        count=count,
        interval=interval,
        confidence_method="sensitivity",
        warnings=quality + interval_warnings,
        model_versions=ModelVersions(
            detector=f"{det.name}@{det.version}", backbone=recon.backbone, associator=config.assoc
        ),
        input_hash=_input_hash(frames),
        recon_cache_key=cache_key,
        recon_cache_hit=cache_hit,
        zero_count=count == 0,
        instances=clustered.instances,
        per_frame_counts=per_frame,
        detection_status=clustered.detection_status,
        reject_reasons=clustered.reject_reasons,
        match_evidence=clustered.evidence,
        timings=timings,
        peak_vram_mb=recon.peak_vram_mb,
        config=run_metadata(config.to_dict()),
    )
    timings["total"] = time.perf_counter() - t0

    if export:
        export_all(
            result,
            out_dir,
            frames=frames,
            detections=detections,
            scene_scale=compute_scene_scale(recon),
        )
    return result


def _load_calibration(config: Config) -> CalibrationArtifact | None:
    if config.calibration_path is None:
        return None
    return CalibrationArtifact.load(config.calibration_path)


def _mu_at(
    art: CalibrationArtifact, theta: float, config: Config, count_at: Callable[[float, float], int]
) -> float:
    """Count under the Jackknife+ tuned parameter (``k``: factor of ``merge_k``; ``scale``:
    multiplier of the default count)."""
    if art.tuned == "k":
        return float(count_at(config.merge_k * theta, config.score_threshold))
    if art.tuned == "scale":
        return theta * count_at(config.merge_k, config.score_threshold)
    raise ValueError(f"unknown tuned parameter {art.tuned!r}")


def _input_hash(frames: list[Frame]) -> str:
    from countonce.recon.cache import input_hash

    return input_hash(frames)
