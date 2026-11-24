"""End-to-end pipeline: ingest -> detect -> recon -> lift -> associate -> export (PLAN 8.11)."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from countonce.confidence import k_grid, sensitivity_range, threshold_grid
from countonce.config import Config
from countonce.conformal import CalibrationArtifact, interval_from_artifact, resolve_method
from countonce.detect.base import Detector, filter_by_score
from countonce.detect.cache import CachedDetector, DetectionCache
from countonce.detect.prompts import (
    ParsedPrompt,
    detect_with_negatives,
    negatives_supported,
    parse_prompt,
)
from countonce.errors import ErrorCode, MediaError
from countonce.explain import build_camera_records, build_detection_records
from countonce.export import export_all
from countonce.ingest.frames import IMAGE_EXTS, drop_blurry, extract_frames
from countonce.ingest.privacy import (
    PrivacyRegionDetector,
    blur_image_file,
    build_privacy_detector,
)
from countonce.ingest.sanitize import SanitizeReport, sanitize_image, validate_video
from countonce.ingest.select import cap_frames, drop_near_duplicates
from countonce.lift import LiftKwargs, apply_lift_region, compute_scene_scale, lift_detections
from countonce.merge.base import Associator
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator, V2Config
from countonce.merge.sync import SyncAssociator
from countonce.oom import Attempt, RungUnavailable, free_cache, run_with_oom_ladder
from countonce.optim import check_sdpa, configure_inference, wrap_detector
from countonce.provenance import run_metadata
from countonce.recon.base import Reconstructor
from countonce.recon.cache import CachedReconstructor, ReconCache
from countonce.recon.chunked import ChunkedReconstructor
from countonce.types import CountResult, Detection, Frame, Interval, ModelVersions
from countonce.types import Recon as DenseRecon
from countonce.warnings import assess_quality, negatives_ignored_warning, zero_count_warning

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def collect_paths(
    inputs: list[Path] | Path,
    out_dir: Path,
    fps: float,
    report: SanitizeReport | None = None,
) -> list[Path]:
    """Expand a directory, video file or list of images into an ordered image list.

    Images are re-written without EXIF/GPS into ``out_dir/sanitized`` before anything else sees
    them (PLAN 14.8); videos are probed first. Typed ``MediaError`` on bad media.
    """
    items = [Path(inputs)] if isinstance(inputs, (str, Path)) else [Path(p) for p in inputs]
    clean_root = out_dir / "sanitized"
    paths: list[Path] = []
    for i, item in enumerate(items):
        if item.is_dir():
            for p in sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                paths.append(sanitize_image(p, clean_root / str(i), report=report))
        elif not item.exists():
            raise FileNotFoundError(f"missing input: {item}")
        elif item.suffix.lower() in VIDEO_EXTS:
            validate_video(item, report=report)
            paths += extract_frames(item, out_dir / "frames", fps)
        else:  # images, and anything unsupported -> ERR_UNSUPPORTED_MEDIA
            paths.append(sanitize_image(item, clean_root / str(i), report=report))
    if not paths:
        raise MediaError("no input images found", code=ErrorCode.ERR_TOO_FEW_FRAMES)
    return paths


def apply_privacy_blur(
    paths: list[Path], config: Config, out_dir: Path, detector: PrivacyRegionDetector | None = None
) -> tuple[list[Path], dict[Path, np.ndarray]]:
    """Blur face/plate regions (after sanitization); returns new paths and ``path -> mask``."""
    det = detector if detector is not None else build_privacy_detector(config.privacy_model_path)
    new_paths, masks = [], {}
    for p in paths:
        dest, mask = blur_image_file(p, out_dir / "privacy", det, config.privacy_blur_sigma)
        new_paths.append(dest)
        masks[dest] = mask
    return new_paths, masks


def make_frames(
    paths: list[Path],
    preprocess_mode: str | None = None,
    masks: dict[Path, np.ndarray] | None = None,
) -> list[Frame]:
    """Frames at native size. With a backbone ``preprocess_mode`` ("crop"/"pad") each frame also
    gets the exact `GridMap` onto that backbone's input grid (PLAN 8.1)."""
    from PIL import Image

    from countonce.ingest.frames import backbone_grid_map

    frames = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            w, h = im.size
        gm = backbone_grid_map(w, h, preprocess_mode) if preprocess_mode else None
        pm = masks.get(p) if masks else None
        frames.append(
            Frame(id=i, path=p, width=w, height=h, sha256=_sha256(p), privacy_mask=pm, grid_map=gm)
        )
    return frames


def regrid_frames(frames: list[Frame], mode: str | None, rec: object) -> list[Frame]:
    """Frames whose ``grid_map`` matches the reconstructor's current (lowered) input tier."""
    target = getattr(rec, "grid_target", None)
    if not mode or target is None:
        return frames
    from countonce.ingest.frames import backbone_grid_map

    return [replace(f, grid_map=backbone_grid_map(f.width, f.height, mode, target)) for f in frames]


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
    """Associator at strictness ``k``, wrapped by the motion filter when it is enabled."""
    inner = _make_base_associator(config, k)
    if not config.motion_filter:
        return inner
    from countonce.merge.motion import MotionAwareAssociator, MotionConfig

    cfg = MotionConfig(
        residual_norm=config.motion_residual_norm,
        min_agree=config.motion_min_agree,
        min_views=config.motion_min_views,
        conf_factor=config.motion_conf_factor,
    )
    return MotionAwareAssociator(inner, cfg)  # type: ignore[return-value]


def _make_base_associator(config: Config, k: float) -> Associator:
    """Associator at strictness ``k``: v1 uses it directly, v2 scales its reprojection gate."""
    if config.assoc == "v1":
        return V1Associator(k=k, cannot_link=config.cannot_link)
    base = V2Config()
    gate = base.max_reproj_norm * k / config.merge_k
    v2cfg = V2Config(
        **{
            **base.__dict__,
            "max_reproj_norm": gate,
            "cannot_link": config.cannot_link,
            "use_appearance": config.appearance,
            "hallucination": config.hallucination,
        }
    )
    if config.assoc == "v3":
        from countonce.merge.v3 import V3Associator, load_cached

        # k is the sensitivity grid's merge strictness: larger k merges more, so the match
        # probability threshold drops (0.5 at the configured merge_k).
        thr = min(max(1.0 - 0.5 * k / config.merge_k, 0.05), 0.95)
        return V3Associator(model=load_cached(config.v3_checkpoint), threshold=thr)
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
    privacy_detector: PrivacyRegionDetector | None = None,
) -> CountResult:
    """Count distinct instances of ``prompt`` in the images/video at ``inputs``.

    ``detector`` / ``reconstructor`` override the config-selected backends (tests, notebooks);
    ``privacy_detector`` likewise overrides the model-file detector used when ``privacy_blur``.
    """
    if config.assoc not in ("v1", "v2", "v3", "sync"):
        raise NotImplementedError(f"associator {config.assoc!r} is not implemented yet")
    if config.assoc == "v3":
        from countonce.merge.v3 import require_checkpoint

        require_checkpoint(config.v3_checkpoint)  # fail before any expensive stage
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # Ingest
    out_dir = Path(config.out_dir)
    sanitize_report = SanitizeReport()
    paths = collect_paths(inputs, out_dir, config.fps, sanitize_report)
    if config.blur_threshold > 0:
        paths, _ = drop_blurry(paths, config.blur_threshold)
    if config.dedup_threshold > 0:
        paths, _ = drop_near_duplicates(paths, config.dedup_threshold)
    if config.max_frames > 0:
        paths = cap_frames(paths, config.max_frames)
    if not paths:
        raise ValueError("no frames left after filtering")
    privacy_masks = None
    if config.privacy_blur:
        paths, privacy_masks = apply_privacy_blur(paths, config, out_dir, privacy_detector)
    rec = reconstructor if reconstructor is not None else build_reconstructor(config)
    configure_inference(rec, config, "recon")
    if config.sdpa_check:
        check_sdpa("reconstructor")
    frames = make_frames(paths, getattr(rec, "preprocess_mode", None), privacy_masks)
    timings["ingest"] = time.perf_counter() - t0

    # Detect
    t = time.perf_counter()
    det = detector if detector is not None else build_detector(config)
    configure_inference(det, config, "detector")
    base_det = det  # wrappers hide capability flags
    det = wrap_detector(det, config)
    if config.cache_dir is not None:
        det = CachedDetector(det, DetectionCache(Path(config.cache_dir) / "detect"))

    parsed = parse_prompt(prompt) if config.negative_prompts else ParsedPrompt([prompt], [], prompt)
    neg_warnings: list[dict[str, str]] = []
    use_negatives = parsed.has_negatives
    if use_negatives and not negatives_supported(base_det):
        log.warning(
            "detector %s has no phrase scores; ignoring negative prompts %s",
            base_det.name,
            parsed.negatives,
        )
        neg_warnings.append(negatives_ignored_warning(base_det.name, parsed.negatives))
        use_negatives = False
    detect_prompt = parsed.positive_prompt if parsed.has_negatives else prompt
    if not use_negatives and parsed.has_negatives:
        detect_prompt = prompt  # nothing is stripped when negatives cannot be applied

    def run_detect(fr: list[Frame]) -> list[Detection]:
        if use_negatives:
            return detect_with_negatives(det, fr, parsed, config.negative_margin)
        return det.detect(fr, detect_prompt)

    def detect_stage(att: Attempt) -> list[Detection]:
        if att.resolution_scale < 1.0:
            raise RungUnavailable("detectors have no resolution control")
        if att.chunk_size is None:
            return run_detect(frames)
        out: list[Detection] = []  # smaller detector batch; free memory between batches
        for i in range(0, len(frames), att.chunk_size):
            out.extend(run_detect(frames[i : i + att.chunk_size]))
            free_cache()
        return out

    raw_detections = run_with_oom_ladder(
        detect_stage, stage_name="detect", n_frames=len(frames), resolution_tiers=()
    ).value
    detections = filter_by_score(raw_detections, config.score_threshold)
    timings["detect"] = time.perf_counter() - t

    # Reconstruct, under the OOM degradation ladder (rung 0 = the configured run, unchanged)
    t = time.perf_counter()
    with_raising = getattr(rec, "with_oom_raising", None)
    base_rec = with_raising() if with_raising is not None else rec  # OOM must reach the ladder
    cache_dir = Path(config.cache_dir) if config.cache_dir is not None else None

    def recon_stage(att: Attempt) -> tuple[DenseRecon, str | None, bool, list[Frame]]:
        r = base_rec
        stage_frames = frames
        if att.resolution_scale < 1.0:
            with_res = getattr(r, "with_resolution_scale", None)
            if with_res is None:
                raise RungUnavailable("backbone has no resolution control")
            r = with_res(att.resolution_scale)
            stage_frames = regrid_frames(frames, getattr(rec, "preprocess_mode", None), r)
        if att.chunk_size is not None:
            # a ladder chunk can be smaller than the configured overlap; keep overlap < chunk
            overlap = max(2, min(config.chunk_overlap, att.chunk_size - 1))
            r = ChunkedReconstructor(r, att.chunk_size, overlap, config.chunk_stitch)
        elif config.chunked_recon:
            cap = _chunk_cap(config)
            if cap is not None:
                r = ChunkedReconstructor(r, cap, config.chunk_overlap, config.chunk_stitch)
        if cache_dir is not None:
            got = CachedReconstructor(r, ReconCache(cache_dir / "recon"), "0").reconstruct_cached(
                stage_frames
            )
            return got.recon, got.key, got.hit, stage_frames
        return r.reconstruct(stage_frames), None, False, stage_frames

    ladder = run_with_oom_ladder(recon_stage, stage_name="recon", n_frames=len(frames))
    recon, cache_key, cache_hit, frames = ladder.value  # type: ignore[misc]
    timings["recon"] = time.perf_counter() - t

    # Lift + associate
    t = time.perf_counter()
    lift_kw = lift_kwargs(config, recon)
    detections, lift_fallbacks = apply_lift_region(detections, config.lift_region)
    lifted = lift_detections(detections, recon, {f.id: f for f in frames}, **lift_kw)
    frame_map = {f.id: f for f in frames}
    clustered = _make_associator(config, config.merge_k).associate(lifted, recon, frame_map)
    timings["associate"] = time.perf_counter() - t

    # Sensitivity range: re-count over a grid of merge strictness and detector threshold.
    t = time.perf_counter()
    lift_by_thr = {config.score_threshold: lifted}

    def count_at(k: float, thr: float) -> int:
        if thr not in lift_by_thr:
            kept = apply_lift_region(filter_by_score(raw_detections, thr), config.lift_region)[0]
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
        thresholds=config.warning_thresholds(),
    )
    zero_warn, below = zero_count_warning(
        prompt=prompt,
        count=count,
        interval_low=interval.low,
        raw_detections=raw_detections,
        threshold=config.score_threshold,
        thresholds=config.warning_thresholds(),
    )
    result = CountResult(
        prompt=prompt,
        count=count,
        interval=interval,
        confidence_method="sensitivity",
        warnings=quality + neg_warnings + interval_warnings + ([zero_warn] if zero_warn else []),
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
        detections=build_detection_records(lifted, recon, frame_map, config.store_features),
        cameras=build_camera_records(recon),
        timings=timings,
        oom_fallback_rung=ladder.rung,
        peak_vram_mb=recon.peak_vram_mb,
        config={
            **run_metadata(config.to_dict()),
            "lift_fallbacks": lift_fallbacks,
            **({"below_threshold": below} if zero_warn else {}),
            "exif_stripped": sanitize_report.to_dict()["exif_stripped"],
            "sanitize": sanitize_report.to_dict(),
        },
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
