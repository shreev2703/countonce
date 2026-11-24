"""End-to-end pipeline: ingest -> detect -> recon -> lift -> associate -> export (PLAN 8.11)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from countonce.config import Config
from countonce.detect.base import Detector, filter_by_score
from countonce.detect.cache import CachedDetector, DetectionCache
from countonce.export import export_all
from countonce.ingest.frames import IMAGE_EXTS, drop_blurry, extract_frames
from countonce.ingest.select import cap_frames, drop_near_duplicates
from countonce.lift import compute_scene_scale, lift_detections
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator
from countonce.provenance import run_metadata
from countonce.recon.base import Reconstructor
from countonce.recon.cache import CachedReconstructor, ReconCache
from countonce.types import CountResult, Frame, Interval, ModelVersions
from countonce.types import Recon as DenseRecon

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


def make_frames(paths: list[Path]) -> list[Frame]:
    from PIL import Image

    frames = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            w, h = im.size
        frames.append(Frame(id=i, path=p, width=w, height=h, sha256=_sha256(p)))
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
    if config.assoc not in ("v1", "v2"):
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
    frames = make_frames(paths)
    timings["ingest"] = time.perf_counter() - t0

    # Detect
    t = time.perf_counter()
    det = detector if detector is not None else build_detector(config)
    if config.cache_dir is not None:
        det = CachedDetector(det, DetectionCache(Path(config.cache_dir) / "detect"))
    detections = filter_by_score(det.detect(frames, prompt), config.score_threshold)
    timings["detect"] = time.perf_counter() - t

    # Reconstruct
    t = time.perf_counter()
    rec = reconstructor if reconstructor is not None else build_reconstructor(config)
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
    lifted = lift_detections(detections, recon, {f.id: f for f in frames})
    associator = V1Associator(k=config.merge_k) if config.assoc == "v1" else V2Associator()
    clustered = associator.associate(lifted, recon, {f.id: f for f in frames})
    timings["associate"] = time.perf_counter() - t

    per_frame: dict[int, int] = {f.id: 0 for f in frames}
    for ld in lifted:
        if ld.ok:
            per_frame[ld.det.frame_id] += 1

    count = len(clustered.instances)
    result = CountResult(
        prompt=prompt,
        count=count,
        # Placeholder until the sensitivity range lands: a degenerate interval at the count.
        interval=Interval(count, count, config.interval_level, "sensitivity"),
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


def _input_hash(frames: list[Frame]) -> str:
    from countonce.recon.cache import input_hash

    return input_hash(frames)
