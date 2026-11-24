"""Typed run configuration (docs/PLAN.md section 8.11), loadable from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from countonce.warnings import WarningThresholds

DetectorName = Literal["countgd", "gdino", "countgdpp", "color_blob"]
BackboneName = Literal["vggt", "vggt_omega", "vggt_t3", "colmap", "fixture"]
AssocName = Literal["v1", "v2", "v3", "sync"]
HallucinationMode = Literal["off", "downweight", "drop"]
LiftRegion = Literal["mask", "box"]
MasksMode = Literal["none", "box_fill", "sam"]
CompileModeName = Literal["off", "default", "reduce-overhead"]
IntervalMethodName = Literal["sensitivity", "conformal", "jackknife_plus"]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: DetectorName = "countgd"
    backbone: BackboneName = "vggt"
    assoc: AssocName = "v1"

    # Ablation switches (PLAN 9.6). Defaults reproduce the shipped behaviour.
    cannot_link: bool = True  # v1/v2: never merge two detections from the same frame
    appearance: bool = True  # v2/sync: use Detection.feature (cost, gate, split check)
    # Natural-language prompts (PLAN 15): "red apples, not leaves" -> positives + negatives.
    # Off passes the prompt to the detector verbatim. A box is dropped when its best negative
    # phrase score exceeds its positive score by more than ``negative_margin``.
    negative_prompts: bool = True
    negative_margin: float = Field(0.0, ge=0)
    hallucination: HallucinationMode = "downweight"  # v2: one-view ghost filter
    lift_region: LiftRegion = "mask"  # "mask": use Detection.mask when present; "box": box centre
    # Mask producer run after detection (PLAN 8.2 masks.py): "none" leaves Detection.mask unset
    # (lifting falls back to the central box region), "box_fill" fills each box, "sam" prompts SAM 2
    # with each box and needs ``mask_checkpoint`` (and the ``sam2`` package).
    masks: MasksMode = "none"
    mask_checkpoint: Path | None = None
    mask_model_cfg: str = "sam2_hiera_s.yaml"
    # Dynamic-object residual filter (PLAN 8.3b): flag detections that do not reproject onto a
    # detection in the other views, downweight them and merge moving-object ghosts.
    motion_filter: bool = False
    motion_residual_norm: float = Field(0.5, gt=0)  # view agrees if residual / box size <= this
    motion_min_agree: float = Field(0.5, ge=0, le=1)  # flag if agreeing share of views is below
    motion_min_views: int = Field(2, ge=1)  # checked views needed before flagging
    motion_conf_factor: float = Field(0.5, ge=0, le=1)  # conf multiplier, fully flagged instance

    # Ingest
    fps: float = Field(2.0, gt=0)  # video sampling rate
    max_frames: int = Field(32, ge=0)  # 0 = no cap
    dedup_threshold: float = Field(0.02, ge=0)  # near-duplicate frame filter; 0 = off
    blur_threshold: float = Field(0.0, ge=0)  # Laplacian variance floor; 0 = off

    # Chunked reconstruction (PLAN 8.3): stitch overlapping chunks when frames exceed the cap
    chunked_recon: bool = False
    chunk_size: int = Field(0, ge=0)  # 0 = use the memory-guard calibration (no chunking if none)
    chunk_overlap: int = Field(4, ge=2)  # shared frames between adjacent chunks
    chunk_stitch: Literal["chain", "pose_graph"] = "chain"  # pose_graph: global Sim(3) sync

    # Detect / lift / associate
    score_threshold: float = Field(0.3, ge=0, le=1)
    merge_k: float = Field(0.5, gt=0)  # v1 distance threshold, in object sizes
    # Lift rules; None = dense defaults (5 pixels, 30th percentile) or, for the sparse COLMAP
    # backbone, 3 pixels and no percentile cut.
    lift_min_pixels: int | None = Field(None, ge=1)
    lift_conf_percentile: float | None = Field(None, ge=0, le=100)
    interval_level: float = Field(0.9, gt=0, lt=1)
    # Interval method (PLAN 8.8). "conformal"/"jackknife_plus" need a calibration artifact
    # (bench.calibration + countonce.conformal.CalibrationArtifact JSON).
    interval_method: IntervalMethodName = "sensitivity"
    calibration_path: Path | None = None
    # Quality-warning threshold overrides (PLAN 8.9), keyed by WarningThresholds field name, e.g.
    # ``warnings: {min_frames: 4}``. configs/warnings.yaml (bench.tune_warnings) has the same shape.
    warnings: dict[str, float] = Field(default_factory=dict)
    # Pre-flight capture-check threshold overrides (countonce.ingest.preflight), PROVISIONAL.
    preflight: dict[str, float] = Field(default_factory=dict)
    # Trained CountOnce-Match checkpoint (match.model.save_checkpoint); required for assoc: v3.
    v3_checkpoint: Path | None = None

    # Persist Detection.feature vectors in result.json (schema 1.1) so "explain" can report the
    # appearance cosine. Off by default: embeddings are large.
    store_features: bool = False

    # Face/plate blur (PLAN 14.8). Off by default so bench pixels are untouched; when on, a
    # model file (OpenCV .xml cascade or .onnx face model) must be supplied: nothing is downloaded.
    privacy_blur: bool = False
    privacy_model_path: Path | None = None
    privacy_blur_sigma: float = Field(12.0, gt=0)

    # Backend options passed to the real detector / reconstructor (paths, device, ...)
    detector_options: dict[str, Any] = Field(default_factory=dict)
    backbone_options: dict[str, Any] = Field(default_factory=dict)

    # Inference optimization (PLAN 13 items 2-5). Defaults reproduce the shipped eager behaviour;
    # all switches are no-ops without CUDA.
    compile_reconstructor: CompileModeName = "off"  # torch.compile mode for the backbone
    compile_detector: CompileModeName = "off"
    detector_batch_size: int = Field(0, ge=0)  # frames per detector call; 0 = all at once
    preload_workers: int = Field(0, ge=0)  # frame-loader threads; 0 = load serially
    pin_memory: bool = False  # pin preloaded frame tensors (CUDA only)
    sdpa_check: bool = False  # log the attention kernel in use; warn on the math fallback

    # Output / caching
    out_dir: Path = Path("results")
    cache_dir: Path | None = None  # None = no stage caching

    @field_validator("warnings")
    @classmethod
    def _check_warning_keys(cls, v: dict[str, float]) -> dict[str, float]:
        from countonce.warnings import WarningThresholds

        bad = set(v) - set(WarningThresholds.__dataclass_fields__)
        if bad:
            raise ValueError(f"unknown warning threshold(s): {sorted(bad)}")
        return v

    @field_validator("preflight")
    @classmethod
    def _check_preflight_keys(cls, v: dict[str, float]) -> dict[str, float]:
        from countonce.ingest.preflight import PreflightThresholds

        bad = set(v) - set(PreflightThresholds.__dataclass_fields__)
        if bad:
            raise ValueError(f"unknown preflight threshold(s): {sorted(bad)}")
        return v

    def warning_thresholds(self) -> WarningThresholds:
        """Default thresholds with this config's ``warnings:`` overrides applied."""
        from countonce.warnings import WarningThresholds

        return WarningThresholds.from_overrides(self.warnings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def with_overrides(self, **overrides: Any) -> Config:
        """Copy with non-None overrides applied (used by the CLI)."""
        data = self.model_dump()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return Config(**data)
