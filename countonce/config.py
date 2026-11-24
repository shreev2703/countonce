"""Typed run configuration (docs/PLAN.md section 8.11), loadable from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from countonce.warnings import WarningThresholds

DetectorName = Literal["countgd", "gdino", "countgdpp"]
BackboneName = Literal["vggt", "vggt_omega", "vggt_t3", "colmap"]
AssocName = Literal["v1", "v2", "v3", "sync"]
HallucinationMode = Literal["off", "downweight", "drop"]
LiftRegion = Literal["mask", "box"]
IntervalMethodName = Literal["sensitivity", "conformal", "jackknife_plus"]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: DetectorName = "countgd"
    backbone: BackboneName = "vggt"
    assoc: AssocName = "v1"

    # Ablation switches (PLAN 9.6). Defaults reproduce the shipped behaviour.
    cannot_link: bool = True  # v1/v2: never merge two detections from the same frame
    appearance: bool = True  # v2/sync: use Detection.feature (cost, gate, split check)
    hallucination: HallucinationMode = "downweight"  # v2: one-view ghost filter
    lift_region: LiftRegion = "mask"  # "mask": use Detection.mask when present; "box": box centre
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
    # Trained CountOnce-Match checkpoint (match.model.save_checkpoint); required for assoc: v3.
    v3_checkpoint: Path | None = None

    # Backend options passed to the real detector / reconstructor (paths, device, ...)
    detector_options: dict[str, Any] = Field(default_factory=dict)
    backbone_options: dict[str, Any] = Field(default_factory=dict)

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
