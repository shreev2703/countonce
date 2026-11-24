"""Strict, versioned wire schemas (docs/PLAN.md section 8.10).

Every persisted ``result.json`` / ``markers.json`` carries ``schema_version``.
Readers accept every older version through the ``UPGRADERS`` registry, so old
shared links and historical jobs keep loading after the pipeline moves on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CURRENT_VERSION = "1.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntervalModel(_Strict):
    low: int
    high: int
    level: float = Field(gt=0.0, lt=1.0)
    method: str
    n_calibration: int | None = None


class ModelVersionsModel(_Strict):
    detector: str
    backbone: str
    associator: str
    calibration: str | None = None


class InstanceModel(_Strict):
    id: int
    xyz: tuple[float, float, float]
    size: float
    members: list[str]
    n_views: int
    conf: float = Field(ge=0.0, le=1.0)
    possibly_moving: bool = False


class MatchEvidenceModel(_Strict):
    det_a: str
    det_b: str
    dist_3d_norm: float
    reproj_error_px: float
    appearance_sim: float
    match_prob: float
    merged: bool
    reason: str


class WarningModel(_Strict):
    code: str
    message: str
    fix: str | None = None


class ResultV1(_Strict):
    schema_version: str = "1.0"
    prompt: str
    negative_prompts: list[str] = []
    count: int = Field(ge=0)
    zero_count: bool = False
    interval: IntervalModel
    confidence_method: str
    model_versions: ModelVersionsModel
    input_hash: str
    recon_cache_key: str | None = None
    recon_cache_hit: bool = False
    oom_fallback_rung: int = 0
    peak_vram_mb: float = 0.0
    timings: dict[str, float] = {}
    config: dict[str, Any] = {}
    instances: list[InstanceModel] = []
    per_frame_counts: dict[str, int] = {}
    detection_status: dict[str, str] = {}
    reject_reasons: dict[str, str] = {}
    match_evidence: list[MatchEvidenceModel] = []
    warnings: list[WarningModel] = []


class MarkerModel(_Strict):
    id: int
    xyz: tuple[float, float, float]
    size: float
    conf: float = Field(ge=0.0, le=1.0)
    n_views: int
    members: list[str]
    possibly_moving: bool = False


class MarkersV1(_Strict):
    schema_version: str = "1.0"
    count: int = Field(ge=0)
    interval_low: int
    interval_high: int
    scene_scale: float | None = None
    markers: list[MarkerModel] = []


# version -> function upgrading a raw dict of that version to the next one.
# Add an entry (and a stored example under tests/data/) with every version bump.
UPGRADERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

_RESULT_MODELS: dict[str, type[BaseModel]] = {"1.0": ResultV1}
_MARKER_MODELS: dict[str, type[BaseModel]] = {"1.0": MarkersV1}


def _upgrade(raw: dict[str, Any]) -> dict[str, Any]:
    version = raw.get("schema_version")
    if not isinstance(version, str):
        raise ValueError("missing schema_version")
    seen = set()
    while version != CURRENT_VERSION:
        if version in seen or version not in UPGRADERS:
            raise ValueError(f"unsupported schema_version {version!r}")
        seen.add(version)
        raw = UPGRADERS[version](dict(raw))
        version = raw.get("schema_version")
    return raw


def load_result(raw: dict[str, Any]) -> ResultV1:
    """Validate (and upgrade, if old) a raw ``result.json`` dict."""
    return ResultV1.model_validate(_upgrade(raw))


def load_markers(raw: dict[str, Any]) -> MarkersV1:
    """Validate (and upgrade, if old) a raw ``markers.json`` dict."""
    return MarkersV1.model_validate(_upgrade(raw))
