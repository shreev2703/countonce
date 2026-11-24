"""Core data model (docs/PLAN.md section 7), shared across the pipeline.

Plain dataclasses with numpy arrays. Serialization to the strict wire format
lives in ``countonce/schemas/``; ``CountResult.to_dict`` is a light helper only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"

DETECTION_STATUSES = ("new", "duplicate", "rejected")
LIFT_FAIL_REASONS = ("low_conf", "privacy_mask", "too_few_pixels")
INTERVAL_METHODS = ("sensitivity", "split_conformal", "jackknife_plus")
CONFIDENCE_METHODS = ("heuristic", "sensitivity", "temperature_scaled", "learned")


@dataclass(frozen=True)
class PixelMap:
    """Affine map from a frame's working pixels back to its original image (PLAN 8.1).

    ``x_orig = x * sx + ox``, ``y_orig = y * sy + oy``. ``orig_width``/``orig_height`` are the
    original image size, so overlays can be drawn in the image the user actually supplied.
    """

    sx: float
    sy: float
    ox: float
    oy: float
    orig_width: int
    orig_height: int

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        return x * self.sx + self.ox, y * self.sy + self.oy


@dataclass
class Frame:
    id: int
    path: Path
    width: int
    height: int
    sha256: str  # content hash (stage caching)
    privacy_mask: np.ndarray | None = None  # HxW bool, blurred faces/plates
    blur_score: float = 0.0
    pixel_map: PixelMap | None = None  # working pixels -> original pixels (set by resize_to_grid)
    original_path: Path | None = None  # the un-resized image that ``pixel_map`` maps onto

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.original_path is not None:
            self.original_path = Path(self.original_path)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame width/height must be positive")
        if self.privacy_mask is not None and self.privacy_mask.shape != (self.height, self.width):
            raise ValueError("privacy_mask must have shape (height, width)")


@dataclass
class Detection:
    id: str  # f"{frame_id}:{k}"
    frame_id: int
    score: float
    box_xyxy: np.ndarray | None = None  # (4,) pixels
    point_xy: np.ndarray | None = None  # (2,) pixels
    mask: np.ndarray | None = None  # HxW bool
    feature: np.ndarray | None = None  # appearance embedding, filled on demand

    def __post_init__(self) -> None:
        if self.box_xyxy is None and self.point_xy is None and self.mask is None:
            raise ValueError("Detection needs at least one of box_xyxy, point_xy, mask")
        if self.box_xyxy is not None and np.shape(self.box_xyxy) != (4,):
            raise ValueError("box_xyxy must have shape (4,)")
        if self.point_xy is not None and np.shape(self.point_xy) != (2,):
            raise ValueError("point_xy must have shape (2,)")

    @staticmethod
    def make_id(frame_id: int, k: int) -> str:
        return f"{frame_id}:{k}"


@dataclass
class Recon:
    """Dense multi-view reconstruction (extrinsics are world->camera, OpenCV)."""

    backbone: str  # "vggt" | "vggt_omega" | "vggt_t3" | "colmap"
    intrinsics: np.ndarray  # (N,3,3)
    extrinsics: np.ndarray  # (N,3,4)
    depth: np.ndarray  # (N,H,W)
    depth_conf: np.ndarray  # (N,H,W)
    world_points: np.ndarray  # (N,H,W,3)
    world_points_conf: np.ndarray  # (N,H,W)
    scene_scale: float = 1.0
    runtime_s: float = 0.0
    peak_vram_mb: float = 0.0

    def __post_init__(self) -> None:
        n = self.intrinsics.shape[0]
        if self.intrinsics.shape != (n, 3, 3):
            raise ValueError("intrinsics must be (N,3,3)")
        if self.extrinsics.shape != (n, 3, 4):
            raise ValueError("extrinsics must be (N,3,4)")
        if self.depth.ndim != 3 or self.depth.shape[0] != n:
            raise ValueError("depth must be (N,H,W)")
        h, w = self.depth.shape[1:]
        if self.depth_conf.shape != (n, h, w):
            raise ValueError("depth_conf must match depth shape")
        if self.world_points.shape != (n, h, w, 3):
            raise ValueError("world_points must be (N,H,W,3)")
        if self.world_points_conf.shape != (n, h, w):
            raise ValueError("world_points_conf must be (N,H,W)")

    @property
    def n_views(self) -> int:
        return int(self.intrinsics.shape[0])


@dataclass
class LiftedDetection:
    det: Detection
    xyz: np.ndarray | None = None
    size: float | None = None
    lift_conf: float = 0.0
    motion_residual: float | None = None  # cross-view reprojection residual
    fail_reason: str | None = None  # one of LIFT_FAIL_REASONS, or None
    possibly_moving: bool = False  # residual flags the object as dynamic

    def __post_init__(self) -> None:
        if self.fail_reason is not None and self.fail_reason not in LIFT_FAIL_REASONS:
            raise ValueError(f"unknown fail_reason {self.fail_reason!r}")
        if self.xyz is not None and np.shape(self.xyz) != (3,):
            raise ValueError("xyz must have shape (3,)")

    @property
    def ok(self) -> bool:
        return self.fail_reason is None and self.xyz is not None


@dataclass
class MatchEvidence:
    """Powers the 'Explain this match' inspector."""

    det_a: str
    det_b: str
    dist_3d_norm: float  # 3D distance / mean size
    reproj_error_px: float
    appearance_sim: float
    match_prob: float  # calibrated probability
    merged: bool
    reason: str  # e.g. "same_frame_cannot_link", "below_threshold"


@dataclass
class Instance:
    id: int
    xyz: np.ndarray
    size: float
    members: list[str]
    n_views: int
    conf: float  # calibrated
    possibly_moving: bool = False  # dynamic-scene flag

    def __post_init__(self) -> None:
        if np.shape(self.xyz) != (3,):
            raise ValueError("xyz must have shape (3,)")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("conf must be in [0, 1]")


@dataclass
class Interval:
    low: int
    high: int
    level: float  # e.g. 0.90
    method: str  # one of INTERVAL_METHODS
    n_calibration: int | None = None

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("interval low must be <= high")
        if not 0.0 < self.level < 1.0:
            raise ValueError("level must be in (0, 1)")
        if self.method not in INTERVAL_METHODS:
            raise ValueError(f"unknown interval method {self.method!r}")


@dataclass
class ModelVersions:
    detector: str  # name + checkpoint hash
    backbone: str
    associator: str  # "v1" | "v2" | "v3@<checkpoint>"
    calibration: str | None = None


@dataclass
class CountResult:
    prompt: str
    count: int
    interval: Interval
    model_versions: ModelVersions
    input_hash: str  # content-addressable cache key
    schema_version: str = SCHEMA_VERSION
    confidence_method: str = "heuristic"  # one of CONFIDENCE_METHODS
    recon_cache_key: str | None = None  # key of the reconstruction used
    negative_prompts: list[str] = field(default_factory=list)
    zero_count: bool = False
    instances: list[Instance] = field(default_factory=list)
    per_frame_counts: dict[int, int] = field(default_factory=dict)
    detection_status: dict[str, str] = field(default_factory=dict)
    reject_reasons: dict[str, str] = field(default_factory=dict)
    match_evidence: list[MatchEvidence] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)  # {code, message, fix}
    recon_cache_hit: bool = False
    oom_fallback_rung: int = 0  # 0 = no fallback used
    timings: dict[str, float] = field(default_factory=dict)
    peak_vram_mb: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be >= 0")
        if self.confidence_method not in CONFIDENCE_METHODS:
            raise ValueError(f"unknown confidence_method {self.confidence_method!r}")
        bad = set(self.detection_status.values()) - set(DETECTION_STATUSES)
        if bad:
            raise ValueError(f"unknown detection status {sorted(bad)}")

    @property
    def n_possibly_moving(self) -> int:
        return sum(i.possibly_moving for i in self.instances)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (numpy -> lists, Path -> str)."""
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
