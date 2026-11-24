"""Data-quality gates for rendered synthetic scenes (PLAN §10.2, §16B.1).

A rendered scene is described by `RenderedScene` (plain numpy arrays, renderer-agnostic) plus a
`SceneMetadata` record. `validate_scene` returns the quarantine reasons (empty = clean);
`validate_scenes` aggregates them into a `QuarantineReport` that serializes to JSON and Markdown.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_ERROR = "schema_invalid"
NAN_DEPTH = "nan_or_inf_depth"
BLACK_FRAME = "all_black_frame"
OVEREXPOSED = "overexposed_frame"
ZERO_TARGET = "zero_visible_target_pixels"
FEW_VIEWS = "fewer_than_2_views_see_target"
DEGENERATE_RIG = "degenerate_camera_rig"
REASONS = (
    SCHEMA_ERROR,
    NAN_DEPTH,
    BLACK_FRAME,
    OVEREXPOSED,
    ZERO_TARGET,
    FEW_VIEWS,
    DEGENERATE_RIG,
)

BLACK_MEAN_MAX = 2.0  # mean uint8 intensity at or below this = all black
OVEREXPOSED_FRACTION = 0.98  # fraction of pixels at >= 250 in every channel
MIN_CAMERA_SPREAD = 1e-3  # relative to mean distance of cameras from their centroid
MIN_TARGET_PIXELS = 1
TARGET_QUARANTINE_TARGET = 0.02  # PLAN: aim for < 2% quarantined


class SceneMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    recipe_version: str
    seed: int = Field(ge=0)
    renderer: str
    prompt: str
    n_objects: int = Field(ge=1)
    n_views: int = Field(ge=1)
    target_ids: list[int] = Field(min_length=1)
    object_positions: list[tuple[float, float, float]]
    resolution: tuple[int, int]


@dataclass
class RenderedScene:
    """images (V,H,W,3) uint8; depth (V,H,W) float; instance_masks (V,H,W) int (0 = background);
    camera centres (V,3) in world coordinates; metadata is a raw dict (schema-checked here)."""

    metadata: dict
    images: np.ndarray
    depth: np.ndarray
    instance_masks: np.ndarray
    camera_centers: np.ndarray
    camera_forward: np.ndarray | None = None  # (V,3) optional viewing directions
    extras: dict = field(default_factory=dict)


def check_schema(metadata: dict) -> list[str]:
    try:
        SceneMetadata.model_validate(metadata)
    except ValidationError as e:
        return [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
    return []


def degenerate_rig(centers: np.ndarray, forward: np.ndarray | None = None) -> bool:
    c = np.asarray(centers, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2 or not np.isfinite(c).all():
        return True
    spread = np.linalg.norm(c - c.mean(axis=0), axis=1).mean()
    scale = max(np.linalg.norm(c.mean(axis=0)), 1.0)
    if spread / scale < MIN_CAMERA_SPREAD:
        return True
    if forward is not None:
        f = np.asarray(forward, dtype=float)
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        if np.linalg.norm(f - f.mean(axis=0), axis=1).max() < MIN_CAMERA_SPREAD:
            return True
    return False


def validate_scene(scene: RenderedScene) -> list[str]:
    """Return the sorted list of quarantine reasons for one scene (empty means clean)."""
    if check_schema(scene.metadata):
        return [SCHEMA_ERROR]  # remaining checks rely on valid metadata
    reasons: set[str] = set()
    imgs = np.asarray(scene.images)
    for f in imgs:
        if float(f.mean()) <= BLACK_MEAN_MAX:
            reasons.add(BLACK_FRAME)
        if float((f >= 250).all(axis=-1).mean()) >= OVEREXPOSED_FRACTION:
            reasons.add(OVEREXPOSED)
    if not np.isfinite(scene.depth).all():
        reasons.add(NAN_DEPTH)
    targets = np.asarray(scene.metadata["target_ids"])
    visible = np.isin(scene.instance_masks, targets)
    per_view = visible.reshape(visible.shape[0], -1).sum(axis=1)
    if int(per_view.sum()) < MIN_TARGET_PIXELS:
        reasons.add(ZERO_TARGET)
    # a target is "seen" in a view when it has visible pixels there; need >= 2 such views for any
    seen_views = np.zeros(len(targets), dtype=int)
    for j, tid in enumerate(targets):
        seen_views[j] = int(((scene.instance_masks == tid).reshape(len(imgs), -1).sum(1) > 0).sum())
    if not (seen_views >= 2).any():
        reasons.add(FEW_VIEWS)
    if degenerate_rig(scene.camera_centers, scene.camera_forward):
        reasons.add(DEGENERATE_RIG)
    return sorted(reasons)


@dataclass
class QuarantineReport:
    n_scenes: int = 0
    quarantined: dict[str, list[str]] = field(default_factory=dict)  # scene_id -> reasons

    @property
    def n_quarantined(self) -> int:
        return len(self.quarantined)

    @property
    def fraction(self) -> float:
        return self.n_quarantined / self.n_scenes if self.n_scenes else 0.0

    @property
    def counts_by_reason(self) -> dict[str, int]:
        c = Counter(r for rs in self.quarantined.values() for r in rs)
        return {r: c.get(r, 0) for r in REASONS}

    def to_dict(self) -> dict:
        return {
            "n_scenes": self.n_scenes,
            "n_quarantined": self.n_quarantined,
            "fraction_quarantined": self.fraction,
            "target_fraction": TARGET_QUARANTINE_TARGET,
            "counts_by_reason": self.counts_by_reason,
            "quarantined": self.quarantined,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Synthetic data quarantine report",
            "",
            f"- Scenes checked: {self.n_scenes}",
            f"- Quarantined: {self.n_quarantined} ({self.fraction:.2%}; target < "
            f"{TARGET_QUARANTINE_TARGET:.0%})",
            "",
            "| Reason | Scenes |",
            "|---|---|",
        ]
        lines += [f"| {r} | {n} |" for r, n in self.counts_by_reason.items()]
        if self.quarantined:
            lines += ["", "## Quarantined scenes", ""]
            lines += [f"- `{sid}`: {', '.join(rs)}" for sid, rs in sorted(self.quarantined.items())]
        return "\n".join(lines) + "\n"

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        jp, mp = out / "quarantine.json", out / "quarantine.md"
        jp.write_text(json.dumps(self.to_dict(), indent=2))
        mp.write_text(self.to_markdown())
        return jp, mp


def validate_scenes(scenes: Iterable[RenderedScene]) -> QuarantineReport:
    report = QuarantineReport()
    for i, s in enumerate(scenes):
        report.n_scenes += 1
        reasons = validate_scene(s)
        if reasons:
            sid = str(s.metadata.get("scene_id", f"scene_{i}"))
            if sid in report.quarantined:
                sid = f"{sid}#{i}"
            report.quarantined[sid] = reasons
    return report
