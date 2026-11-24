"""Data-quality gates for real-data loaders (PLAN §16B.1, §18 data-quality test).

``validate_scene`` takes a loader's scene (``BenchScene``, ``FujiScene`` or ``FscImage``) and
returns a list of ``Issue``. Any issue quarantines the scene. The quarantine record matches the
synthetic one in ``synth/validate.py`` (``scene_id`` -> sorted reasons) and is written as
``quarantine.jsonl``, one JSON object per line: ``{"scene_id", "reasons", "details", "dataset"}``.

Image checks are header-only: PIL opens the file lazily (no pixel decode), and a tail-marker
check catches the common truncation case (JPEG EOI, PNG IEND) that a header read cannot see.
Optional ``BenchScene.extra`` keys the loaders may set: ``n_gt_items`` (annotation entries behind
``count``), ``n_cameras`` / ``n_annotated_frames`` (must equal the frame count), ``points_xy`` and
``boxes_xyxy`` (checked against the first frame's size).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MISSING_GT = "missing_gt"
EMPTY_GT = "empty_gt"
GT_COUNT_MISMATCH = "gt_count_mismatch"
IMAGE_COUNT_MISMATCH = "image_count_mismatch"
UNREADABLE_IMAGE = "unreadable_image"
TRUNCATED_IMAGE = "truncated_image"
DUPLICATE_IMAGE = "duplicate_image"
OUT_OF_BOUNDS = "out_of_bounds_annotation"
REASONS = (
    MISSING_GT,
    EMPTY_GT,
    GT_COUNT_MISMATCH,
    IMAGE_COUNT_MISMATCH,
    UNREADABLE_IMAGE,
    TRUNCATED_IMAGE,
    DUPLICATE_IMAGE,
    OUT_OF_BOUNDS,
)
DATASETS = (
    "fuji_sfm",
    "onescene",
    "rgbd_videocount",
    "videocount",
    "fsc147",
    "wgisd",
    "minneapple",
    "applegrowthvision",
    "applegrowthvision_images",
)


@dataclass(frozen=True)
class Issue:
    scene_id: str
    reason: str  # one of REASONS
    detail: str = ""


class DataQualityError(ValueError):
    """A single-scene dataset (Fuji-SfM) failed validation."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(f"{i.scene_id}: {i.reason} ({i.detail})" for i in issues))


# --- image checks -------------------------------------------------------------------------------


def image_size(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    """((w, h), None) on success, else (None, reason). Header-only decode plus a tail check."""
    try:
        if path.stat().st_size == 0:
            return None, UNREADABLE_IMAGE
        with Image.open(path) as im:
            size = im.size
            fmt = im.format
    except (OSError, ValueError, SyntaxError):
        return None, UNREADABLE_IMAGE
    try:
        with open(path, "rb") as f:
            f.seek(-12, 2)
            tail = f.read()
    except OSError:
        return None, TRUNCATED_IMAGE
    if (fmt == "JPEG" and b"\xff\xd9" not in tail) or (fmt == "PNG" and b"IEND" not in tail):
        return size, TRUNCATED_IMAGE
    return size, None


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_images(sid: str, paths: Iterable[Path]) -> tuple[list[Issue], tuple[int, int] | None]:
    issues: list[Issue] = []
    first: tuple[int, int] | None = None
    seen: dict[str, Path] = {}
    for p in paths:
        size, bad = image_size(Path(p))
        if bad:
            issues.append(Issue(sid, bad, str(p)))
        if size is not None and first is None:
            first = size
        if bad == UNREADABLE_IMAGE:
            continue
        try:
            digest = _file_hash(Path(p))
        except OSError:
            continue
        if digest in seen:
            issues.append(Issue(sid, DUPLICATE_IMAGE, f"{p} == {seen[digest]}"))
        else:
            seen[digest] = Path(p)
    return issues, first


def _oob(sid: str, pts: Any, boxes: Any, size: tuple[int, int] | None) -> list[Issue]:
    out: list[Issue] = []
    p = np.asarray(pts if pts is not None else [], dtype=float).reshape(-1, 2)
    b = np.asarray(boxes if boxes is not None else [], dtype=float).reshape(-1, 4)
    if len(p) and not np.isfinite(p).all():
        out.append(Issue(sid, OUT_OF_BOUNDS, "non-finite point"))
    if len(b) and not np.isfinite(b).all():
        out.append(Issue(sid, OUT_OF_BOUNDS, "non-finite box"))
    if size is None:
        return out
    w, h = size
    if len(p):
        bad = int(((p[:, 0] < 0) | (p[:, 0] > w) | (p[:, 1] < 0) | (p[:, 1] > h)).sum())
        if bad:
            out.append(Issue(sid, OUT_OF_BOUNDS, f"{bad} point(s) outside {w}x{h}"))
    if len(b):
        bad = int(
            (
                (b[:, 0] < 0)
                | (b[:, 1] < 0)
                | (b[:, 2] > w)
                | (b[:, 3] > h)
                | (b[:, 2] <= b[:, 0])
                | (b[:, 3] <= b[:, 1])
            ).sum()
        )
        if bad:
            out.append(Issue(sid, OUT_OF_BOUNDS, f"{bad} box(es) degenerate or outside {w}x{h}"))
    return out


# --- per-type rules -----------------------------------------------------------------------------


def _validate_bench(s: Any) -> list[Issue]:
    sid = s.scene_id
    issues: list[Issue] = []
    if s.count is None:
        issues.append(Issue(sid, MISSING_GT, "count is None"))
    elif s.count <= 0:
        issues.append(Issue(sid, EMPTY_GT, f"count={s.count}"))
    n_items = s.extra.get("n_gt_items")
    if n_items is not None and s.count is not None and int(n_items) != int(s.count):
        issues.append(Issue(sid, GT_COUNT_MISMATCH, f"count={s.count} vs {n_items} annotations"))
    for key in ("n_cameras", "n_annotated_frames"):
        n = s.extra.get(key)
        if n is not None and int(n) != len(s.frames):
            issues.append(Issue(sid, IMAGE_COUNT_MISMATCH, f"{len(s.frames)} frames vs {key}={n}"))
    if not s.frames:
        issues.append(Issue(sid, IMAGE_COUNT_MISMATCH, "no frames"))
    img_issues, size = _check_images(sid, s.frames)
    issues += img_issues
    issues += _oob(sid, s.extra.get("points_xy"), s.extra.get("boxes_xyxy"), size)
    return issues


def _validate_fuji(s: Any) -> list[Issue]:
    sid = f"fuji_sfm:{Path(s.root).name}"
    issues: list[Issue] = []
    n = len(s.gt_xyz)
    if n == 0:
        issues.append(Issue(sid, EMPTY_GT, "no GT apples"))
    if len(s.gt_ids) != n:
        issues.append(Issue(sid, GT_COUNT_MISMATCH, f"{len(s.gt_ids)} ids vs {n} positions"))
    if n and not np.isfinite(np.asarray(s.gt_xyz, dtype=float)).all():
        issues.append(Issue(sid, OUT_OF_BOUNDS, "non-finite GT position"))
    n_img = len(s.image_paths)
    for name, d in (("intrinsics", s.intrinsics), ("extrinsics", s.extrinsics)):
        if d and (len(d) != n_img or max(d) >= n_img):
            issues.append(Issue(sid, IMAGE_COUNT_MISMATCH, f"{n_img} images vs {len(d)} {name}"))
    issues += _check_images(sid, s.image_paths)[0]
    return issues


def _validate_fsc(s: Any) -> list[Issue]:
    sid = f"{getattr(s, 'dataset', 'fsc147')}:{s.name}"
    issues: list[Issue] = []
    if s.count == 0:
        issues.append(Issue(sid, EMPTY_GT, "no GT points"))
    if not Path(s.path).is_file():
        return [*issues, Issue(sid, UNREADABLE_IMAGE, "image missing")]
    img_issues, size = _check_images(sid, [s.path])
    return [
        *issues,
        *img_issues,
        *_oob(sid, s.points_xy, getattr(s, "boxes_xyxy", s.exemplar_boxes_xyxy), size),
    ]


def validate_scene(scene: Any) -> list[Issue]:
    """All data-quality issues for one loader scene (empty list = clean)."""
    if hasattr(scene, "gt_xyz"):
        return _validate_fuji(scene)
    if hasattr(scene, "points_xy"):
        return _validate_fsc(scene)
    return _validate_bench(scene)


# --- quarantine ---------------------------------------------------------------------------------


@dataclass
class QuarantineLog:
    """Same shape as ``synth.validate.QuarantineReport``: scene_id -> sorted reasons."""

    dataset: str = ""
    n_scenes: int = 0
    quarantined: dict[str, list[str]] = field(default_factory=dict)
    details: dict[str, list[str]] = field(default_factory=dict)

    def add(self, issues: list[Issue]) -> None:
        self.n_scenes += 1
        if not issues:
            return
        sid = issues[0].scene_id
        self.quarantined[sid] = sorted({i.reason for i in issues})
        self.details[sid] = [f"{i.reason}: {i.detail}" for i in issues]

    @property
    def counts_by_reason(self) -> dict[str, int]:
        c = Counter(r for rs in self.quarantined.values() for r in rs)
        return {r: c.get(r, 0) for r in REASONS}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "scene_id": sid,
                    "reasons": rs,
                    "details": self.details.get(sid, []),
                    "dataset": self.dataset,
                }
            )
            for sid, rs in sorted(self.quarantined.items())
        ]
        p.write_text("\n".join(lines) + ("\n" if lines else ""))
        return p


def load_quarantine(path: str | Path) -> set[str]:
    """Scene ids listed in a ``quarantine.jsonl`` (also accepts synth-style records)."""
    ids: set[str] = set()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            ids.add(str(json.loads(line)["scene_id"]))
    return ids


def filter_scenes(
    dataset: str,
    scenes: list[Any],
    quarantine_path: str | Path | None = None,
    validator: Callable[[Any], list[Issue]] = validate_scene,
) -> tuple[list[Any], QuarantineLog]:
    """Split ``scenes`` into clean ones and a quarantine log (written if ``quarantine_path``)."""
    qlog = QuarantineLog(dataset=dataset)
    clean: list[Any] = []
    for s in scenes:
        issues = validator(s)
        qlog.add(issues)
        if not issues:
            clean.append(s)
    if qlog.quarantined:
        log.warning("%s: quarantined %d/%d scenes", dataset, len(qlog.quarantined), qlog.n_scenes)
    if quarantine_path is not None:
        qlog.write(quarantine_path)
    return clean, qlog


def validate_dataset(
    name: str, root: str | Path, quarantine_path: str | Path | None = None
) -> QuarantineLog:
    """Load ``name`` without validation, then validate every scene and write the quarantine file."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    if name == "fuji_sfm":
        from bench.datasets.fuji_sfm import load_fuji_sfm

        scenes: list[Any] = [load_fuji_sfm(root, validate=False)]
    elif name == "fsc147":
        from bench.datasets.fsc147 import load_fsc147

        scenes = list(load_fsc147(root, validate=False))
    elif name in ("wgisd", "minneapple", "applegrowthvision_images"):
        from bench.detector_sanity import load_images

        scenes = list(load_images(name, root, validate=False))
    else:
        from bench.datasets import load_dataset

        scenes = load_dataset(name, root, validate=False, max_frames=None)
    return filter_scenes(name, scenes, quarantine_path)[1]
