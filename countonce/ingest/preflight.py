"""Pre-flight capture check (PLAN 14.8 client-side pre-flight, 8.1, 8.9).

Python reference for the later browser port (Canvas/WASM): the same three signals, computed
from the photos alone before any GPU work. Per-photo sharpness reuses
``frames.laplacian_variance``; frame differences reuse ``select.thumbnail`` (16x16 grayscale,
mean absolute difference, the same measure as ``drop_near_duplicates``).

Every threshold in ``PreflightThresholds`` is PROVISIONAL: none is tuned on real captures yet.
``bench/tune_warnings.py`` only tunes ``WarningThresholds`` and writes ``configs/warnings.yaml``
once real bench data exists; when that file has a ``min_frames`` entry it is used for
``min_photos`` (see ``load_thresholds``). Override via ``preflight:`` in a config.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from countonce.ingest.frames import laplacian_variance
from countonce.ingest.select import thumbnail

BLURRY = "blurry"
NEAR_DUPLICATE = "near_duplicate"
TOO_FEW_VIEWS = "too_few_views"
LOW_OVERLAP = "low_overlap"
LOW_DIVERSITY = "low_diversity"


@dataclass
class PreflightThresholds:
    """PROVISIONAL defaults, not tuned on real captures."""

    min_photos: int = 3  # same as WarningThresholds.min_frames
    min_sharpness: float = 0.0  # absolute Laplacian-variance floor; 0 = off (resolution-dependent)
    min_sharpness_ratio: float = 0.3  # blurry if below this fraction of the set's median
    dup_distance: float = 0.02  # = Config.dedup_threshold default
    max_neighbor_distance: float = 0.25  # a photo further than this from every other is isolated
    max_isolated_fraction: float = 0.5  # low overlap when more than this share is isolated
    min_diversity: float = 0.03  # mean pairwise thumbnail distance below this = all alike

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, float] | None) -> PreflightThresholds:
        base = cls()
        out: dict[str, Any] = {}
        for k, v in (overrides or {}).items():
            if k not in cls.__dataclass_fields__:
                raise ValueError(f"unknown preflight threshold {k!r}")
            out[k] = type(getattr(base, k))(v)
        return cls(**out)


def load_thresholds(
    overrides: Mapping[str, float] | None = None, tuned: Path | None = Path("configs/warnings.yaml")
) -> PreflightThresholds:
    """Provisional defaults; ``min_photos`` follows tuned ``min_frames`` when that file exists."""
    merged: dict[str, float] = {}
    if tuned is not None and tuned.is_file():
        import yaml

        data = yaml.safe_load(tuned.read_text()) or {}
        vals = data.get("thresholds", data) if isinstance(data, dict) else {}
        if isinstance(vals, dict) and isinstance(vals.get("min_frames"), int | float):
            merged["min_photos"] = vals["min_frames"]
    merged.update(overrides or {})
    return PreflightThresholds.from_overrides(merged)


@dataclass
class PreflightReport:
    photos: list[str]
    sharpness: list[float]
    blurry: list[int]  # 0-based indices
    duplicate_groups: list[list[int]]
    diversity: float  # mean pairwise 16x16 thumbnail distance (0 = identical, 1 = black/white)
    isolated: list[int]  # photos with no sufficiently similar neighbour
    too_few_views: bool
    low_overlap: bool
    tips: list[str] = field(default_factory=list)
    ok: bool = True
    thresholds: dict[str, Any] = field(default_factory=dict)
    thresholds_provisional: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _join(nums: Sequence[int]) -> str:
    s = [str(n) for n in nums]
    return s[0] if len(s) == 1 else ", ".join(s[:-1]) + " and " + s[-1]


def _groups(dist: np.ndarray, cut: float) -> list[list[int]]:
    n = len(dist)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < cut:
                parent[find(i)] = find(j)
    by_root: dict[int, list[int]] = {}
    for i in range(n):
        by_root.setdefault(find(i), []).append(i)
    return [g for g in by_root.values() if len(g) > 1]


def preflight(
    paths: Sequence[Path | str], thresholds: PreflightThresholds | None = None
) -> PreflightReport:
    """Check a set of photos before upload. Photo numbers in tips are 1-based, in input order."""
    th = thresholds or PreflightThresholds()
    ps = [Path(p) for p in paths]
    n = len(ps)
    sharp = [laplacian_variance(p) for p in ps]
    med = statistics.median(sharp) if sharp else 0.0
    blurry = [
        i
        for i, s in enumerate(sharp)
        if s < th.min_sharpness or (n >= 2 and s < th.min_sharpness_ratio * med)
    ]

    thumbs = [thumbnail(p) for p in ps]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = float(np.abs(thumbs[i] - thumbs[j]).mean())
    dup_groups = _groups(dist, th.dup_distance)
    diversity = float(dist[np.triu_indices(n, 1)].mean()) if n >= 2 else 0.0

    isolated: list[int] = []
    if n >= 2:
        for i in range(n):
            others = np.delete(dist[i], i)
            if float(others.min()) > th.max_neighbor_distance:
                isolated.append(i)
    too_few = n < th.min_photos
    low_overlap = n >= 2 and len(isolated) / n > th.max_isolated_fraction

    tips: list[str] = []
    for i in blurry:
        tips.append(f"Photo {i + 1} is blurry, retake it.")
    for g in dup_groups:
        tips.append(
            f"Photos {_join([i + 1 for i in g])} are nearly identical, keep one and move "
            "sideways for the next."
        )
    if too_few:
        tips.append(
            f"Only {n} photo(s); take at least {th.min_photos}, moving sideways a little "
            "between each."
        )
    if low_overlap:
        tips.append(
            "The photos look very different from each other, so they may not overlap; "
            "move less between photos."
        )
    elif n >= th.min_photos and len(dup_groups) == 0 and diversity < th.min_diversity:
        tips.append("The photos all look alike; step to the side between shots.")

    return PreflightReport(
        photos=[p.name for p in ps],
        sharpness=sharp,
        blurry=blurry,
        duplicate_groups=dup_groups,
        diversity=diversity,
        isolated=isolated,
        too_few_views=too_few,
        low_overlap=low_overlap,
        tips=tips,
        ok=not tips,
        thresholds=asdict(th),
    )
