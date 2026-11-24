"""Shared scene type for the video / multi-view benchmark loaders.

``BenchScene`` is a loader's output: ordered frame paths plus one GT count. Two adapters connect it
to the existing bench code: ``to_sweep_scenes`` (the ``discover_scenes`` dict format that
``bench.sweep`` consumes) and ``to_gt_doc`` (the GT file ``bench.evaluate`` reads).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from countonce.ingest.frames import IMAGE_EXTS


@dataclass
class BenchScene:
    scene_id: str
    frames: list[Path]  # frame order = video / capture order
    count: int  # GT count for ``prompt``
    prompt: str = "object"
    split: str = ""  # "" = unknown / not split by the dataset
    tier: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def subsample_frames(
    frames: list[Path], max_frames: int | None, stride: int | None = None
) -> list[Path]:
    """Deterministic frame subsampling for videos: every ``stride``-th frame, then evenly spaced
    down to ``max_frames`` (endpoints kept so the subset spans the whole clip)."""
    out = list(frames)
    if stride and stride > 1:
        out = out[::stride]
    if max_frames and 0 < max_frames < len(out):
        idx = np.unique(np.linspace(0, len(out) - 1, max_frames).round().astype(int))
        out = [out[i] for i in idx]
    return out


def list_frame_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def to_sweep_scenes(
    scenes: list[BenchScene], out_dir: str | Path, split: str | None = None
) -> list[dict[str, Any]]:
    """Materialise scenes as ``out_dir/<id>/images/*`` (symlinks, no copies) plus ``gt.json`` and
    return them in ``bench.sweep.discover_scenes`` format. ``split`` keeps only that split
    (scenes with an unknown split are kept)."""
    out = Path(out_dir)
    res: list[dict[str, Any]] = []
    for s in scenes:
        if split and s.split and s.split != split:
            continue
        if not s.frames:
            continue
        d = out / s.scene_id
        img = d / "images"
        img.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(s.frames):
            link = img / f"{i:05d}{f.suffix.lower()}"
            if not link.exists():
                os.symlink(Path(f).resolve(), link)
        gt = {"count": s.count, "prompt": s.prompt, "tier": s.tier, **s.extra}
        (d / "gt.json").write_text(json.dumps(gt, indent=2))
        res.append({"id": s.scene_id, "dir": d, "tier": s.tier, "gt": gt})
    return res


def to_gt_doc(scenes: list[BenchScene], split: str = "test") -> dict[str, Any]:
    """GT document in the ``bench.evaluate.load_gt`` format."""
    return {
        "split": split,
        "scenes": {
            s.scene_id: {"count": s.count} for s in scenes if not s.split or s.split == split
        },
    }
