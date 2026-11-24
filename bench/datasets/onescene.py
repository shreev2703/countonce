"""OneScene-Bench loader (arXiv 2610.38926; PLAN §9.1, Phase 2.5 "OneScene-Bench evaluation").

The counting part draws on DL3DV-10K scenes. Release location, license and layout are all
unverified (docs/DATASETS.md), so the layout is entirely assumed and isolated in ``_read_layout``.
Multi-view scenes are unordered image sets, not videos, so no frame subsampling by default.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.datasets.common import BenchScene, list_frame_files, subsample_frames


def _read_layout(root: Path) -> list[BenchScene]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS): the only function that knows file names.

    Assumed layout:
        root/counting.json   {"scenes": [{"scene_id", "category", "count", "split"?}]}
            TODO: verify file name and keys (answers may be free-text VLM answers)
        root/scenes/<scene_id>/images/*.jpg   (TODO: verify; DL3DV frames may be at another path)
    """
    ann_file = root / "counting.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"{ann_file} missing (layout TODO: verify)")
    scenes: list[BenchScene] = []
    for e in json.loads(ann_file.read_text())["scenes"]:
        d = root / "scenes" / str(e["scene_id"]) / "images"
        if not d.is_dir():
            continue
        scenes.append(
            BenchScene(
                scene_id=f"onescene_{e['scene_id']}",
                frames=list_frame_files(d),
                count=int(e["count"]),
                prompt=str(e.get("category", "object")),
                split=str(e.get("split", "")),
                extra={"dataset": "onescene"},
            )
        )
    return scenes


def load_onescene(
    root: str | Path,
    *,
    max_frames: int | None = None,
    stride: int | None = None,
    validate: bool = True,
    quarantine_path: str | Path | None = None,
) -> list[BenchScene]:
    """Load a downloaded OneScene-Bench directory (no subsampling unless asked)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/download_onescene.sh first")
    scenes = _read_layout(root)
    if validate:  # before subsampling, so count/frame mismatches see every frame
        from bench.datasets.validate import filter_scenes

        scenes, _ = filter_scenes("onescene", scenes, quarantine_path)
    for s in scenes:
        s.frames = subsample_frames(s.frames, max_frames, stride)
    return scenes
