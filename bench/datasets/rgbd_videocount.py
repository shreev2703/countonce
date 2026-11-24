"""RGBD-VideoCount loader (PLAN §9.1, Phase 5 "vs DG-Net").

Source: HF dataset ``aerospace123/RGBD-VideoCount`` (CC BY 4.0), one 2.7 GB ``RGBD-VideoCount.zip``
(from the repo file tree). The README documents the top-level directory structure only:

    images/  Depth_Data/  object_annotations/  count_annotations/
    dataset_split.json  video_class.txt  exemplars_{train,val,test}.json

Everything below the top level (per-clip sub-folders, JSON schemas) is NOT documented, so it is
assumed and isolated in ``_read_layout``. Fix only that function after the first download.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.datasets.common import BenchScene, list_frame_files, subsample_frames


def _read_layout(root: Path) -> list[BenchScene]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS): the only function that knows file names.

    Assumed layout:
        root/images/<clip>/*.jpg                RGB frames, filename order = time order
            TODO: verify (could be flat images/<clip>_<n>.jpg)
        root/count_annotations/<clip>.json      {"<category>": <int count>, ...}
            TODO: verify schema; could be a list of {"category", "count"}
        root/dataset_split.json                 {"train": [clip, ...], "val": [...], "test": [...]}
            TODO: verify keys and whether entries are clip names or paths
        root/Depth_Data/, object_annotations/, exemplars_*.json   unused by this loader
    One scene per (clip, category); the category is the text prompt.
    """
    img_root = root / "images"
    ann_root = root / "count_annotations"
    if not img_root.is_dir() or not ann_root.is_dir():
        raise FileNotFoundError(
            f"expected {img_root} and {ann_root} (layout TODO: verify; "
            "see scripts/download_rgbd_videocount.sh)"
        )
    split_of: dict[str, str] = {}
    split_file = root / "dataset_split.json"
    if split_file.exists():
        for name, clips in json.loads(split_file.read_text()).items():
            for c in clips:
                split_of[Path(str(c)).stem] = name
    scenes: list[BenchScene] = []
    for ann in sorted(ann_root.glob("*.json")):
        clip = ann.stem
        clip_dir = img_root / clip
        if not clip_dir.is_dir():
            continue
        frames = list_frame_files(clip_dir)
        counts = json.loads(ann.read_text())
        for cat, n in sorted(counts.items()):
            scenes.append(
                BenchScene(
                    scene_id=f"rgbdvc_{clip}_{cat}".replace(" ", "_"),
                    frames=frames,
                    count=int(n),
                    prompt=str(cat),
                    split=split_of.get(clip, ""),
                    extra={"dataset": "rgbd_videocount", "clip": clip},
                )
            )
    return scenes


def load_rgbd_videocount(
    root: str | Path,
    *,
    max_frames: int | None = 20,
    stride: int | None = None,
    validate: bool = True,
    quarantine_path: str | Path | None = None,
) -> list[BenchScene]:
    """Load a downloaded RGBD-VideoCount directory, subsampling each clip's frames."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/download_rgbd_videocount.sh first")
    scenes = _read_layout(root)
    if validate:  # before subsampling, so count/frame mismatches see every frame
        from bench.datasets.validate import filter_scenes

        scenes, _ = filter_scenes("rgbd_videocount", scenes, quarantine_path)
    for s in scenes:
        s.frames = subsample_frames(s.frames, max_frames, stride)
    return scenes
