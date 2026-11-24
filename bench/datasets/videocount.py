"""VideoCount loader (CountVid benchmark, PLAN §9.1, Phase 5 "VideoCount subset vs CountVid").

Source: github.com/niki-amini-naieni/CountVid. TAO / MOT20 / penguin / crystal videos. The release
layout was not inspected (no public file tree was checked), so it is entirely assumed and isolated
in ``_read_layout``. Source-video licenses (TAO, MOT20 research-only) bind derived frames.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.datasets.common import BenchScene, list_frame_files, subsample_frames


def _read_layout(root: Path, frames_dir: Path | None) -> list[BenchScene]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS): the only function that knows file names.

    Assumed layout:
        root/annotations.json         {"<video>": {"count": int, "prompt": str, "source": str}}
            TODO: verify file name, schema, and whether count is per-video or per-prompt
        root/frames/<video>/*.jpg     pre-extracted frames (TODO: verify)
        root/videos/<video>.mp4       otherwise; frames are extracted on demand into ``frames_dir``
            TODO: verify container/extension and fps
    """
    ann_file = root / "annotations.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"{ann_file} missing (layout TODO: verify)")
    ann = json.loads(ann_file.read_text())
    scenes: list[BenchScene] = []
    for video, a in sorted(ann.items()):
        pre = root / "frames" / video
        if pre.is_dir():
            frames = list_frame_files(pre)
        elif (root / "videos" / f"{video}.mp4").is_file():
            if frames_dir is None:
                raise ValueError("videos need frames_dir for frame extraction")
            from countonce.ingest.frames import extract_frames

            frames = extract_frames(root / "videos" / f"{video}.mp4", frames_dir / video, fps=2.0)
        else:
            continue
        scenes.append(
            BenchScene(
                scene_id=f"videocount_{video}",
                frames=frames,
                count=int(a["count"]),
                prompt=str(a.get("prompt", "object")),
                extra={"dataset": "videocount", "source": a.get("source", "")},
            )
        )
    return scenes


def load_videocount(
    root: str | Path,
    *,
    max_frames: int | None = 20,
    stride: int | None = None,
    frames_dir: str | Path | None = None,
) -> list[BenchScene]:
    """Load a downloaded VideoCount directory, subsampling each video's frames."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/download_videocount.sh first")
    scenes = _read_layout(root, Path(frames_dir) if frames_dir else None)
    for s in scenes:
        s.frames = subsample_frames(s.frames, max_frames, stride)
    return scenes
