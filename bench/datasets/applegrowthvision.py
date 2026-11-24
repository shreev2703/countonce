"""AppleGrowthVision loader (PLAN §9.1): orchard images with sparse boxes, plus counted trees.

Two views of one release (``scripts/download_applegrowthvision.sh``):
- ``load_applegrowthvision_images``: the annotated single images (sparse YOLO boxes), a detector
  sanity set. Boxes are sparse, so counts are lower bounds on apples visible; read results so.
- ``load_applegrowthvision``: the 10 trees with a manual apple count, each exposed as a
  multi-view ``BenchScene`` for ``bench.evaluate`` (opt-in dataset ``applegrowthvision``).
Layout lives in the ``_read_*`` functions.

LICENSE (docs/DATASETS.md): original TODO: verify; the HF FiftyOne copy states CC BY 4.0, which
does not prove the original's terms. Treated as non-commercial research only.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.datasets.common import BenchScene, list_frame_files, subsample_frames
from bench.datasets.crop_common import (
    CropImage,
    box_centres,
    check_root,
    finish,
    image_size,
    read_yolo,
)

LICENSE = "original TODO: verify; HF copy CC BY 4.0; treat as non-commercial"
SPLITS = ("train", "val", "test")
PROMPT = "apple"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _read_images(root: Path) -> list[CropImage]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS).

    Assumed layout:
        root/images/<stem>.jpg, root/labels/<stem>.txt   YOLO "class cx cy w h" (normalised)
        root/splits.json   optional {"train": [stems], "val": [...], "test": [...]};
                           without it every image is split "train"
    Only images that have a label file are used (the 1,125 annotated ones).
    TODO: verify directory names, stereo left/right naming, label format and split file.
    """
    img_dir = root / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} missing (layout TODO: verify)")
    split_of: dict[str, str] = {}
    sp = root / "splits.json"
    if sp.exists():
        for split, stems in json.loads(sp.read_text()).items():
            split_of.update({Path(s).stem: split for s in stems})
    out: list[CropImage] = []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        lab = root / "labels" / f"{img.stem}.txt"
        if not lab.exists():
            continue
        boxes = read_yolo(lab, image_size(img))
        out.append(
            CropImage(
                name=img.name,
                path=img,
                class_name=PROMPT,
                points_xy=box_centres(boxes),
                boxes_xyxy=boxes,
                split=split_of.get(img.stem, "train"),
                dataset="applegrowthvision",
            )
        )
    return out


def load_applegrowthvision_images(
    root: str | Path,
    split: str | None = None,
    limit: int | None = None,
    *,
    validate: bool = True,
    quarantine_path: str | Path | None = None,
) -> list[CropImage]:
    """Annotated single images (detector-sanity use)."""
    root = Path(root)
    check_root(root, "download_applegrowthvision.sh")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return finish(
        "applegrowthvision_images", _read_images(root), split, limit, validate, quarantine_path
    )


def _read_trees(root: Path) -> list[BenchScene]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS).

    Assumed layout:
        root/trees/counts.json              {"<tree_id>": <manual apple count>} (10 trees)
        root/trees/<tree_id>/images/*.jpg   the multi-view capture of that tree
    TODO: verify where the 10 counted trees live, how images map to trees, and whether the manual
    count covers all fruit or only the visible side.
    """
    cf = root / "trees" / "counts.json"
    if not cf.exists():
        raise FileNotFoundError(f"{cf} missing (layout TODO: verify)")
    scenes: list[BenchScene] = []
    for tid, count in sorted(json.loads(cf.read_text()).items()):
        d = root / "trees" / str(tid) / "images"
        if not d.is_dir():
            continue
        scenes.append(
            BenchScene(
                scene_id=f"agv_tree_{tid}",
                frames=list_frame_files(d),
                count=int(count),
                prompt=PROMPT,
                extra={"dataset": "applegrowthvision"},
            )
        )
    return scenes


def load_applegrowthvision(
    root: str | Path,
    *,
    max_frames: int | None = None,
    stride: int | None = None,
    validate: bool = True,
    quarantine_path: str | Path | None = None,
) -> list[BenchScene]:
    """The counted trees as multi-view scenes (no subsampling unless asked)."""
    root = Path(root)
    check_root(root, "download_applegrowthvision.sh")
    scenes = _read_trees(root)
    if validate:
        from bench.datasets.validate import filter_scenes

        scenes, _ = filter_scenes("applegrowthvision", scenes, quarantine_path)
    for s in scenes:
        s.frames = subsample_frames(s.frames, max_frames, stride)
    return scenes
