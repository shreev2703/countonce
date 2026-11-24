"""WGISD loader (PLAN §9.1): Embrapa Wine Grape Instance Segmentation Dataset, single images.

300 images, 4,432 grape-cluster boxes (YOLO text), 137 images with binary masks. Single views
only: no cross-view identity, so this is detector sanity data, not dedup data. Layout lives in
``_read_layout``; fix only that after the first download (``scripts/download_wgisd.sh``).

LICENSE (docs/DATASETS.md): CONFLICTING. Dataset Ninja says CC BY-NC-SA 4.0; the official README
says free use with commercial users contacting Embrapa. Treated as non-commercial research only
until resolved. TODO: verify.
"""

from __future__ import annotations

from pathlib import Path

from bench.datasets.crop_common import (
    CropImage,
    box_centres,
    check_root,
    finish,
    image_size,
    read_yolo,
)

LICENSE = "CONFLICT: CC BY-NC-SA 4.0 (Dataset Ninja) vs Embrapa README; treat as non-commercial"
SPLITS = ("train", "test")
PROMPT = "grape cluster"


def _read_layout(root: Path) -> list[CropImage]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS): the only function that knows file names.

    Assumed layout:
        root/data/<stem>.jpg    image
        root/data/<stem>.txt    YOLO boxes "class cx cy w h" (normalised)
        root/data/<stem>.npz    optional binary masks (137 images)
        root/train.txt, root/test.txt   one image stem per line
    TODO: verify every path, the split-list format (stems vs file names) and mask key/shape.
    """
    data = root / "data"
    if not data.is_dir():
        raise FileNotFoundError(f"{data} missing (layout TODO: verify)")
    out: list[CropImage] = []
    for split in SPLITS:
        lst = root / f"{split}.txt"
        if not lst.exists():
            raise FileNotFoundError(f"{lst} missing (layout TODO: verify)")
        for line in lst.read_text().splitlines():
            stem = Path(line.strip()).stem
            if not stem:
                continue
            img = data / f"{stem}.jpg"
            if not img.exists():
                raise FileNotFoundError(f"{img} listed in {lst.name} but missing")
            boxes = read_yolo(data / f"{stem}.txt", image_size(img))
            mask = data / f"{stem}.npz"
            out.append(
                CropImage(
                    name=f"{stem}.jpg",
                    path=img,
                    class_name=PROMPT,
                    points_xy=box_centres(boxes),
                    boxes_xyxy=boxes,
                    split=split,
                    dataset="wgisd",
                    mask_path=mask if mask.exists() else None,
                )
            )
    return out


def load_wgisd(
    root: str | Path,
    split: str | None = None,
    limit: int | None = None,
    *,
    validate: bool = True,
    quarantine_path: str | Path | None = None,
) -> list[CropImage]:
    """Load a downloaded WGISD directory; optionally one split and the first ``limit`` images."""
    root = Path(root)
    check_root(root, "download_wgisd.sh")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return finish("wgisd", _read_layout(root), split, limit, validate, quarantine_path)
