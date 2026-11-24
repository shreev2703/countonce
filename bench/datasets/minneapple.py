"""MinneApple loader (PLAN §9.1): apple detection / segmentation and counting patches.

Single images with instance masks, no cross-view identity: per-image counting for detector
sanity only. The count of an image is the number of instances in its mask. Layout lives in
``_read_layout`` (``scripts/download_minneapple.sh``).

LICENSE (docs/DATASETS.md): UMN Digital Conservancy Terms of Use; CC BY-NC-SA 3.0 US per a
third-party site (unconfirmed). Treated as non-commercial research only. TODO: verify.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bench.datasets.crop_common import CropImage, check_root, finish, instance_centroids

log = logging.getLogger(__name__)

LICENSE = "UMN Conservancy ToU; CC BY-NC-SA 3.0 US unconfirmed; treat as non-commercial"
SUBSETS = ("detection", "counting")
SPLITS = ("train", "test")
PROMPT = "apple"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _read_layout(root: Path, subset: str) -> list[CropImage]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS): the only function that knows file names.

    Assumed layout (one archive per subset in the real release):
        root/detection/<split>/images/<stem>.png
        root/detection/<split>/masks/<stem>.png    instance-id mask (0 = background)
        root/counting/<split>/images/<stem>.jpg    counting patches
        root/counting/<split>/masks/<stem>.png     instance-id mask per patch
    Images without a mask (the hidden test labels) are skipped.
    TODO: verify directory names, extensions, mask encoding and the split names.
    """
    out: list[CropImage] = []
    for split in SPLITS:
        img_dir = root / subset / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            mask = root / subset / split / "masks" / f"{img.stem}.png"
            if not mask.exists():
                log.info("minneapple: no mask for %s, skipped", img.name)
                continue
            out.append(
                CropImage(
                    name=f"{subset}/{img.name}",
                    path=img,
                    class_name=PROMPT,
                    points_xy=instance_centroids(mask),
                    boxes_xyxy=np.zeros((0, 4)),
                    split=split,
                    dataset="minneapple",
                    mask_path=mask,
                )
            )
    if not out:
        raise FileNotFoundError(f"no labelled images under {root / subset} (layout TODO: verify)")
    return out


def load_minneapple(
    root: str | Path,
    split: str | None = None,
    limit: int | None = None,
    *,
    subset: str = "detection",
    validate: bool = True,
    quarantine_path: str | Path | None = None,
) -> list[CropImage]:
    """Load a downloaded MinneApple directory (``subset``: detection images or counting patches)."""
    root = Path(root)
    check_root(root, "download_minneapple.sh")
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return finish("minneapple", _read_layout(root, subset), split, limit, validate, quarantine_path)
