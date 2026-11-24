"""FSC-147 loader (PLAN §9.1): single-image class-agnostic counting, detector sanity only.

Yields one ``FscImage`` per image: path, class name (used as the text prompt), GT points and
exemplar boxes. There is no multi-view identity GT here; this never feeds the dedup benchmark.
Everything layout-specific lives in ``_read_layout``; fix only that function after the first
download (``scripts/download_fsc147.sh``). Licence of the scraped images is unresolved
(docs/DATASETS.md): evaluate only, never redistribute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass
class FscImage:
    name: str  # file name, e.g. "2.jpg"
    path: Path
    class_name: str  # the prompt
    points_xy: np.ndarray  # (N,2) GT points, pixels of the 384-px image
    exemplar_boxes_xyxy: np.ndarray  # (K,4), K is normally 3
    split: str

    @property
    def count(self) -> int:
        return int(len(self.points_xy))


def _read_layout(root: Path) -> tuple[list[FscImage], dict[str, list[str]]]:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS): the only function that knows file names.

    Assumed layout, to be replaced after inspecting the download:
        root/images_384_VarV2/<name>.jpg
        root/annotation_FSC147_384.json   {name: {"points": [[x, y], ...],
                                           "box_examples_coordinates": [[[x,y]*4], ...]}}
        root/Train_Test_Val_FSC_147.json  {"train": [names], "val": [names], "test": [names]}
        root/ImageClasses_FSC147.txt      one "<name>.jpg<TAB>class name" per line
    TODO: verify every path, key, the corner order of exemplar boxes and the (x, y) point order.
    """
    ann_f = root / "annotation_FSC147_384.json"
    split_f = root / "Train_Test_Val_FSC_147.json"
    cls_f = root / "ImageClasses_FSC147.txt"
    img_dir = root / "images_384_VarV2"
    for f in (ann_f, split_f, cls_f):
        if not f.exists():
            raise FileNotFoundError(f"{f} missing (layout TODO: verify)")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} missing (layout TODO: verify)")
    ann = json.loads(ann_f.read_text())
    splits = json.loads(split_f.read_text())
    classes: dict[str, str] = {}
    for line in cls_f.read_text().splitlines():
        if line.strip():
            key, _, cname = line.partition("\t") if "\t" in line else line.partition(" ")
            classes[key.strip()] = cname.strip()
    out: list[FscImage] = []
    for split, names in splits.items():
        for n in names:
            if n not in ann:
                raise KeyError(f"{n} in split {split!r} but not in annotations (TODO: verify)")
            a = ann[n]
            pts = np.asarray(a.get("points", []), dtype=float).reshape(-1, 2)
            corners = np.asarray(a.get("box_examples_coordinates", []), dtype=float).reshape(
                -1, 4, 2
            )
            boxes = (
                np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)
                if len(corners)
                else np.zeros((0, 4))
            )
            out.append(FscImage(n, img_dir / n, classes.get(n, ""), pts, boxes, split))
    return out, splits


def load_fsc147(
    root: str | Path, split: str | None = None, limit: int | None = None
) -> list[FscImage]:
    """Load a downloaded FSC-147 directory; optionally one split and the first ``limit`` images
    (sorted by name for determinism)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/download_fsc147.sh first")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    items, _ = _read_layout(root)
    if split is not None:
        items = [i for i in items if i.split == split]
    items.sort(key=lambda i: i.name)
    return items[:limit] if limit is not None else items
