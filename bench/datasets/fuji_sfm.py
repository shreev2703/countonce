"""Fuji-SfM loader (PLAN §9.1): SfM images, per-apple 3D GT, per-tree grouping.

Verified against the real Zenodo 10.5281/zenodo.3712808 download (docs/DATASETS.md, Fuji-SfM).
License: CC BY-NC-SA 4.0 (``LICENSE.txt``, research and education only, no commercial use). Do not
redistribute the images or any GT derived from them, and do not upload them to a HF dataset.

Real layout (``Fuji-SfM_dataset/``):
    LICENSE.txt, readme.txt
    1-Mask-set/{training,validation}_images_and_annotations/  288 image crops (*.jpg, per-image
        *.csv) plus ``training_set_masks.json`` / ``validation_set_masks.json``; ``raw_images/``
        holds the 12 full frames the crops were cut from. These crops are NOT the SfM frames.
    2-SfM-set/raw_images_{east,west}_side/*.JPG               291 + 291 frames, 5184x3456, no poses
    3-3D_data/Fuji_apple_trees_point_cloud.txt                scene cloud (not read here)
    3-3D_data/GroundTruth-3D_apples_locations/{training_set,test_set}/TreeNN-annotations_NNNNNN.txt
        one file per apple: whitespace-separated xyz rows (its fruit points), metres, in global
        coordinates shifted by T=[-336300, -4623400, 0] (``3-3D_data/readme.txt``). NNNNNN restarts
        at 000000 for every tree, so an apple id is ``TreeNN/NNNNNN``.
    1,455 apples: training_set Tree09-11 (433), test_set Tree01-08 (1,022).

Camera poses are not shipped, so ``intrinsics``/``extrinsics`` stay empty: recover them with
COLMAP/VGGT and align to the GT frame with ``countonce.recon.align`` (Sim(3)), see
``align_recon_to_gt``. No file maps SfM frames to trees; the frames are one sequence per side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from countonce.recon.align import Sim3, align_by_camera_centers, transform_extrinsics

GT_SHIFT = (-336300.0, -4623400.0, 0.0)  # T from 3-3D_data/readme.txt (saved = global + T)
_GT_DIR = Path("3-3D_data") / "GroundTruth-3D_apples_locations"
_GT_SPLITS = {"training_set": "training", "test_set": "test"}
_GT_NAME = re.compile(r"^(Tree\d+)-annotations_(\d+)\.txt$")


@dataclass
class FujiScene:
    root: Path
    image_paths: list[Path]
    gt_ids: list[str]
    gt_xyz: np.ndarray  # (M,3) in the GT/global frame
    # Optional dataset cameras, keyed by image index; empty if poses must be recovered.
    intrinsics: dict[int, np.ndarray] = field(default_factory=dict)
    extrinsics: dict[int, np.ndarray] = field(default_factory=dict)  # world-to-camera (3,4)
    image_size: dict[int, tuple[int, int]] = field(default_factory=dict)  # (width, height)
    mask_paths: dict[int, Path] = field(default_factory=dict)  # always empty: see mask_images
    # Per-apple GT detail, aligned with gt_ids / gt_xyz (the centroid of the apple's points).
    gt_tree: list[str] = field(default_factory=list)  # "Tree01".."Tree11"
    gt_split: list[str] = field(default_factory=list)  # dataset split: "training" | "test"
    gt_extent: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (M,3) bbox size, m
    gt_n_points: list[int] = field(default_factory=list)
    gt_shift: tuple[float, float, float] = GT_SHIFT  # add -shift to recover the global frame
    side_indices: dict[str, list[int]] = field(default_factory=dict)  # "east"|"west" -> image ids
    # The separate 288-image mask set (crops of 12 raw frames, not the SfM frames).
    mask_images: list[Path] = field(default_factory=list)
    mask_annotations: dict[str, Path] = field(default_factory=dict)  # split -> *_masks.json

    @property
    def trees(self) -> dict[str, list[str]]:
        """Dataset split -> sorted tree ids."""
        out: dict[str, set[str]] = {}
        for t, sp in zip(self.gt_tree, self.gt_split, strict=True):
            out.setdefault(sp, set()).add(t)
        return {k: sorted(v) for k, v in out.items()}

    def tree_mask(self, tree: str) -> np.ndarray:
        """Boolean mask over the GT apples of one tree."""
        return np.fromiter((t == tree for t in self.gt_tree), dtype=bool, count=len(self.gt_tree))


def resolve_root(root: Path) -> Path:
    """Accept ``Fuji-SfM_dataset/`` itself or a parent holding ``unpacked/Fuji-SfM_dataset/``."""
    for cand in (root, root / "unpacked" / "Fuji-SfM_dataset", root / "Fuji-SfM_dataset"):
        if (cand / "2-SfM-set").is_dir():
            return cand
    return root


def read_gt_apples(root: Path) -> list[tuple[str, str, str, np.ndarray]]:
    """(apple id, tree, dataset split, (n,3) points) per GT file, sorted by split then path."""
    base = root / _GT_DIR
    out: list[tuple[str, str, str, np.ndarray]] = []
    for sub, split in _GT_SPLITS.items():
        for f in sorted((base / sub).glob("*.txt")):
            m = _GT_NAME.match(f.name)
            if not m:
                raise ValueError(f"unexpected GT file name {f.name!r} in {base / sub}")
            pts = np.loadtxt(f, dtype=np.float64, ndmin=2)
            if pts.shape[1] != 3:
                raise ValueError(f"{f}: expected xyz rows, got shape {pts.shape}")
            out.append((f"{m[1]}/{m[2]}", m[1], split, pts))
    return out


def _read_layout(root: Path) -> FujiScene:
    """LAYOUT-SPECIFIC: the only function that knows file names (see the module docstring)."""
    root = resolve_root(root)
    sfm = root / "2-SfM-set"
    side_indices: dict[str, list[int]] = {}
    images: list[Path] = []
    for side in ("east", "west"):
        files = sorted((sfm / f"raw_images_{side}_side").glob("*.JPG"))
        side_indices[side] = list(range(len(images), len(images) + len(files)))
        images += files
    if not images:
        raise FileNotFoundError(f"no images under {sfm}/raw_images_{{east,west}}_side")
    apples = read_gt_apples(root)
    if not apples:
        raise FileNotFoundError(f"no GT apple files under {root / _GT_DIR}")
    scene = FujiScene(
        root=root,
        image_paths=images,
        gt_ids=[a[0] for a in apples],
        gt_xyz=np.stack([a[3].mean(axis=0) for a in apples]),
        gt_tree=[a[1] for a in apples],
        gt_split=[a[2] for a in apples],
        gt_extent=np.stack([np.ptp(a[3], axis=0) for a in apples]),
        gt_n_points=[len(a[3]) for a in apples],
        side_indices=side_indices,
    )
    mask_dir = root / "1-Mask-set"
    for sub, split in (("training", "train"), ("validation", "val")):
        d = mask_dir / f"{sub}_images_and_annotations"
        scene.mask_images += sorted(d.glob("*.jpg"))
        if (d / f"{sub}_set_masks.json").exists():
            scene.mask_annotations[split] = d / f"{sub}_set_masks.json"
    return scene


def load_fuji_sfm(
    root: str | Path, *, validate: bool = True, quarantine_path: str | Path | None = None
) -> FujiScene:
    """Load a downloaded Fuji-SfM directory (see ``scripts/download_fuji_sfm.sh``). With
    ``validate`` a scene failing the data-quality gates is quarantined and ``DataQualityError``
    is raised (a single scene, so there is nothing to skip to)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/download_fuji_sfm.sh first")
    scene = _read_layout(root)
    if validate:
        from bench.datasets.validate import DataQualityError, filter_scenes, validate_scene

        if not filter_scenes("fuji_sfm", [scene], quarantine_path)[0]:
            raise DataQualityError(validate_scene(scene))
    return scene


def align_recon_to_gt(
    scene: FujiScene, recon_extrinsics: np.ndarray, frame_ids: list[int]
) -> tuple[Sim3, dict[int, np.ndarray]]:
    """Sim(3) from a reconstruction's world frame to the GT frame using camera centres of frames
    that have dataset poses. Returns the transform and the recon extrinsics re-expressed in the
    GT frame (keyed by frame id), ready for ``bench.gt_matching.match_detections_to_gt``.

    ``recon_extrinsics`` is (len(frame_ids),3,4) world-to-camera, row i for ``frame_ids[i]``.
    """
    rows = [i for i, f in enumerate(frame_ids) if f in scene.extrinsics]
    if len(rows) < 3:
        raise ValueError("need >= 3 frames with dataset poses to align to the GT frame")
    gt_ext = np.stack([scene.extrinsics[frame_ids[i]] for i in rows])
    sim = align_by_camera_centers(np.asarray(recon_extrinsics)[rows], gt_ext)
    moved = transform_extrinsics(recon_extrinsics, sim)
    return sim, {f: moved[i] for i, f in enumerate(frame_ids)}
