"""Fuji-SfM loader (PLAN §9.1): images, GT apple positions, optional cameras.

The real file layout is UNVERIFIED (docs/DATASETS.md: "exact file layout and camera/pose
convention TODO: verify after download"). Everything layout-specific lives in ``_read_layout``;
fix only that function after the first download. The rest of this module works on ``FujiScene``.

Open format TODOs (from docs/DATASETS.md):
- TODO: verify file layout of Zenodo 10.5281/zenodo.3712808 (and what sibling 3712807 holds).
- TODO: verify whether per-image cameras/poses ship; if not, recover them with COLMAP/VGGT and
  align to the GT frame with ``countonce.recon.align`` (Sim(3)), see ``align_recon_to_gt``.
- TODO: verify GT unit/axis convention and whether the 1,455 apples are given per tree.
- TODO: read LICENSE.txt before any redistribution (conflicting licenses; treat as research-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from countonce.recon.align import Sim3, align_by_camera_centers, transform_extrinsics


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
    mask_paths: dict[int, Path] = field(default_factory=dict)  # 288 images have instance masks


def _read_layout(root: Path) -> FujiScene:
    """LAYOUT-SPECIFIC (UNVERIFIED GUESS): the only function that knows file names.

    Assumed layout, to be replaced after inspecting the download:
        root/images/*.jpg           582 RGB frames
        root/masks/<image stem>.png instance masks (288 images)
        root/gt_apples.npy          (1455, 3) apple positions, global frame
        root/cameras.npz            optional: K (N,3,3), extrinsics (N,3,4), size (N,2)
    TODO: verify every path and the pose convention after download.
    """
    images = sorted((root / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images under {root / 'images'} (layout TODO: verify)")
    gt_file = root / "gt_apples.npy"
    if not gt_file.exists():
        raise FileNotFoundError(f"{gt_file} missing (layout TODO: verify)")
    gt = np.load(gt_file).reshape(-1, 3)
    scene = FujiScene(
        root=root,
        image_paths=images,
        gt_ids=[f"apple_{i:04d}" for i in range(len(gt))],
        gt_xyz=gt,
    )
    for i, p in enumerate(images):
        m = root / "masks" / f"{p.stem}.png"
        if m.exists():
            scene.mask_paths[i] = m
    cam = root / "cameras.npz"
    if cam.exists():
        z = np.load(cam)
        for i in range(len(images)):
            scene.intrinsics[i] = z["K"][i]
            scene.extrinsics[i] = z["extrinsics"][i]
            scene.image_size[i] = (int(z["size"][i][0]), int(z["size"][i][1]))
    return scene


def load_fuji_sfm(root: str | Path) -> FujiScene:
    """Load a downloaded Fuji-SfM directory (see ``scripts/download_fuji_sfm.sh``)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/download_fuji_sfm.sh first")
    return _read_layout(root)


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
