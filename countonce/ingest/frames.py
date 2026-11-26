# Ported from vggt-sparse-recon: vggt_bench/io/frames.py. Adapted for countonce.
"""Choosing input frames for a reconstruction.

Phase 1's scripts carry their own copy of this logic and are left alone
deliberately. New scripts import it from here so the Phase 3 sweep and the
Phase 4 captures do not grow a third and fourth variant: both methods have to
see the *same* frame subset for any side-by-side number to mean anything.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

from countonce.types import GridMap, PixelMap

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(scene_dir: Path, subdir: str = "images") -> list[Path]:
    """Every image in a scene's images folder (default images/), in filename order."""
    return sorted(p for p in (scene_dir / subdir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def select_frames(scene_dir: Path, num_frames: int) -> list[Path]:
    """An evenly spaced subset of a scene's frames; all of them when num_frames <= 0.

    Evenly spaced rather than the first N, so a subset still spans the whole
    camera trajectory. That is what makes the view count, and not the amount of
    the scene covered, the variable under test.
    """
    return sample_frames(list_images(scene_dir), num_frames, None)


def sample_frames(paths: list[Path], n: int, seed: int | None) -> list[Path]:
    """Pick n frames from an ordered list, identically for every method.

    seed=None gives the evenly spaced, wide-baseline subset. An int gives a
    reproducible random draw without replacement, returned in original order.
    n >= len(paths) returns everything: a scene with fewer images than n is
    clamped rather than padded with duplicates.
    """
    if n <= 0 or n >= len(paths):
        return list(paths)
    if seed is None:
        idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(len(paths), n, replace=False))
    return [paths[i] for i in idx]


# --- Phase 4: phone-video captures ----------------------------------------------------------

VGGT_SIZE = 518
VGGT_PATCH = 14


def _scaled_height(width: int, height: int, target: int = VGGT_SIZE) -> int:
    """Height after scaling to `target` wide, rounded to the patch grid, before any crop."""
    return round(height * (target / width) / VGGT_PATCH) * VGGT_PATCH


def scaled_target(scale: float, base: int = VGGT_SIZE) -> int:
    """Input tier for a resolution ``scale``: ``base * scale`` snapped to the 14 px patch grid.

    Never below one patch; ``scale >= 1`` returns ``base`` (a tier is only ever lowered).
    """
    if scale >= 1.0:
        return base
    if scale <= 0:
        raise ValueError(f"resolution scale must be positive, got {scale}")
    return max(int(round(base * scale / VGGT_PATCH)), 1) * VGGT_PATCH


def vggt_grid_size(width: int, height: int, target: int = VGGT_SIZE) -> tuple[int, int]:
    """(W, H) that VGGT's load_and_preprocess_images(mode="crop") produces for an image.

    Width becomes `target`; height keeps the aspect ratio rounded to the 14 px patch grid, then
    is centre-cropped to at most `target`. Saving frames at this size makes VGGT's own
    preprocessing an identity, so its intrinsics apply to the saved files directly.
    """
    return target, min(_scaled_height(width, height, target), target)


def extract_frames(video: Path, out_dir: Path, fps: float = 2.0) -> list[Path]:
    """Sample a video at `fps` into out_dir/frame_000001.jpg, ... and return them in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.jpg"):  # a re-run must not mix in older frames
        stale.unlink()
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video), "-map_metadata", "-1",
        "-vf", f"fps={fps}", "-q:v", "2", str(out_dir / "frame_%06d.jpg"),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if proc.returncode != 0 or not frames:
        raise RuntimeError(f"ffmpeg failed on {video}: {proc.stderr.strip() or 'no frames'}")
    return frames


def laplacian_variance(path: Path) -> float:
    """Sharpness score: variance of the Laplacian of the grayscale image (low = blurry)."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float64)
    return float(laplace(gray).var())


def drop_blurry(paths: list[Path], threshold: float) -> tuple[list[Path], dict[str, float]]:
    """Keep frames whose sharpness is at least `threshold`; also return every frame's score."""
    scores = {p.name: laplacian_variance(p) for p in paths}
    return [p for p in paths if scores[p.name] >= threshold], scores


def backbone_grid_map(
    width: int, height: int, mode: str = "crop", target: int = VGGT_SIZE
) -> GridMap:
    """Exact map from an original image onto VGGT's input grid.

    Mirrors `vggt.utils.load_fn.load_and_preprocess_images` (facebookresearch/vggt):
      crop: width -> `target`, height -> round(h * target / w / 14) * 14, then a vertical
            centre crop of `(new_h - target) // 2` rows when new_h > target.
      pad:  the longer side -> `target`, the other rounded to a multiple of 14, then centred
            in a `target x target` canvas with `(target - new) // 2` rows/cols of padding.
    The two axes get slightly different scales because of the rounding to the patch grid.
    """
    if mode == "crop":
        new_w, new_h = target, max(_scaled_height(width, height, target), VGGT_PATCH)
        top = (new_h - target) // 2 if new_h > target else 0
        return GridMap(new_w / width, new_h / height, 0.0, -float(top), new_w, min(new_h, target))
    if mode == "pad":
        if width >= height:
            new_w, new_h = target, max(_scaled_height(width, height, target), VGGT_PATCH)
        else:
            new_h = target
            new_w = max(round(width * (target / height) / VGGT_PATCH) * VGGT_PATCH, VGGT_PATCH)
        pad_top, pad_left = (target - new_h) // 2, (target - new_w) // 2
        return GridMap(
            new_w / width, new_h / height, float(pad_left), float(pad_top), target, target
        )
    raise ValueError(f"unknown preprocessing mode {mode!r} (expected 'crop' or 'pad')")


def grid_pixel_map(width: int, height: int) -> PixelMap:
    """Map from a frame's `vggt_grid_size` pixels back to the original (width, height) image."""
    gm = backbone_grid_map(width, height, "crop")
    return PixelMap(
        sx=1 / gm.sx, sy=1 / gm.sy, ox=-gm.ox / gm.sx, oy=-gm.oy / gm.sy,
        orig_width=width, orig_height=height,
    )  # fmt: skip


def resize_to_grid(paths: list[Path], out_dir: Path) -> list[Path]:
    """Write each frame as a PNG at its vggt_grid_size, centre-cropping tall frames."""
    return resize_to_grid_mapped(paths, out_dir)[0]


def resize_to_grid_mapped(paths: list[Path], out_dir: Path) -> tuple[list[Path], list[PixelMap]]:
    """`resize_to_grid` plus, per frame, the `PixelMap` back to its original pixels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out, maps = [], []
    for p in paths:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            maps.append(grid_pixel_map(*rgb.size))
            w, h = vggt_grid_size(*rgb.size)
            new_h = _scaled_height(rgb.width, rgb.height)
            rgb = rgb.resize((w, new_h), Image.Resampling.BICUBIC)
            top = (new_h - h) // 2  # 0 unless a tall frame needs its centre cropped
            rgb = rgb.crop((0, top, w, top + h))
            dest = out_dir / f"{p.stem}.png"
            rgb.save(dest)
        out.append(dest)
    return out, maps
