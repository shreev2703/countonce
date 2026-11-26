"""Give path-based backbones that need one image root (COLMAP) a single directory."""

from __future__ import annotations

import shutil
from pathlib import Path


def single_root(image_paths: list[Path], stage_dir: Path) -> list[Path]:
    """``image_paths`` unchanged if they share a parent, else copies in ``stage_dir``.

    Copies are named ``<index>_<name>`` so that order is kept and equal names cannot collide.
    The pipeline's sanitized frames live in one directory per input, which needs this.
    """
    paths = [Path(p) for p in image_paths]
    if len({p.parent.resolve() for p in paths}) <= 1:
        return paths
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(paths):
        dst = stage_dir / f"{i:05d}_{p.name}"
        shutil.copyfile(p, dst)
        out.append(dst)
    return out
