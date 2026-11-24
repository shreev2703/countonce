"""On-disk reconstruction cache keyed by input hash + model version (PLAN section 4 rule 4)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from countonce.recon.base import Reconstructor
from countonce.types import Frame, Recon

_ARRAYS = (
    "intrinsics", "extrinsics", "depth", "depth_conf", "world_points", "world_points_conf",
)  # fmt: skip
_SCALARS = ("scene_scale", "runtime_s", "peak_vram_mb")


def input_hash(frames: list[Frame]) -> str:
    """Hash of the ordered frame contents (view order matters to a reconstruction)."""
    h = hashlib.sha256()
    for f in frames:
        h.update(f.sha256.encode())
        h.update(b"\0")
    return h.hexdigest()


def recon_cache_key(
    frames: list[Frame], backbone: str, version: str, config: dict[str, Any]
) -> str:
    """Stable key: input hash + backbone + model version + settings."""
    payload = json.dumps(
        {"input": input_hash(frames), "backbone": backbone, "version": version, "config": config},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ReconCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str) -> Recon | None:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                kwargs: dict[str, Any] = {name: z[name] for name in _ARRAYS}
                kwargs.update({name: float(z[name]) for name in _SCALARS})
                return Recon(backbone=str(z["backbone"]), **kwargs)
        except (OSError, ValueError, KeyError):
            return None  # corrupt entry: treat as a miss and recompute

    def put(self, key: str, recon: Recon) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {name: getattr(recon, name) for name in _ARRAYS}
        data.update({name: np.float64(getattr(recon, name)) for name in _SCALARS})
        data["backbone"] = np.array(recon.backbone)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **data)
        os.replace(tmp, self.path(key))  # atomic: a crash never leaves a half-written entry
        return self.path(key)


@dataclass
class CachedRecon:
    recon: Recon
    key: str  # goes into CountResult.recon_cache_key
    hit: bool  # goes into CountResult.recon_cache_hit


class CachedReconstructor:
    """Wraps any ``Reconstructor`` with the on-disk cache."""

    def __init__(self, inner: Reconstructor, cache: ReconCache, version: str) -> None:
        self.inner = inner
        self.cache = cache
        self.version = version
        self.name = inner.name

    def key_for(self, frames: list[Frame]) -> str:
        return recon_cache_key(frames, self.inner.name, self.version, self.inner.config())

    def reconstruct_cached(self, frames: list[Frame]) -> CachedRecon:
        key = self.key_for(frames)
        cached = self.cache.get(key)
        if cached is not None:
            return CachedRecon(cached, key, True)
        recon = self.inner.reconstruct(frames)
        self.cache.put(key, recon)
        return CachedRecon(recon, key, False)

    def reconstruct(self, frames: list[Frame]) -> Recon:
        return self.reconstruct_cached(frames).recon

    def config(self) -> dict[str, Any]:
        return self.inner.config()


