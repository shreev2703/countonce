"""On-disk reconstruction cache keyed by input hash + model version (PLAN section 4 rule 4)."""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from countonce.oom import RungUnavailable
from countonce.recon.base import Reconstructor
from countonce.storage import BlobNotFound, BlobStore, LocalBlobStore, expire_prefix
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
    """Recon entries as ``<prefix>/<key>.npz`` blobs. ``ReconCache(dir)`` is the on-disk cache;
    pass ``blobs`` (e.g. an ``S3BlobStore``) to keep entries in object storage instead.
    ``ttl_s`` makes ``expire()`` drop entries older than that (PLAN 13 item 6b)."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        blobs: BlobStore | None = None,
        prefix: str = "",
        ttl_s: float | None = None,
    ) -> None:
        if blobs is None:
            if root is None:
                raise ValueError("ReconCache needs a root directory or a blob store")
            blobs = LocalBlobStore(root)
        self.root = Path(root) if root is not None else None
        self.blobs = blobs
        self.prefix = prefix.strip("/")
        self.ttl_s = ttl_s

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}.npz" if self.prefix else f"{key}.npz"

    def path(self, key: str) -> Path:
        """Local file for ``key`` (local backend only)."""
        if not isinstance(self.blobs, LocalBlobStore):
            raise TypeError("path() is only meaningful for the local backend")
        return self.blobs.path(self._key(key))

    def get(self, key: str) -> Recon | None:
        try:
            raw = self.blobs.get(self._key(key))
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                kwargs: dict[str, Any] = {name: z[name] for name in _ARRAYS}
                kwargs.update({name: float(z[name]) for name in _SCALARS})
                return Recon(backbone=str(z["backbone"]), **kwargs)
        except (BlobNotFound, OSError, ValueError, KeyError):
            return None  # missing or corrupt entry: treat as a miss and recompute

    def put(self, key: str, recon: Recon) -> str:
        data: dict[str, Any] = {name: getattr(recon, name) for name in _ARRAYS}
        data.update({name: np.float64(getattr(recon, name)) for name in _SCALARS})
        data["backbone"] = np.array(recon.backbone)
        buf = io.BytesIO()
        np.savez_compressed(buf, **data)
        self.blobs.put(self._key(key), buf.getvalue())  # atomic per backend
        return self._key(key)

    def expire(self, now: float | None = None) -> int:
        """Delete entries older than ``ttl_s``; returns how many (0 if no TTL is set)."""
        if self.ttl_s is None:
            return 0
        return expire_prefix(
            self.blobs, self.prefix + "/" if self.prefix else "", self.ttl_s,
            (lambda: now) if now is not None else time.time,
        )  # fmt: skip


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

    def with_resolution_scale(self, scale: float) -> CachedReconstructor:
        """Same cache over a lower-tier inner; the tier is in ``config()`` so keys differ."""
        fn = getattr(self.inner, "with_resolution_scale", None)
        if fn is None:
            raise RungUnavailable("inner reconstructor has no resolution control")
        return CachedReconstructor(fn(scale), self.cache, self.version)

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
