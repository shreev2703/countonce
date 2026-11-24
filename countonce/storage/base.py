"""``BlobStore``: the object-storage seam (PLAN 14.5 "R2 (or S3), signed URLs only", 13 item 6b).

Keys are ``/``-separated relative names (``uploads/files/<id>/a.jpg``). A backend is either a
directory (``LocalBlobStore``) or an S3-compatible bucket (``S3BlobStore``, which also covers
Cloudflare R2 through an endpoint URL). Both pass one contract suite in ``tests/test_storage.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from countonce.errors import CountOnceError, ErrorCode


class BlobNotFound(CountOnceError):
    """``get`` of a key that does not exist."""

    def __init__(self, key: str) -> None:
        super().__init__(f"no such blob: {key!r}", code=ErrorCode.ERR_INVALID_INPUT)
        self.key = key


class SignedURLError(CountOnceError):
    """A signed URL is expired, tampered with, or for another method."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.ERR_INVALID_INPUT)


@dataclass(frozen=True)
class BlobInfo:
    key: str
    size: int
    modified: float  # unix seconds


@runtime_checkable
class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> bool:
        """Remove ``key``; True if it existed."""
        ...

    def presign_get(self, key: str, ttl_s: float = 900.0) -> str: ...
    def presign_put(self, key: str, ttl_s: float = 900.0) -> str: ...
    def list_prefix(self, prefix: str = "") -> list[BlobInfo]:
        """Blobs whose key starts with ``prefix``, sorted by key."""
        ...


def check_key(key: str) -> str:
    """Reject keys that could escape a root or are empty (applies to every backend)."""
    parts = key.split("/")
    if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts) or "\\" in key:
        raise ValueError(f"bad blob key {key!r}")
    return key


def delete_prefix(store: BlobStore, prefix: str) -> int:
    """Delete every blob under ``prefix``; returns how many."""
    return sum(1 for b in store.list_prefix(prefix) if store.delete(b.key))


def expire_prefix(
    store: BlobStore, prefix: str, ttl_s: float, clock: Callable[[], float] = time.time
) -> int:
    """Delete blobs under ``prefix`` older than ``ttl_s`` (cache TTL, PLAN 13 item 6b)."""
    cutoff = clock() - ttl_s
    return sum(1 for b in store.list_prefix(prefix) if b.modified <= cutoff and store.delete(b.key))
