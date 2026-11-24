"""Pick a backend from config and environment (``COUNTONCE_STORAGE=local|s3``)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from countonce.storage.base import BlobStore
from countonce.storage.local import LocalBlobStore


def open_blobstore(
    kind: str | None = None,
    local_root: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> BlobStore:
    """``kind`` (an explicit config/flag value) beats ``COUNTONCE_STORAGE``; default ``local``.

    local: root is ``local_root``, else ``COUNTONCE_STORAGE_DIR``, else ``./storage``; the URL
    signing secret is ``COUNTONCE_STORAGE_SECRET`` (random per process if unset).
    s3: ``COUNTONCE_S3_BUCKET`` is required; ``COUNTONCE_S3_ENDPOINT_URL`` points at R2 or MinIO;
    ``COUNTONCE_S3_ACCESS_KEY_ID`` / ``COUNTONCE_S3_SECRET_ACCESS_KEY`` / ``COUNTONCE_S3_REGION``
    / ``COUNTONCE_S3_PREFIX`` are optional (boto3's default credential chain otherwise).
    """
    e = os.environ if env is None else env
    kind = (kind or e.get("COUNTONCE_STORAGE") or "local").lower()
    if kind == "local":
        root = local_root or e.get("COUNTONCE_STORAGE_DIR") or "storage"
        return LocalBlobStore(root, secret=e.get("COUNTONCE_STORAGE_SECRET") or None)
    if kind != "s3":
        raise ValueError(f"unknown storage backend {kind!r}; expected local or s3")
    bucket = e.get("COUNTONCE_S3_BUCKET")
    if not bucket:
        raise ValueError("COUNTONCE_STORAGE=s3 needs COUNTONCE_S3_BUCKET")
    from countonce.storage.s3 import S3BlobStore

    return S3BlobStore(
        bucket,
        endpoint_url=e.get("COUNTONCE_S3_ENDPOINT_URL") or None,
        region=e.get("COUNTONCE_S3_REGION") or None,
        access_key_id=e.get("COUNTONCE_S3_ACCESS_KEY_ID") or None,
        secret_access_key=e.get("COUNTONCE_S3_SECRET_ACCESS_KEY") or None,
        prefix=e.get("COUNTONCE_S3_PREFIX", ""),
    )


def cache_kwargs(sub: str, *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Extra ``ReconCache`` / ``DetectionCache`` keyword args for the configured backend.

    Empty for ``local`` (the cache then lives in its own ``cache_dir``, as before). For ``s3``
    the cache shares the bucket under ``cache/<sub>/``.
    """
    e = os.environ if env is None else env
    if (e.get("COUNTONCE_STORAGE") or "local").lower() != "s3":
        return {}
    return {"blobs": open_blobstore("s3", env=e), "prefix": f"cache/{sub}"}
