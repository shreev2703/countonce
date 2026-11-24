"""Object storage abstraction: ``BlobStore`` with local-disk and S3/R2 backends."""

from countonce.storage.base import (
    BlobInfo,
    BlobNotFound,
    BlobStore,
    SignedURLError,
    delete_prefix,
    expire_prefix,
)
from countonce.storage.factory import open_blobstore
from countonce.storage.local import LocalBlobStore

__all__ = [
    "BlobInfo",
    "BlobNotFound",
    "BlobStore",
    "LocalBlobStore",
    "SignedURLError",
    "delete_prefix",
    "expire_prefix",
    "open_blobstore",
]
