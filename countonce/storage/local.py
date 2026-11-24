"""``LocalBlobStore``: blobs as files under a root, URLs signed with HMAC-SHA256 and an expiry.

``presign_*`` return relative URLs ``<base_url>/<key>?method=..&expires=..&sig=..``.
An HTTP layer calls ``verify(method, key, expires, sig)``; it raises ``SignedURLError`` once the
expiry has passed or if the signature does not match (any change to key, method or expiry).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from countonce.storage.base import BlobInfo, BlobNotFound, SignedURLError, check_key


class LocalBlobStore:
    def __init__(
        self,
        root: str | Path,
        *,
        secret: bytes | str | None = None,
        base_url: str = "/blobs",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.base_url = base_url.rstrip("/")
        self.clock = clock
        # no configured secret: a per-process random one, so URLs die with the process
        raw = secret if secret is not None else os.urandom(32)
        self._secret = raw.encode() if isinstance(raw, str) else raw

    # -- paths ----------------------------------------------------------------------------
    def path(self, key: str) -> Path:
        """File for ``key``. An absolute path inside the root is also accepted, so a stored
        legacy ``job_files.key`` still resolves."""
        if os.path.isabs(key):
            rel = Path(key).resolve().relative_to(self.root.resolve())  # ValueError if outside
            key = rel.as_posix()
        return self.root / check_key(key)

    def key_of(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    # -- BlobStore ------------------------------------------------------------------------
    def put(self, key: str, data: bytes) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, p)  # atomic: a reader sees the old or the new blob, never half

    def get(self, key: str) -> bytes:
        try:
            return self.path(key).read_bytes()
        except FileNotFoundError:
            raise BlobNotFound(key) from None

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def delete(self, key: str) -> bool:
        try:
            self.path(key).unlink()
            return True
        except FileNotFoundError:
            return False

    def list_prefix(self, prefix: str = "") -> list[BlobInfo]:
        out: list[BlobInfo] = []
        if not self.root.is_dir():
            return out
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix == ".tmp":
                continue
            key = p.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                st = p.stat()
                out.append(BlobInfo(key, st.st_size, st.st_mtime))
        return sorted(out, key=lambda b: b.key)

    # -- signed URLs ----------------------------------------------------------------------
    def _sig(self, method: str, key: str, expires: int) -> str:
        msg = f"{method}\n{key}\n{expires}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def _presign(self, method: str, key: str, ttl_s: float) -> str:
        check_key(key)
        expires = int(self.clock() + ttl_s)
        return (
            f"{self.base_url}/{quote(key)}?method={method}&expires={expires}"
            f"&sig={self._sig(method, key, expires)}"
        )

    def presign_get(self, key: str, ttl_s: float = 900.0) -> str:
        return self._presign("GET", key, ttl_s)

    def presign_put(self, key: str, ttl_s: float = 900.0) -> str:
        return self._presign("PUT", key, ttl_s)

    def verify(self, method: str, key: str, expires: int, sig: str) -> None:
        """Raise ``SignedURLError`` unless the signature matches and has not expired."""
        if not hmac.compare_digest(self._sig(method, key, int(expires)), sig):
            raise SignedURLError("bad signature")
        if self.clock() > int(expires):
            raise SignedURLError("signed URL expired")

    def verify_url(self, method: str, url: str) -> str:
        """Check a URL from ``presign_*`` for ``method``; returns its key."""
        parts = urlsplit(url)
        q = parse_qs(parts.query)
        try:
            key = parts.path[len(self.base_url) :].lstrip("/")
            key = unquote(key)
            if q["method"][0] != method:
                raise SignedURLError(f"URL was signed for {q['method'][0]}, not {method}")
            self.verify(method, key, int(q["expires"][0]), q["sig"][0])
        except (KeyError, ValueError, IndexError):
            raise SignedURLError("malformed signed URL") from None
        return key
