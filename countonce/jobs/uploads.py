"""Resumable chunked uploads (PLAN 14.8) on the sqlite job store. Stdlib only.

A session is declared up front (filename, total size, whole-file sha256, chunk size). The client
sends numbered chunks, each checked against its own sha256, in any order and any number of times
(a repeat of an already stored chunk is a no-op). ``status`` reports which chunks arrived and the
next missing byte offset, so an interrupted client resumes instead of restarting. ``finalize``
assembles the chunks, verifies the whole-file hash, runs the existing media sanitization and
attaches the clean file to a job.

``UploadTransport`` is the seam: ``LocalTransport`` calls ``UploadService`` in-process; a later
HTTP layer implements the same four methods (and R2 multipart or TUS can replace the chunk store
there, see docs/DEVIATIONS.md). ``upload_file`` is the client loop and only sees the transport.

Tables are created additively (``CREATE TABLE IF NOT EXISTS``) on first use, so an existing job
database upgrades in place. Chunks live under blob prefix ``sessions/<id>/``; the finished,
sanitized file under ``files/<id>/`` (a ``BlobStore``: local disk under ``<root>``, or S3/R2),
which job retention (not session purge) governs.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from countonce.errors import CountOnceError, ErrorCode
from countonce.ingest.sanitize import (
    DEFAULT_LIMITS,
    MediaLimits,
    SanitizeReport,
    check_extension,
    sanitize_image,
    validate_video,
)
from countonce.jobs.store import SqliteJobStore
from countonce.storage import BlobStore, LocalBlobStore, delete_prefix

UPLOAD_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS upload_sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        total_bytes BIGINT NOT NULL,
        sha256 TEXT NOT NULL,
        chunk_size BIGINT NOT NULL,
        n_chunks INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        dir TEXT NOT NULL,
        job_id TEXT,
        created_at DOUBLE PRECISION NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS upload_chunks (
        session_id TEXT NOT NULL REFERENCES upload_sessions (id),
        idx INTEGER NOT NULL,
        size BIGINT NOT NULL,
        sha256 TEXT NOT NULL,
        PRIMARY KEY (session_id, idx)
    )""",
]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class UploadError(CountOnceError):
    """A typed upload failure (``code`` says whether the client should retry or restart)."""


def _err(code: ErrorCode, message: str) -> UploadError:
    return UploadError(message, code=code)


@dataclass(frozen=True)
class UploadLimits:
    """Size limits; load with ``UploadLimits.from_yaml(resolve_config('configs/uploads.yaml'))``."""

    max_bytes: int = 4 * 1024**3
    min_chunk_bytes: int = 1
    max_chunk_bytes: int = 64 * 1024**2
    max_chunks: int = 10_000
    session_ttl_s: float = 86400.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> UploadLimits:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown upload limit keys: {sorted(unknown)}")
        return cls(**raw)


class UploadTransport(Protocol):
    """What a client needs from the server; an HTTP client can implement it unchanged."""

    def create_session(
        self, user_id: str, filename: str, total_bytes: int, sha256: str, chunk_size: int
    ) -> dict[str, Any]: ...
    def put_chunk(
        self, session_id: str, index: int, data: bytes, sha256: str | None = None
    ) -> dict[str, Any]: ...
    def status(self, session_id: str) -> dict[str, Any]: ...
    def finalize(
        self, session_id: str, job_id: str | None = None, prompt: str | None = None
    ) -> dict[str, Any]: ...


class UploadService:
    def __init__(
        self,
        store: SqliteJobStore,
        root: str | Path | None = None,
        limits: UploadLimits | None = None,
        media_limits: MediaLimits = DEFAULT_LIMITS,
        blobs: BlobStore | None = None,
    ) -> None:
        """Chunks and finished files go through ``blobs`` (default: the store's, else a
        ``LocalBlobStore`` at ``root``); keys are ``sessions/<id>/<n>.part`` and
        ``files/<id>/<name>``."""
        self.store = store
        self.root = Path(root) if root is not None else store.path.parent / "uploads"
        self.blobs: BlobStore = blobs or store.blobs or LocalBlobStore(self.root)
        self.limits = limits or UploadLimits()
        self.media_limits = media_limits
        with store._tx() as conn:
            for stmt in UPLOAD_SCHEMA:
                conn.execute(stmt)

    # -- helpers --------------------------------------------------------------------------
    def _session(self, session_id: str, *, live: bool = True) -> dict[str, Any]:
        rows = self.store._query("SELECT * FROM upload_sessions WHERE id = ?", (session_id,))
        if not rows:
            raise _err(ErrorCode.ERR_INVALID_INPUT, f"unknown upload session {session_id!r}")
        row = rows[0]
        if live and row["expires_at"] <= self.store.clock():
            raise _err(
                ErrorCode.ERR_INVALID_INPUT,
                f"upload session {session_id!r} expired; start a new upload",
            )
        return row

    def _expected_size(self, sess: dict[str, Any], index: int) -> int:
        if index == sess["n_chunks"] - 1:
            return int(sess["total_bytes"] - index * sess["chunk_size"])
        return int(sess["chunk_size"])

    def _received(self, session_id: str) -> dict[int, dict[str, Any]]:
        rows = self.store._query(
            "SELECT idx, size, sha256 FROM upload_chunks WHERE session_id = ? ORDER BY idx",
            (session_id,),
        )
        return {r["idx"]: r for r in rows}

    # -- API ------------------------------------------------------------------------------
    def owner(self, session_id: str) -> str | None:
        """The user who opened a session, or None if it does not exist."""
        rows = self.store._query("SELECT user_id FROM upload_sessions WHERE id = ?", (session_id,))
        return str(rows[0]["user_id"]) if rows else None

    def create_session(
        self, user_id: str, filename: str, total_bytes: int, sha256: str, chunk_size: int
    ) -> dict[str, Any]:
        lim = self.limits
        name = Path(filename).name  # never trust a client path
        if not name or name in (".", ".."):
            raise _err(ErrorCode.ERR_INVALID_INPUT, f"bad filename {filename!r}")
        kind = check_extension(Path(name))  # ERR_UNSUPPORTED_MEDIA
        sha256 = sha256.lower()
        if not _HEX64.match(sha256):
            raise _err(ErrorCode.ERR_INVALID_INPUT, "sha256 must be 64 hex characters")
        if total_bytes <= 0:
            raise _err(ErrorCode.ERR_INVALID_INPUT, "total_bytes must be positive")
        media_cap = (
            self.media_limits.max_image_bytes
            if kind == "image"
            else self.media_limits.max_video_bytes
        )
        cap = min(lim.max_bytes, media_cap)
        if total_bytes > cap:
            raise _err(ErrorCode.ERR_FILE_TOO_LARGE, f"{total_bytes} bytes exceeds the {cap} limit")
        if not lim.min_chunk_bytes <= chunk_size <= lim.max_chunk_bytes:
            raise _err(
                ErrorCode.ERR_INVALID_INPUT,
                f"chunk_size must be between {lim.min_chunk_bytes} and {lim.max_chunk_bytes}",
            )
        n_chunks = math.ceil(total_bytes / chunk_size)
        if n_chunks > lim.max_chunks:
            raise _err(
                ErrorCode.ERR_INVALID_INPUT,
                f"{n_chunks} chunks exceeds the {lim.max_chunks} limit; use a larger chunk_size",
            )
        sid = uuid.uuid4().hex
        sdir = self._local_ref(f"sessions/{sid}")
        now = self.store.clock()
        with self.store._tx() as conn:
            conn.execute(
                "INSERT INTO upload_sessions (id, user_id, filename, total_bytes, sha256, "
                "chunk_size, n_chunks, dir, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    user_id,
                    name,
                    total_bytes,
                    sha256,
                    chunk_size,
                    n_chunks,
                    sdir,
                    now,
                    now + lim.session_ttl_s,
                ),
            )
        return self.status(sid)

    def put_chunk(
        self, session_id: str, index: int, data: bytes, sha256: str | None = None
    ) -> dict[str, Any]:
        """Store chunk ``index``. ``sha256`` is the client's hash of ``data``; a mismatch raises
        ``ERR_CHUNK_CORRUPT`` and stores nothing. Without it only the length is checked (the
        whole-file hash is still verified at finalize). Re-sending a stored chunk is a no-op."""
        sess = self._session(session_id)
        if sess["status"] != "open":
            raise _err(ErrorCode.ERR_INVALID_INPUT, f"session is {sess['status']}, not open")
        if not 0 <= index < sess["n_chunks"]:
            raise _err(ErrorCode.ERR_INVALID_INPUT, f"chunk index {index} out of range")
        want = self._expected_size(sess, index)
        if len(data) != want:
            raise _err(
                ErrorCode.ERR_CHUNK_CORRUPT, f"chunk {index} is {len(data)} bytes, expected {want}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if sha256 is not None and sha256.lower() != digest:
            raise _err(ErrorCode.ERR_CHUNK_CORRUPT, f"chunk {index} sha256 mismatch")
        have = self._received(session_id).get(index)
        if have is not None:
            if have["sha256"] != digest:
                raise _err(
                    ErrorCode.ERR_INVALID_INPUT,
                    f"chunk {index} was already stored with different content",
                )
            return {**self.status(session_id), "duplicate": True}
        self.blobs.put(self._chunk_key(session_id, index), data)  # a chunk is absent or complete
        with self.store._tx() as conn:
            conn.execute(
                "INSERT INTO upload_chunks (session_id, idx, size, sha256) VALUES (?,?,?,?) "
                "ON CONFLICT (session_id, idx) DO NOTHING",
                (session_id, index, len(data), digest),
            )
            conn.execute(
                "UPDATE upload_sessions SET expires_at = ? WHERE id = ?",
                (self.store.clock() + self.limits.session_ttl_s, session_id),
            )
        return {**self.status(session_id), "duplicate": False}

    def status(self, session_id: str) -> dict[str, Any]:
        """Resume info: received chunk indices, next missing chunk and its byte offset."""
        sess = self._session(session_id)
        got = sorted(self._received(session_id))
        have = set(got)
        nxt = next((i for i in range(sess["n_chunks"]) if i not in have), None)
        return {
            "session_id": session_id,
            "status": sess["status"],
            "filename": sess["filename"],
            "total_bytes": sess["total_bytes"],
            "sha256": sess["sha256"],
            "chunk_size": sess["chunk_size"],
            "n_chunks": sess["n_chunks"],
            "received": got,
            "next_missing_index": nxt,
            "next_offset": None if nxt is None else nxt * sess["chunk_size"],
            "complete": nxt is None,
            "job_id": sess["job_id"],
            "expires_at": sess["expires_at"],
        }

    def finalize(
        self, session_id: str, job_id: str | None = None, prompt: str | None = None
    ) -> dict[str, Any]:
        """Assemble, verify, sanitize and attach to ``job_id`` (or queue a new job with ``prompt``).

        Idempotent: finalizing a finished session returns the original result. A wrong whole-file
        hash or a media rejection marks the session ``failed`` and discards its chunks.
        """
        sess = self._session(session_id)
        if sess["status"] == "finalized":
            return {"session_id": session_id, "job_id": sess["job_id"], "already_finalized": True}
        if sess["status"] != "open":
            raise _err(ErrorCode.ERR_INVALID_INPUT, f"session is {sess['status']}, not open")
        if job_id is None and not prompt:
            raise _err(ErrorCode.ERR_INVALID_INPUT, "finalize needs a job_id or a prompt")
        st = self.status(session_id)
        if not st["complete"]:
            missing = sorted(set(range(sess["n_chunks"])) - set(st["received"]))
            raise _err(
                ErrorCode.ERR_UPLOAD_INCOMPLETE,
                f"{len(missing)} chunk(s) missing, first is {missing[0]}",
            )
        if job_id is not None:
            self._check_job(job_id, sess["user_id"])

        with tempfile.TemporaryDirectory(prefix="countonce-upload-") as tmp_name:
            tmp = Path(tmp_name)
            raw = tmp / "raw" / sess["filename"]
            raw.parent.mkdir()
            h = hashlib.sha256()
            with raw.open("wb") as out:
                for i in range(sess["n_chunks"]):
                    piece = self.blobs.get(self._chunk_key(session_id, i))
                    h.update(piece)
                    out.write(piece)
            if h.hexdigest() != sess["sha256"]:
                self._fail(sess)
                raise _err(
                    ErrorCode.ERR_CORRUPT_MEDIA,
                    "assembled file does not match the declared sha256; upload it again",
                )
            report = SanitizeReport()
            try:
                if check_extension(raw) == "image":
                    final = sanitize_image(raw, tmp / "clean", self.media_limits, report)
                else:
                    validate_video(raw, self.media_limits, report)
                    final = raw
            except CountOnceError:
                self._fail(sess)
                raise
            data = final.read_bytes()
            blob_key = f"files/{session_id}/{final.name}"
            self.blobs.put(blob_key, data)
        digest = hashlib.sha256(data).hexdigest()
        file_ref = self._local_ref(blob_key)
        exif_stripped = bool(report.images)
        if job_id is None:
            job_id, _ = self.store.submit(
                sess["user_id"], prompt or "", [file_ref], idempotency_key=f"upload:{session_id}"
            )
            self._tag_file(job_id, file_ref, len(data), digest, exif_stripped)
        else:
            self._attach(job_id, file_ref, len(data), digest, exif_stripped)
        with self.store._tx() as conn:
            conn.execute(
                "UPDATE upload_sessions SET status = 'finalized', job_id = ?, expires_at = ? "
                "WHERE id = ?",
                (job_id, self.store.clock() + self.limits.session_ttl_s, session_id),
            )
            conn.execute("DELETE FROM upload_chunks WHERE session_id = ?", (session_id,))
        delete_prefix(self.blobs, f"sessions/{session_id}/")
        return {
            "session_id": session_id,
            "job_id": job_id,
            "path": file_ref,
            "bytes": len(data),
            "sha256": digest,
            "sanitize": report.to_dict(),
            "already_finalized": False,
        }

    # -- internals ------------------------------------------------------------------------
    def _chunk_key(self, session_id: str, index: int) -> str:
        return f"sessions/{session_id}/{index:06d}.part"

    def _local_ref(self, blob_key: str) -> str:
        """What goes in ``job_files.key`` / ``upload_sessions.dir``: the absolute file path on the
        local backend (as before), the blob key otherwise."""
        if isinstance(self.blobs, LocalBlobStore):
            return str(self.blobs.path(blob_key))
        return blob_key

    def _fail(self, sess: dict[str, Any]) -> None:
        with self.store._tx() as conn:
            conn.execute("UPDATE upload_sessions SET status = 'failed' WHERE id = ?", (sess["id"],))
            conn.execute("DELETE FROM upload_chunks WHERE session_id = ?", (sess["id"],))
        delete_prefix(self.blobs, f"sessions/{sess['id']}/")

    def _check_job(self, job_id: str, user_id: str) -> None:
        job = self.store.get(job_id)
        if job is None or job["status"] == "deleted":
            raise _err(ErrorCode.ERR_INVALID_INPUT, f"no such job: {job_id}")
        if job["user_id"] != user_id:
            raise _err(ErrorCode.ERR_INVALID_INPUT, "job belongs to another user")
        if job["status"] not in ("draft", "queued"):
            raise _err(ErrorCode.ERR_INVALID_INPUT, f"job is {job['status']}; cannot add files")

    def _attach(self, job_id: str, key: str, size: int, digest: str, exif_stripped: bool) -> None:
        with self.store._tx() as conn:
            conn.execute(
                "INSERT INTO job_files (id, job_id, kind, key, bytes, sha256, exif_stripped) "
                "VALUES (?, ?, 'input', ?, ?, ?, ?)",
                (
                    uuid.uuid4().hex,
                    job_id,
                    key,
                    size,
                    digest,
                    int(exif_stripped),
                ),
            )

    def _tag_file(self, job_id: str, key: str, size: int, digest: str, exif_stripped: bool) -> None:
        with self.store._tx() as conn:
            conn.execute(
                "UPDATE job_files SET bytes = ?, sha256 = ?, exif_stripped = ? "
                "WHERE job_id = ? AND key = ?",
                (size, digest, int(exif_stripped), job_id, key),
            )


def purge_upload_sessions(store: SqliteJobStore, now: float | None = None) -> int:
    """Delete expired sessions: chunk files, chunk rows and session rows. Returns the count.

    Called by ``SqliteJobStore.purge_expired``. A finalized session's sanitized file is the job's
    input and is removed with the job, not here.
    """
    now = store.clock() if now is None else now
    with store._tx() as conn:
        for stmt in UPLOAD_SCHEMA:
            conn.execute(stmt)
        rows = conn.execute(
            "SELECT id, dir FROM upload_sessions WHERE expires_at <= ?", (now,)
        ).fetchall()
        for r in rows:
            conn.execute("DELETE FROM upload_chunks WHERE session_id = ?", (r["id"],))
            conn.execute("DELETE FROM upload_sessions WHERE id = ?", (r["id"],))
    for r in rows:
        if store.blobs is not None:
            delete_prefix(store.blobs, f"sessions/{r['id']}/")
        if Path(r["dir"]).is_absolute():  # sessions made before BlobStore, or on local disk
            shutil.rmtree(r["dir"], ignore_errors=True)
    return len(rows)


class LocalTransport:
    """In-process transport: calls an ``UploadService`` directly."""

    def __init__(self, service: UploadService) -> None:
        self.service = service

    def create_session(
        self, user_id: str, filename: str, total_bytes: int, sha256: str, chunk_size: int
    ) -> dict[str, Any]:
        return self.service.create_session(user_id, filename, total_bytes, sha256, chunk_size)

    def put_chunk(
        self, session_id: str, index: int, data: bytes, sha256: str | None = None
    ) -> dict[str, Any]:
        return self.service.put_chunk(session_id, index, data, sha256)

    def status(self, session_id: str) -> dict[str, Any]:
        return self.service.status(session_id)

    def finalize(
        self, session_id: str, job_id: str | None = None, prompt: str | None = None
    ) -> dict[str, Any]:
        return self.service.finalize(session_id, job_id, prompt)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upload_file(
    transport: UploadTransport,
    user_id: str,
    path: Path,
    *,
    chunk_size: int = 4 * 1024**2,
    resume: str | None = None,
    job_id: str | None = None,
    prompt: str | None = None,
    on_session: Callable[[str], None] | None = None,
    on_chunk: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Upload ``path`` through ``transport``; with ``resume`` send only the missing chunks.

    ``on_session`` receives the session id before any chunk is sent (so a caller can print it for
    ``--resume``); ``on_chunk(index, n_chunks)`` fires after each chunk actually sent. Raises
    ``UploadError`` if the resumed session was declared for a different file.
    """
    path = Path(path)
    size = path.stat().st_size
    digest = file_sha256(path)
    if resume is None:
        st = transport.create_session(user_id, path.name, size, digest, chunk_size)
    else:
        st = transport.status(resume)
        if st["sha256"] != digest or st["total_bytes"] != size:
            raise _err(
                ErrorCode.ERR_INVALID_INPUT,
                f"{path.name} differs from the file in session {resume}",
            )
    sid = st["session_id"]
    if on_session is not None:
        on_session(sid)
    csize, n = st["chunk_size"], st["n_chunks"]
    have = set(st["received"])
    with path.open("rb") as f:
        for i in range(n):
            if i in have:
                continue
            f.seek(i * csize)
            data = f.read(csize)
            transport.put_chunk(sid, i, data, hashlib.sha256(data).hexdigest())
            if on_chunk is not None:
                on_chunk(i, n)
    return transport.finalize(sid, job_id, prompt)


def open_service(
    db: Path,
    root: Path | None = None,
    limits_path: Path | None = None,
    blobs: BlobStore | None = None,
) -> UploadService:
    """Service for a job database; limits from ``configs/uploads.yaml`` (CWD, else packaged)."""
    from countonce.resources import resolve_config

    limits = UploadLimits.from_yaml(resolve_config(limits_path or Path("configs/uploads.yaml")))
    return UploadService(SqliteJobStore(db), root, limits, blobs=blobs)


__all__ = [
    "LocalTransport",
    "UploadError",
    "UploadLimits",
    "UploadService",
    "UploadTransport",
    "file_sha256",
    "open_service",
    "purge_upload_sessions",
    "upload_file",
]
