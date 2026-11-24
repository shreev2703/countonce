"""Job queue store (PLAN 14.3 tables, 14.4 queue semantics) on stdlib ``sqlite3``.

``JobStore`` is the protocol the worker and (later) the admin metrics depend on.
``SqliteJobStore`` is the local implementation. The SQL is plain standard SQL (``ON CONFLICT DO
NOTHING``, ``UPDATE ... RETURNING``, scalar sub-selects, ``COALESCE``) with ``?`` placeholders, so a
Postgres store can reuse every statement: ``PostgresJobStore`` (``jobs/pg_store.py``) subclasses
this class, translates ``?`` to ``%s`` and claims with ``FOR UPDATE SKIP LOCKED``.

Atomic claim: a single ``UPDATE jobs ... WHERE id = (SELECT id ... LIMIT 1) RETURNING *``. SQLite
serialises writers, so two workers can never claim the same row. On Postgres the same statement
needs ``FOR UPDATE SKIP LOCKED`` in the sub-select, which is what this stands in for.

Lifecycle: ``queued -> running -> done | queued (retry) | dead_letter | failed``. ``attempts`` is
incremented on each claim. Retry vs dead-letter after a *reported* error comes from
``countonce.errors.retry_decision``. A *silent* worker (no heartbeat for ``heartbeat_timeout_s``)
is requeued, at most ``MAX_STALE_ATTEMPTS`` tries in total, then ``failed`` with
``ERR_WORKER_TIMEOUT``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import shutil
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from countonce.errors import ErrorCode, retry_decision
from countonce.storage import BlobStore, delete_prefix

HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_TIMEOUT_S = 120.0
MAX_STALE_ATTEMPTS = 2

STATUSES = ("queued", "running", "done", "failed", "dead_letter", "deleted")
TERMINAL = ("done", "failed", "dead_letter", "deleted")

RETENTION_DAYS = 30.0
DAY_S = 86400.0

CLAIM_LOCK = ""  # Postgres: "FOR UPDATE SKIP LOCKED"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        prompt TEXT NOT NULL,
        negative_prompts TEXT NOT NULL DEFAULT '[]',
        options_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'queued',
        stage TEXT NOT NULL DEFAULT 'queued',
        progress REAL NOT NULL DEFAULT 0,
        error_code TEXT,
        error_message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        worker_id TEXT,
        idempotency_key TEXT,
        input_hash TEXT,
        model_routing_id TEXT,
        site_id TEXT,
        captured_at DOUBLE PRECISION,
        created_at DOUBLE PRECISION NOT NULL,
        started_at DOUBLE PRECISION,
        finished_at DOUBLE PRECISION,
        heartbeat_at DOUBLE PRECISION,
        next_attempt_at DOUBLE PRECISION NOT NULL DEFAULT 0,
        UNIQUE (user_id, idempotency_key)
    )""",
    "CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs (status, next_attempt_at, created_at)",
    """CREATE TABLE IF NOT EXISTS job_files (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs (id),
        kind TEXT NOT NULL,
        key TEXT NOT NULL,
        bytes INTEGER,
        sha256 TEXT,
        exif_stripped INTEGER NOT NULL DEFAULT 0,
        privacy_masked INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS results (
        job_id TEXT PRIMARY KEY REFERENCES jobs (id),
        schema_version TEXT,
        count INTEGER NOT NULL,
        interval_low INTEGER,
        interval_high INTEGER,
        interval_level REAL,
        interval_method TEXT,
        n_calibration INTEGER,
        zero_count INTEGER NOT NULL DEFAULT 0,
        result_json TEXT,
        warnings_json TEXT,
        model_versions_json TEXT,
        recon_cache_hit INTEGER NOT NULL DEFAULT 0,
        oom_fallback_rung INTEGER NOT NULL DEFAULT 0,
        peak_vram_mb REAL NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs (id),
        user_id TEXT,
        correct INTEGER NOT NULL,
        user_count INTEGER,
        comment TEXT,
        created_at DOUBLE PRECISION
    )""",
    """CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        training_opt_in INTEGER NOT NULL DEFAULT 0,
        created_at DOUBLE PRECISION
    )""",
    """CREATE TABLE IF NOT EXISTS share_tokens (
        token TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs (id),
        created_at DOUBLE PRECISION NOT NULL,
        revoked_at DOUBLE PRECISION
    )""",
    """CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        salt TEXT NOT NULL,
        hashed_key TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL,
        last_used_at DOUBLE PRECISION,
        revoked_at DOUBLE PRECISION
    )""",
    """CREATE TABLE IF NOT EXISTS rate_buckets (
        user_id TEXT PRIMARY KEY,
        tokens DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS workers (
        id TEXT PRIMARY KEY,
        name TEXT,
        kind TEXT NOT NULL DEFAULT 'laptop',
        last_heartbeat DOUBLE PRECISION,
        gpu_name TEXT,
        vram_mb INTEGER,
        cpu_name TEXT,
        cpu_count INTEGER,
        version TEXT
    )""",
]


def build_claim_sql(lock: str = "") -> str:
    """The atomic claim statement; ``lock`` is ``"FOR UPDATE SKIP LOCKED"`` on Postgres."""
    return f"""UPDATE jobs SET status = 'running', worker_id = ?, attempts = attempts + 1,
    started_at = COALESCE(started_at, ?), heartbeat_at = ?, stage = 'starting', progress = 0
WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND next_attempt_at <= ?
            ORDER BY created_at, id LIMIT 1 {lock})
  AND status = 'queued'
RETURNING *"""


CLAIM_SQL = build_claim_sql(CLAIM_LOCK)

_INSERT_JOB = """INSERT INTO jobs (id, user_id, prompt, negative_prompts, options_json,
    idempotency_key, input_hash, model_routing_id, site_id, captured_at, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (user_id, idempotency_key) DO NOTHING"""

_FAIL_STALE = """UPDATE jobs SET status = 'failed', error_code = 'ERR_WORKER_TIMEOUT',
    error_message = 'worker stopped sending heartbeats', finished_at = ?
WHERE status = 'running' AND heartbeat_at < ? AND attempts >= ?"""

_REQUEUE_STALE = """UPDATE jobs SET status = 'queued', stage = 'queued', progress = 0,
    worker_id = NULL, next_attempt_at = ?
WHERE status = 'running' AND heartbeat_at < ? AND attempts < ?"""


class RateLimitError(RuntimeError):
    """A user exceeded their submit token bucket; retry after ``retry_after_s``."""

    def __init__(self, user_id: str, retry_after_s: float) -> None:
        super().__init__(f"rate limit exceeded for {user_id!r}; retry in {retry_after_s:.1f}s")
        self.user_id = user_id
        self.retry_after_s = retry_after_s


def open_store(
    kind: str = "sqlite", db: str | Path = "jobs.db", *, pg_url: str | None = None, **kw: Any
) -> SqliteJobStore:
    """``--store sqlite|postgres`` factory. ``db`` anchors outputs/uploads for either kind.

    Postgres reads ``pg_url`` or ``COUNTONCE_PG_URL`` and needs the ``db`` extra.
    """
    if kind == "sqlite":
        return SqliteJobStore(db, **kw)
    if kind != "postgres":
        raise ValueError(f"unknown store {kind!r}; expected sqlite or postgres")
    import os

    url = pg_url or os.environ.get("COUNTONCE_PG_URL")
    if not url:
        raise ValueError("--store postgres needs --pg-url or COUNTONCE_PG_URL")
    from countonce.jobs.pg_store import PostgresJobStore

    return PostgresJobStore(url, data_dir=Path(db).parent, **kw)


def _hash_key(salt: str, key: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", key.encode(), bytes.fromhex(salt), 100_000).hex()


@dataclass(frozen=True)
class FailOutcome:
    status: str  # "queued" (retry) | "dead_letter" | "ignored" (job no longer ours)
    backoff_s: float = 0.0


class JobStore(Protocol):
    def submit(
        self,
        user_id: str,
        prompt: str,
        inputs: Sequence[str | Path],
        *,
        options: dict[str, Any] | None = None,
        negative_prompts: Sequence[str] = (),
        idempotency_key: str | None = None,
        input_hash: str | None = None,
        model_routing_id: str | None = None,
        site_id: str | None = None,
        captured_at: float | None = None,
    ) -> tuple[str, bool]: ...
    def get(self, job_id: str) -> dict[str, Any] | None: ...
    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]: ...
    def job_files(self, job_id: str, kind: str = "input") -> list[dict[str, Any]]: ...
    def get_result(self, job_id: str) -> dict[str, Any] | None: ...
    def register_worker(self, worker_id: str, **info: Any) -> None: ...
    def claim(self, worker_id: str) -> dict[str, Any] | None: ...
    def heartbeat(self, job_id: str, worker_id: str) -> bool: ...
    def set_progress(self, job_id: str, worker_id: str, stage: str, progress: float) -> bool: ...
    def complete(self, job_id: str, worker_id: str, result: dict[str, Any]) -> bool: ...
    def fail(
        self, job_id: str, worker_id: str, code: ErrorCode | str, message: str
    ) -> FailOutcome: ...
    def requeue_stale(self) -> tuple[int, int]: ...
    def counts(self) -> dict[str, int]: ...


class SqliteJobStore:
    """``JobStore`` on a SQLite file; one short-lived connection per call (thread-safe)."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT_S,
        max_stale_attempts: int = MAX_STALE_ATTEMPTS,
        max_attempts: int | None = None,
        backoff_base_s: float | None = None,
        rate_limit: int | None = None,
        rate_window_s: float = 60.0,
        output_root: str | Path | None = None,
        blobs: BlobStore | None = None,
    ) -> None:
        """``blobs`` is the object store holding input/output files (None: plain local paths).

        ``rate_limit`` submits per ``rate_window_s`` per user (token bucket; None = off).

        ``output_root`` is where the worker writes ``<output_root>/<job_id>``.
        """
        self.blobs = blobs
        self.rate_limit = rate_limit
        self.rate_window_s = rate_window_s
        self.output_root = Path(output_root) if output_root is not None else None
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.max_stale_attempts = max_stale_attempts
        self._retry_kwargs: dict[str, Any] = {}
        if max_attempts is not None:
            self._retry_kwargs["max_attempts"] = max_attempts
        if backoff_base_s is not None:
            self._retry_kwargs["base_s"] = backoff_base_s
        with self._tx() as conn:
            for stmt in SCHEMA:
                conn.execute(stmt)

    # -- dialect hooks (PostgresJobStore overrides) ----------------------------------------
    _TRUE_SQL = "1"
    _CLAIM_SQL = CLAIM_SQL
    _FLAG_RESULT_COLS = ("recon_cache_hit", "zero_count")

    @staticmethod
    def _flag(value: bool | int) -> Any:
        """A boolean column value: SQLite stores 0/1, Postgres a real boolean."""
        return int(value)

    # -- plumbing -------------------------------------------------------------------------
    @contextmanager
    def _tx(self) -> Iterator[Any]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    # -- submit / read --------------------------------------------------------------------
    def submit(
        self,
        user_id: str,
        prompt: str,
        inputs: Sequence[str | Path],
        *,
        options: dict[str, Any] | None = None,
        negative_prompts: Sequence[str] = (),
        idempotency_key: str | None = None,
        input_hash: str | None = None,
        model_routing_id: str | None = None,
        site_id: str | None = None,
        captured_at: float | None = None,
    ) -> tuple[str, bool]:
        """Queue a job; the same ``(user_id, idempotency_key)`` returns the original job id."""
        job_id = uuid.uuid4().hex
        with self._tx() as conn:
            cur = conn.execute(
                _INSERT_JOB,
                (
                    job_id,
                    user_id,
                    prompt,
                    json.dumps(list(negative_prompts)),
                    json.dumps(options or {}, sort_keys=True),
                    idempotency_key,
                    input_hash,
                    model_routing_id,
                    site_id,
                    captured_at,
                    self.clock(),
                ),
            )
            if cur.rowcount == 0:
                row = conn.execute(
                    "SELECT id FROM jobs WHERE user_id = ? AND idempotency_key = ?",
                    (user_id, idempotency_key),
                ).fetchone()
                return str(row["id"]), False
            self._consume_token(conn, user_id)
            for path in inputs:
                conn.execute(
                    "INSERT INTO job_files (id, job_id, kind, key) VALUES (?, ?, 'input', ?)",
                    (uuid.uuid4().hex, job_id, str(path)),
                )
        return job_id, True

    # -- draft jobs (HTTP API: POST /jobs, then upload, then POST /jobs/{id}/submit) ----------
    def create_draft(
        self,
        user_id: str,
        prompt: str,
        *,
        options: dict[str, Any] | None = None,
        negative_prompts: Sequence[str] = (),
        idempotency_key: str | None = None,
        site_id: str | None = None,
        captured_at: float | None = None,
    ) -> tuple[str, bool]:
        """Create a ``draft`` job (never claimed by a worker); same idempotency rule as submit."""
        job_id = uuid.uuid4().hex
        with self._tx() as conn:
            cur = conn.execute(
                _INSERT_JOB,
                (
                    job_id,
                    user_id,
                    prompt,
                    json.dumps(list(negative_prompts)),
                    json.dumps(options or {}, sort_keys=True),
                    idempotency_key,
                    None,
                    None,
                    site_id,
                    captured_at,
                    self.clock(),
                ),
            )
            if cur.rowcount == 0:
                row = conn.execute(
                    "SELECT id FROM jobs WHERE user_id = ? AND idempotency_key = ?",
                    (user_id, idempotency_key),
                ).fetchone()
                return str(row["id"]), False
            conn.execute(
                "UPDATE jobs SET status = 'draft', stage = 'draft' WHERE id = ?", (job_id,)
            )
        return job_id, True

    def queue_draft(self, job_id: str) -> bool:
        """``draft -> queued`` (consumes a rate-limit token). False if it was not a draft."""
        with self._tx() as conn:
            row = conn.execute(
                "SELECT user_id, status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != "draft":
                return False
            self._consume_token(conn, row["user_id"])
            conn.execute(
                "UPDATE jobs SET status = 'queued', stage = 'queued', created_at = ? WHERE id = ?",
                (self.clock(), job_id),
            )
        return True

    def add_feedback(
        self,
        job_id: str,
        user_id: str | None,
        correct: bool,
        user_count: int | None = None,
        comment: str | None = None,
    ) -> str:
        fid = uuid.uuid4().hex
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO feedback (id, job_id, user_id, correct, user_count, comment, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fid, job_id, user_id, self._flag(correct), user_count, comment, self.clock()),
            )
        return fid

    def get(self, job_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return rows[0] if rows else None

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            return self._query("SELECT * FROM jobs ORDER BY created_at, id")
        return self._query("SELECT * FROM jobs WHERE status = ? ORDER BY created_at, id", (status,))

    def job_files(self, job_id: str, kind: str = "input") -> list[dict[str, Any]]:
        return self._query(
            "SELECT * FROM job_files WHERE job_id = ? AND kind = ? ORDER BY key", (job_id, kind)
        )

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM results WHERE job_id = ?", (job_id,))
        return rows[0] if rows else None

    def counts(self) -> dict[str, int]:
        rows = self._query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        out = dict.fromkeys(STATUSES, 0)
        out.update({r["status"]: int(r["n"]) for r in rows})
        return out

    # -- workers --------------------------------------------------------------------------
    def register_worker(self, worker_id: str, **info: Any) -> None:
        cols = ("name", "kind", "gpu_name", "vram_mb", "cpu_name", "cpu_count", "version")
        vals = [info.get(c) for c in cols]
        if vals[1] is None:
            vals[1] = "laptop"
        with self._tx() as conn:
            conn.execute(
                f"""INSERT INTO workers (id, last_heartbeat, {", ".join(cols)})
                VALUES (?, ?, {", ".join("?" for _ in cols)})
                ON CONFLICT (id) DO UPDATE SET last_heartbeat = excluded.last_heartbeat,
                {", ".join(f"{c} = excluded.{c}" for c in cols)}""",
                (worker_id, self.clock(), *vals),
            )

    # -- queue ----------------------------------------------------------------------------
    def claim(self, worker_id: str) -> dict[str, Any] | None:
        now = self.clock()
        with self._tx() as conn:
            row = conn.execute(self._CLAIM_SQL, (worker_id, now, now, now)).fetchone()
        return dict(row) if row else None

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        now = self.clock()
        with self._tx() as conn:
            conn.execute("UPDATE workers SET last_heartbeat = ? WHERE id = ?", (now, worker_id))
            cur = conn.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND worker_id = ? "
                "AND status = 'running'",
                (now, job_id, worker_id),
            )
            return bool(cur.rowcount == 1)

    def set_progress(self, job_id: str, worker_id: str, stage: str, progress: float) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE jobs SET stage = ?, progress = ?, heartbeat_at = ? WHERE id = ? "
                "AND worker_id = ? AND status = 'running'",
                (stage, min(max(progress, 0.0), 1.0), self.clock(), job_id, worker_id),
            )
            return bool(cur.rowcount == 1)

    def complete(self, job_id: str, worker_id: str, result: dict[str, Any]) -> bool:
        """Mark done and write the result row; no-op (False) if the job is no longer ours."""
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'done', stage = 'done', progress = 1, finished_at = ?, "
                "error_code = NULL, error_message = NULL WHERE id = ? AND worker_id = ? "
                "AND status = 'running'",
                (self.clock(), job_id, worker_id),
            )
            if cur.rowcount == 0:
                return False
            result = {
                k: self._flag(v) if k in self._FLAG_RESULT_COLS else v for k, v in result.items()
            }
            cols = list(result)
            conn.execute(
                f"INSERT INTO results (job_id, {', '.join(cols)}) "
                f"VALUES (?, {', '.join('?' for _ in cols)})",
                (job_id, *result.values()),
            )
            return True

    def fail(self, job_id: str, worker_id: str, code: ErrorCode | str, message: str) -> FailOutcome:
        """Record a reported failure: ``retry_decision`` picks retry-with-backoff or dead letter."""
        now = self.clock()
        code = ErrorCode(code)
        with self._tx() as conn:
            row = conn.execute(
                "SELECT attempts FROM jobs WHERE id = ? AND worker_id = ? AND status = 'running'",
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                return FailOutcome("ignored")
            decision = retry_decision(code, int(row["attempts"]), **self._retry_kwargs)
            if decision.retry:
                conn.execute(
                    "UPDATE jobs SET status = 'queued', stage = 'queued', progress = 0, "
                    "worker_id = NULL, error_code = ?, error_message = ?, next_attempt_at = ? "
                    "WHERE id = ?",
                    (code.value, message, now + decision.backoff_s, job_id),
                )
                return FailOutcome("queued", decision.backoff_s)
            conn.execute(
                "UPDATE jobs SET status = 'dead_letter', error_code = ?, error_message = ?, "
                "finished_at = ? WHERE id = ?",
                (code.value, message, now, job_id),
            )
            return FailOutcome("dead_letter")

    def requeue_stale(self) -> tuple[int, int]:
        """Requeue running jobs with a missed heartbeat; ``(requeued, failed)`` counts."""
        now = self.clock()
        cutoff = now - self.heartbeat_timeout_s
        with self._tx() as conn:
            failed = conn.execute(_FAIL_STALE, (now, cutoff, self.max_stale_attempts)).rowcount
            requeued = conn.execute(_REQUEUE_STALE, (now, cutoff, self.max_stale_attempts)).rowcount
        return requeued, failed

    # -- rate limit -----------------------------------------------------------------------
    def _consume_token(self, conn: Any, user_id: str) -> None:
        """Token bucket: capacity ``rate_limit``, refilled evenly over ``rate_window_s``."""
        if self.rate_limit is None:
            return
        now = self.clock()
        cap = float(self.rate_limit)
        row = conn.execute(
            "SELECT tokens, updated_at FROM rate_buckets WHERE user_id = ?", (user_id,)
        ).fetchone()
        tokens = cap
        if row is not None:
            refill = (now - float(row["updated_at"])) * cap / self.rate_window_s
            tokens = min(cap, float(row["tokens"]) + max(refill, 0.0))
        if tokens < 1.0:
            raise RateLimitError(user_id, (1.0 - tokens) * self.rate_window_s / cap)
        conn.execute(
            "INSERT INTO rate_buckets (user_id, tokens, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (user_id) DO UPDATE SET tokens = excluded.tokens, "
            "updated_at = excluded.updated_at",
            (user_id, tokens - 1.0, now),
        )

    # -- deletion and retention -----------------------------------------------------------
    def _delete_blob(self, key: str) -> int:
        """Remove one stored file via the blob store, or as a plain path if it is outside it."""
        if self.blobs is not None:
            try:
                return int(self.blobs.delete(key))
            except ValueError:  # a legacy absolute path outside the local blob root
                pass
        try:
            Path(key).unlink()
            return 1
        except FileNotFoundError:
            return 0

    def record_file(self, job_id: str, kind: str, key: str, size: int, sha256: str) -> None:
        """Register a stored file (e.g. an uploaded output) in ``job_files``."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO job_files (id, job_id, kind, key, bytes, sha256) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, job_id, kind, key, size, sha256),
            )

    def _remove_files(self, conn: Any, job_id: str, kinds: Sequence[str]) -> int:
        """Delete a job's files on disk and their rows; returns the number of files removed."""
        n = 0
        marks = ", ".join("?" for _ in kinds)
        rows = conn.execute(
            f"SELECT id, key FROM job_files WHERE job_id = ? AND kind IN ({marks})",
            (job_id, *kinds),
        ).fetchall()
        for r in rows:
            n += self._delete_blob(r["key"])
        conn.execute(
            f"DELETE FROM job_files WHERE job_id = ? AND kind IN ({marks})", (job_id, *kinds)
        )
        if "output" in kinds and self.blobs is not None:
            n += delete_prefix(self.blobs, f"outputs/{job_id}/")
        if "output" in kinds and self.output_root is not None:
            out = self.output_root / job_id
            if out.is_dir():
                shutil.rmtree(out, ignore_errors=True)
                n += 1
        return n

    def delete_job(self, job_id: str) -> bool:
        """Remove output files (and inputs), revoke share tokens, set status ``deleted``.

        Returns False if the job does not exist or is already deleted. A running job is refused
        (``ValueError``) so a worker is never writing into a directory being removed.
        """
        with self._tx() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] == "deleted":
                return False
            if row["status"] == "running":
                raise ValueError(f"job {job_id} is running; cannot delete")
            self._remove_files(conn, job_id, ("input", "output"))
            now = self.clock()
            conn.execute(
                "UPDATE share_tokens SET revoked_at = ? WHERE job_id = ? AND revoked_at IS NULL",
                (now, job_id),
            )
            conn.execute("DELETE FROM results WHERE job_id = ?", (job_id,))
            conn.execute(
                "UPDATE jobs SET status = 'deleted', stage = 'deleted', finished_at = COALESCE("
                "finished_at, ?) WHERE id = ?",
                (now, job_id),
            )
        return True

    def purge_expired(
        self, now: float | None = None, retention_days: float = RETENTION_DAYS
    ) -> list[str]:
        """Delete inputs and outputs of finished jobs older than ``retention_days``.

        Age is ``finished_at`` (falling back to ``created_at``). Queued/running jobs are never
        purged. Returns purged job ids; their rows stay with status ``deleted`` for the audit trail.
        """
        now = self.clock() if now is None else now
        cutoff = now - retention_days * DAY_S
        rows = self._query(
            "SELECT id FROM jobs WHERE status IN ('done', 'failed', 'dead_letter', 'draft') "
            "AND COALESCE(finished_at, created_at) <= ? ORDER BY created_at, id",
            (cutoff,),
        )
        purged = []
        for r in rows:
            if self.delete_job(r["id"]):
                purged.append(r["id"])
        from countonce.jobs.uploads import purge_upload_sessions  # lazy: uploads imports store

        purge_upload_sessions(self, now)
        return purged

    # -- share tokens ---------------------------------------------------------------------
    def create_share_token(self, job_id: str) -> str:
        """A random, revocable, read-only token for one job's result."""
        # Fixed prefix: a bare token_urlsafe value can start with "-" and be parsed as a CLI flag.
        # Unprefixed tokens issued earlier still resolve (lookup is by exact stored value).
        token = f"st_{secrets.token_urlsafe(32)}"
        with self._tx() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] == "deleted":
                raise KeyError(f"no such job: {job_id}")
            conn.execute(
                "INSERT INTO share_tokens (token, job_id, created_at) VALUES (?, ?, ?)",
                (token, job_id, self.clock()),
            )
        return token

    def resolve_share_token(self, token: str) -> dict[str, Any] | None:
        """Read-only view ``{job, result}`` for a live token; None if unknown/revoked/deleted."""
        rows = self._query(
            "SELECT job_id FROM share_tokens WHERE token = ? AND revoked_at IS NULL", (token,)
        )
        if not rows:
            return None
        job = self.get(rows[0]["job_id"])
        if job is None or job["status"] == "deleted":
            return None
        public = ("id", "status", "stage", "progress", "prompt", "created_at", "finished_at")
        return {"job": {k: job[k] for k in public}, "result": self.get_result(job["id"])}

    def revoke_share_token(self, token: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE share_tokens SET revoked_at = ? WHERE token = ? AND revoked_at IS NULL",
                (self.clock(), token),
            )
            return bool(cur.rowcount == 1)

    # -- API keys -------------------------------------------------------------------------
    def create_api_key(self, user_id: str) -> tuple[str, str]:
        """Returns ``(key_id, plaintext)``; the plaintext is never stored and shown only once.

        The key is ``co_<key_id>_<secret>``; only a salted PBKDF2 hash of it is kept.
        """
        key_id = uuid.uuid4().hex[:12]
        plaintext = f"co_{key_id}_{secrets.token_urlsafe(32)}"
        salt = secrets.token_hex(16)
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO api_keys (id, user_id, salt, hashed_key, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key_id, user_id, salt, _hash_key(salt, plaintext), self.clock()),
            )
        return key_id, plaintext

    def verify_api_key(self, plaintext: str) -> str | None:
        """The owning user id for a live key (recording ``last_used_at``), else None."""
        parts = plaintext.split("_", 2)
        if len(parts) != 3 or parts[0] != "co":
            return None
        with self._tx() as conn:
            row = conn.execute(
                "SELECT user_id, salt, hashed_key FROM api_keys WHERE id = ? "
                "AND revoked_at IS NULL",
                (parts[1],),
            ).fetchone()
            if row is None or not hmac.compare_digest(
                _hash_key(row["salt"], plaintext), row["hashed_key"]
            ):
                return None
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (self.clock(), parts[1])
            )
            return str(row["user_id"])

    def revoke_api_key(self, key_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (self.clock(), key_id),
            )
            return bool(cur.rowcount == 1)

    def api_keys(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Key metadata (never the hash or plaintext)."""
        sql = "SELECT id, user_id, created_at, last_used_at, revoked_at FROM api_keys"
        if user_id is None:
            return self._query(sql + " ORDER BY created_at, id")
        return self._query(sql + " WHERE user_id = ? ORDER BY created_at, id", (user_id,))

    # -- training opt-in ------------------------------------------------------------------
    def set_training_opt_in(self, user_id: str, opt_in: bool) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO users (id, training_opt_in, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET training_opt_in = excluded.training_opt_in",
                (user_id, self._flag(opt_in), self.clock()),
            )

    def opted_in_users(self) -> set[str]:
        rows = self._query(f"SELECT id FROM users WHERE training_opt_in = {self._TRUE_SQL}")
        return {r["id"] for r in rows}
