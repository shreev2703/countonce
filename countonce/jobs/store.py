"""Job queue store (PLAN 14.3 tables, 14.4 queue semantics) on stdlib ``sqlite3``.

``JobStore`` is the protocol the worker and (later) the admin metrics depend on.
``SqliteJobStore`` is the local implementation. The SQL is plain standard SQL (``ON CONFLICT DO
NOTHING``, ``UPDATE ... RETURNING``, scalar sub-selects, ``COALESCE``) with ``?`` placeholders, so a
Postgres store can reuse every statement: translate ``?`` to ``%s`` and set ``CLAIM_LOCK`` to
``"FOR UPDATE SKIP LOCKED"``.

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

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from countonce.errors import ErrorCode, retry_decision

HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_TIMEOUT_S = 120.0
MAX_STALE_ATTEMPTS = 2

STATUSES = ("queued", "running", "done", "failed", "dead_letter", "deleted")
TERMINAL = ("done", "failed", "dead_letter", "deleted")

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

CLAIM_SQL = f"""UPDATE jobs SET status = 'running', worker_id = ?, attempts = attempts + 1,
    started_at = COALESCE(started_at, ?), heartbeat_at = ?, stage = 'starting', progress = 0
WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND next_attempt_at <= ?
            ORDER BY created_at, id LIMIT 1 {CLAIM_LOCK})
  AND status = 'queued'
RETURNING *"""

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
    ) -> None:
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

    # -- plumbing -------------------------------------------------------------------------
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
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
            for path in inputs:
                conn.execute(
                    "INSERT INTO job_files (id, job_id, kind, key) VALUES (?, ?, 'input', ?)",
                    (uuid.uuid4().hex, job_id, str(path)),
                )
        return job_id, True

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
            row = conn.execute(CLAIM_SQL, (worker_id, now, now, now)).fetchone()
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
            return cur.rowcount == 1

    def set_progress(self, job_id: str, worker_id: str, stage: str, progress: float) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE jobs SET stage = ?, progress = ?, heartbeat_at = ? WHERE id = ? "
                "AND worker_id = ? AND status = 'running'",
                (stage, min(max(progress, 0.0), 1.0), self.clock(), job_id, worker_id),
            )
            return cur.rowcount == 1

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
