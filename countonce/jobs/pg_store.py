"""``PostgresJobStore``: the ``JobStore`` protocol on Postgres (PLAN 14.3 tables, 14.4 queue).

It subclasses ``SqliteJobStore`` so every statement (idempotent insert, heartbeat, stale requeue,
``retry_decision``-driven dead-letter, share tokens, API keys, retention) is shared SQL; only the
connection layer and a few dialect hooks differ:

* ``?`` placeholders are rewritten to ``%s`` for psycopg 3.
* The claim is ``UPDATE ... WHERE id = (SELECT ... LIMIT 1 FOR UPDATE SKIP LOCKED)``, so concurrent
  workers skip rows another worker is claiming instead of blocking or double-claiming.
* Flag columns are real booleans.
* The schema comes from the Alembic migrations (``migrate=True``), not ``CREATE TABLE`` strings.

One short-lived connection per call, like the SQLite store. A pool is a later optimisation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from countonce.db.migrate import upgrade
from countonce.db.url import psycopg_url
from countonce.jobs.store import (
    HEARTBEAT_TIMEOUT_S,
    MAX_STALE_ATTEMPTS,
    SqliteJobStore,
    build_claim_sql,
)
from countonce.storage import BlobStore

CLAIM_LOCK_PG = "FOR UPDATE SKIP LOCKED"


def to_pyformat(sql: str) -> str:
    """``?`` -> ``%s`` (and literal ``%`` -> ``%%``). The shared SQL has no ``?`` in literals."""
    return sql.replace("%", "%%").replace("?", "%s")


class _Conn:
    """Just enough of the sqlite3 connection API (``execute``) for the shared statements."""

    def __init__(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        self._c = conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> psycopg.Cursor[dict[str, Any]]:
        return self._c.execute(to_pyformat(sql), params)


class PostgresJobStore(SqliteJobStore):
    _TRUE_SQL = "TRUE"
    _CLAIM_SQL = build_claim_sql(CLAIM_LOCK_PG)

    @staticmethod
    def _flag(value: bool | int) -> Any:
        return bool(value)

    def __init__(
        self,
        url: str,
        *,
        data_dir: str | Path = ".",
        migrate: bool = True,
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
        """``data_dir`` anchors on-disk uploads/outputs (``<data_dir>/uploads``, ``outputs``)."""
        self.url = psycopg_url(url)
        self.blobs = blobs
        self.rate_limit = rate_limit
        self.rate_window_s = rate_window_s
        data = Path(data_dir)
        self.output_root = Path(output_root) if output_root is not None else data / "outputs"
        self.path = data / "postgres"  # only ``path.parent`` is used (UploadService root)
        self.clock = clock
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.max_stale_attempts = max_stale_attempts
        self._retry_kwargs: dict[str, Any] = {}
        if max_attempts is not None:
            self._retry_kwargs["max_attempts"] = max_attempts
        if backoff_base_s is not None:
            self._retry_kwargs["base_s"] = backoff_base_s
        if migrate:
            upgrade(url)

    @contextmanager
    def _tx(self) -> Iterator[Any]:
        # psycopg's connection context manager commits on success and rolls back on error
        with psycopg.connect(self.url, row_factory=dict_row) as conn:
            yield _Conn(conn)

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._tx() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
