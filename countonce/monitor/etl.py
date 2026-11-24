"""OLTP -> OLAP telemetry export (PLAN 16B.1, "OLTP vs. OLAP separation").

``export`` copies completed-job telemetry out of the sqlite job store into date-partitioned
Parquet: ``<out>/<table>/dt=YYYY-MM-DD/part-NNNNN.parquet`` for the tables ``jobs``,
``stage_timings``, ``feedback``, ``corrections`` and ``cost``. ``dt`` is the UTC day of the row's
event time (job finish, feedback / correction creation).

Idempotency: ``<out>/_watermark.json`` keeps, per table, the newest exported event time plus the
ids exported at exactly that instant, so a re-run exports only newer rows (and ties not yet seen).

``ParquetStore`` is a read-only, DuckDB-backed stand-in for a ``JobStore`` so ``monitor.admin``
and ``monitor.drift`` can run on the export instead of the live queue. Queued/running jobs are
never exported, so queue depth and oldest-queued age are 0 / None on that path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from countonce.corrections import load_corrections
from countonce.jobs.store import SqliteJobStore
from countonce.monitor.drift import row_from_result

TABLES = ("jobs", "stage_timings", "feedback", "corrections", "cost")
COMPLETED = ("done", "failed", "dead_letter")
WATERMARK = "_watermark.json"

_S, _F, _I = pa.string(), pa.float64(), pa.int64()
SCHEMAS: dict[str, pa.Schema] = {
    "jobs": pa.schema(
        [
            ("id", _S),
            ("user_id", _S),
            ("status", _S),
            ("error_code", _S),
            ("worker_id", _S),
            ("worker_kind", _S),
            ("created_at", _F),
            ("started_at", _F),
            ("finished_at", _F),
            ("ts", _F),
            ("dt", _S),
            ("recon_cache_hit", _I),
            ("oom_fallback_rung", _I),
            ("peak_vram_mb", _F),
            ("has_result", pa.bool_()),
            ("result_json", _S),
            ("drift_json", _S),
        ]  # fmt: skip
    ),
    "stage_timings": pa.schema(
        [("job_id", _S), ("stage", _S), ("seconds", _F), ("ts", _F), ("dt", _S), ("id", _S)]
    ),
    "feedback": pa.schema(
        [
            ("id", _S),
            ("job_id", _S),
            ("user_id", _S),
            ("correct", _I),
            ("user_count", _I),
            ("comment", _S),
            ("created_at", _F),
            ("ts", _F),
            ("dt", _S),
        ]  # fmt: skip
    ),
    "corrections": pa.schema(
        [
            ("id", _S),
            ("job_id", _S),
            ("action", _S),
            ("n_detections", _I),
            ("user", _S),
            ("consent", pa.bool_()),
            ("ts", _F),
            ("dt", _S),
        ]  # fmt: skip
    ),
    "cost": pa.schema(
        [
            ("id", _S),
            ("job_id", _S),
            ("worker_kind", _S),
            ("gpu_s", _F),
            ("frames", _I),
            ("cost_per_hour", _F),
            ("cost", _F),
            ("ts", _F),
            ("dt", _S),
        ]  # fmt: skip
    ),
}


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


def parse_since(value: str | float | None) -> float | None:
    """ISO date/time (naive = UTC) or epoch seconds -> epoch seconds."""
    if value is None or isinstance(value, float | int):
        return None if value is None else float(value)
    try:
        return float(value)
    except ValueError:
        pass
    dt = datetime.fromisoformat(value)
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


@dataclass
class ExportSummary:
    rows: dict[str, int]
    files: int
    out: Path

    @property
    def total(self) -> int:
        return sum(self.rows.values())


def _load_watermark(out: Path) -> dict[str, dict[str, Any]]:
    p = out / WATERMARK
    return json.loads(p.read_text()) if p.exists() else {}


def _usage(raw: str | None) -> tuple[dict[str, float], int | None, dict[str, Any] | None]:
    if not raw:
        return {}, None, None
    doc = json.loads(raw)
    timings = {s: float(t) for s, t in (doc.get("timings") or {}).items()}
    frames = doc.get("n_frames")
    if frames is None and isinstance(doc.get("per_frame_counts"), Mapping):
        frames = len(doc["per_frame_counts"])
    return timings, (None if frames is None else int(frames)), doc


def _collect(
    store: SqliteJobStore,
    corrections: Path | None,
    rates: Mapping[str, float | None],
) -> dict[str, list[dict[str, Any]]]:
    """Every exportable row per table (before watermark filtering), each with ``id`` and ``ts``."""
    import sqlite3

    kinds: dict[str, str] = {}
    fb: list[dict[str, Any]] = []
    conn = sqlite3.connect(store.path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            kinds = {r["id"]: r["kind"] for r in conn.execute("SELECT id, kind FROM workers")}
        except sqlite3.OperationalError:
            pass
        try:
            fb = [dict(r) for r in conn.execute("SELECT * FROM feedback")]
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()

    out: dict[str, list[dict[str, Any]]] = {t: [] for t in TABLES}
    for j in store.list_jobs():
        if j["status"] not in COMPLETED:
            continue
        ts = float(j["finished_at"] if j.get("finished_at") is not None else j["created_at"])
        kind = kinds.get(j.get("worker_id") or "", "unknown")
        res = store.get_result(j["id"]) if j["status"] == "done" else None
        timings, frames, doc = _usage(res.get("result_json") if res else None)
        slim = json.dumps({"timings": timings, "n_frames": frames}) if res else None
        drift = None
        if doc is not None:
            drift = json.dumps(row_from_result(doc), default=lambda _: None, allow_nan=True)
        out["jobs"].append(
            {
                "id": j["id"], "user_id": j["user_id"], "status": j["status"],
                "error_code": j.get("error_code"), "worker_id": j.get("worker_id"),
                "worker_kind": kind, "created_at": j["created_at"],
                "started_at": j.get("started_at"), "finished_at": j.get("finished_at"),
                "ts": ts, "dt": day_of(ts),
                "recon_cache_hit": int(res["recon_cache_hit"]) if res else None,
                "oom_fallback_rung": int(res["oom_fallback_rung"]) if res else None,
                "peak_vram_mb": float(res["peak_vram_mb"]) if res else None,
                "has_result": res is not None, "result_json": slim, "drift_json": drift,
            }
        )  # fmt: skip
        if res:
            for stage, sec in timings.items():
                out["stage_timings"].append(
                    {"id": f"{j['id']}:{stage}", "job_id": j["id"], "stage": stage,
                     "seconds": sec, "ts": ts, "dt": day_of(ts)}
                )  # fmt: skip
            gpu = sum(timings.values())
            rate = rates.get(kind)
            out["cost"].append(
                {"id": j["id"], "job_id": j["id"], "worker_kind": kind, "gpu_s": gpu,
                 "frames": frames, "cost_per_hour": rate,
                 "cost": None if rate is None else gpu / 3600.0 * rate,
                 "ts": ts, "dt": day_of(ts)}
            )  # fmt: skip
    for r in fb:
        ts = float(r["created_at"] or 0.0)
        out["feedback"].append({**r, "ts": ts, "dt": day_of(ts)})
    if corrections is not None:
        for c in load_corrections(corrections):
            when = datetime.fromisoformat(c.timestamp)
            ts = (when if when.tzinfo else when.replace(tzinfo=UTC)).timestamp()
            out["corrections"].append(
                {"id": c.key, "job_id": c.job_id, "action": c.action,
                 "n_detections": len(c.detection_ids), "user": c.user, "consent": c.consent,
                 "ts": ts, "dt": day_of(ts)}
            )  # fmt: skip
    return out


def export(
    db: str | Path,
    out: str | Path,
    since: str | float | None = None,
    corrections: str | Path | None = None,
    cost_per_hour: Mapping[str, float | None] | None = None,
) -> ExportSummary:
    """Export new completed-job telemetry from ``db`` to Parquet under ``out`` (idempotent)."""
    out_dir = Path(out)
    store = SqliteJobStore(Path(db))
    lo = parse_since(since)
    collected = _collect(store, Path(corrections) if corrections else None, cost_per_hour or {})
    wm = _load_watermark(out_dir)
    counts: dict[str, int] = {}
    files = 0
    for table in TABLES:
        mark = wm.get(table, {"ts": None, "ids": []})
        seen = set(mark["ids"])
        floor = mark["ts"]
        new = [
            r
            for r in collected[table]
            if (lo is None or r["ts"] >= lo)
            and (floor is None or r["ts"] > floor or (r["ts"] == floor and r["id"] not in seen))
        ]
        new.sort(key=lambda r: (r["ts"], r["id"]))
        counts[table] = len(new)
        if not new:
            continue
        by_day: dict[str, list[dict[str, Any]]] = {}
        for r in new:
            by_day.setdefault(r["dt"], []).append(r)
        for day, rows in sorted(by_day.items()):
            part_dir = out_dir / table / f"dt={day}"
            part_dir.mkdir(parents=True, exist_ok=True)
            n = len(list(part_dir.glob("part-*.parquet")))
            schema = SCHEMAS[table]
            tbl = pa.Table.from_pylist(
                [{k: r.get(k) for k in schema.names} for r in rows], schema=schema
            )
            pq.write_table(tbl, part_dir / f"part-{n:05d}.parquet")
            files += 1
        top = new[-1]["ts"]
        ids = {r["id"] for r in new if r["ts"] == top}
        if floor == top:
            ids |= seen
        wm[table] = {"ts": top, "ids": sorted(ids)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / WATERMARK).write_text(json.dumps(wm, indent=2, sort_keys=True))
    return ExportSummary(counts, files, out_dir)


# ---------------------------------------------------------------- DuckDB read path


def is_export(path: str | Path) -> bool:
    return any((Path(path) / "jobs").glob("dt=*"))


def read_table(
    export_dir: str | Path,
    table: str,
    since: float | None = None,
    until: float | None = None,
) -> list[dict[str, Any]]:
    """Rows of ``table`` from the export via DuckDB, ordered by ``ts`` then ``id``/job."""
    import duckdb

    root = Path(export_dir) / table
    if not list(root.glob("dt=*/part-*.parquet")):
        return []
    conds, params = [], []
    if since is not None:
        conds.append("ts >= ?")
        params.append(since)
    if until is not None:
        conds.append("ts < ?")
        params.append(until)
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    glob = str(root / "dt=*" / "part-*.parquet")
    con = duckdb.connect(":memory:")
    try:
        cur = con.execute(
            "SELECT * FROM read_parquet(?, union_by_name = true, hive_partitioning = false) "
            f"{where} ORDER BY ts, 1",
            [glob, *params],
        )
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
    finally:
        con.close()


class ParquetStore:
    """Read-only ``JobStore`` subset over an export directory (DuckDB), for admin / drift."""

    path = None

    def __init__(
        self, export_dir: str | Path, since: float | None = None, until: float | None = None
    ) -> None:
        self.export_dir = Path(export_dir)
        if not self.export_dir.is_dir():
            raise FileNotFoundError(f"no such export directory: {self.export_dir}")
        rows = read_table(self.export_dir, "jobs", since, until)
        self._jobs = sorted(rows, key=lambda r: (r["created_at"], r["id"]))
        self._results = {r["id"]: r for r in self._jobs if r["has_result"]}
        self._feedback = read_table(self.export_dir, "feedback", since, until)

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        return [dict(j) for j in self._jobs if status is None or j["status"] == status]

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        r = self._results.get(job_id)
        if r is None:
            return None
        return {k: r[k] for k in ("recon_cache_hit", "oom_fallback_rung", "peak_vram_mb")} | {
            "job_id": job_id,
            "result_json": r["result_json"],
        }

    def worker_kinds(self) -> dict[str, str]:
        return {j["worker_id"]: j["worker_kind"] for j in self._jobs if j["worker_id"]}

    def feedback_accuracy(self) -> dict[str, Any] | None:
        n = len(self._feedback)
        if not n:
            return None
        return {"n": n, "accuracy": sum(float(r["correct"]) for r in self._feedback) / n}


def load_drift_rows(
    export_dir: str | Path, since: float | None = None, until: float | None = None
) -> list[dict[str, Any]]:
    """Drift feature rows of done jobs in the export (``drift_json`` column)."""
    rows = []
    for r in read_table(export_dir, "jobs", since, until):
        if r["status"] == "done" and r.get("drift_json"):
            rows.append(json.loads(r["drift_json"]))
    return rows


__all__: Sequence[str] = ("export", "ParquetStore", "load_drift_rows", "is_export", "read_table")
