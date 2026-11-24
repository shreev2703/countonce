"""OLTP -> OLAP export: parity with the live store, idempotent re-runs, day-boundary partitions."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.jobs.store import SqliteJobStore
from countonce.monitor import admin, drift, etl

pytest.importorskip("duckdb")

DAY1 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC).timestamp()
MIDNIGHT = datetime(2026, 3, 2, 0, 0, tzinfo=UTC).timestamp()
NOW = MIDNIGHT + 86400.0
RATES = {"laptop": 3.6, "cloud": 7.2}


def _add_job(conn, i, end, status="done", wid="w1", timings=None, frames=10):
    conn.execute(
        "INSERT INTO jobs (id, user_id, prompt, status, created_at, started_at, finished_at,"
        " worker_id, error_code) VALUES (?, 'u', 'p', ?, ?, ?, ?, ?, ?)",
        (f"j{i}", status, end - 50, end - 40, end, wid, "ERR_X" if status != "done" else None),
    )
    if status == "done":
        doc = {
            "timings": timings or {"recon": 10.0 + i, "detect": 2.0},
            "per_frame_counts": {str(k): 1 for k in range(frames)},
            "detection_status": {f"d{k}": "ok" for k in range(3)},
            "config": {"lift_fallbacks": 1},
            "prompt": "chair",
        }
        conn.execute(
            "INSERT INTO results (job_id, count, result_json, recon_cache_hit, oom_fallback_rung,"
            " peak_vram_mb) VALUES (?, 1, ?, ?, ?, ?)",
            (f"j{i}", json.dumps(doc), i % 2, i % 3, 1000.0 + i),
        )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "jobs.db"
    SqliteJobStore(path)
    conn = sqlite3.connect(path)
    for wid, kind in (("w1", "laptop"), ("w2", "cloud")):
        conn.execute("INSERT INTO workers (id, kind) VALUES (?, ?)", (wid, kind))
    _add_job(conn, 0, DAY1)
    _add_job(conn, 1, DAY1 + 60, wid="w2")
    _add_job(conn, 2, MIDNIGHT - 1)  # last second of day 1
    _add_job(conn, 3, MIDNIGHT)  # first instant of day 2
    _add_job(conn, 4, MIDNIGHT + 500, status="failed", wid="w2")
    _add_job(conn, 5, MIDNIGHT + 900, status="dead_letter")
    for k, (ts, ok) in enumerate([(DAY1, 1), (MIDNIGHT + 5, 0), (MIDNIGHT + 6, 1)]):
        conn.execute(
            "INSERT INTO feedback (id, job_id, user_id, correct, created_at)"
            " VALUES (?, 'j0', 'u', ?, ?)",
            (f"f{k}", ok, ts),
        )
    conn.commit()
    conn.close()
    return path


def _n(out: Path, table: str) -> int:
    return len(etl.read_table(out, table))


def test_export_matches_live_metrics(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "olap"
    summary = etl.export(db, out, cost_per_hour=RATES)
    assert summary.rows["jobs"] == 6 and summary.rows["feedback"] == 3
    live = admin.compute_metrics(SqliteJobStore(db), NOW, RATES)
    olap = admin.compute_metrics(etl.ParquetStore(out), NOW, RATES)
    assert olap == live
    assert admin.evaluate_alerts(etl.ParquetStore(out), [], NOW, RATES) == []
    assert live["feedback"]["n"] == 3


def test_cost_rows_and_null_rate(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "olap"
    etl.export(db, out, cost_per_hour=RATES)
    rows = {r["job_id"]: r for r in etl.read_table(out, "cost")}
    assert rows["j0"]["gpu_s"] == 12.0 and rows["j0"]["cost"] == pytest.approx(12 * 0.001)
    assert rows["j1"]["cost"] == pytest.approx(13 * 0.002)
    out2 = tmp_path / "olap2"
    etl.export(db, out2)
    assert all(r["cost"] is None for r in etl.read_table(out2, "cost"))


def test_rerun_adds_no_rows_and_picks_up_new(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "olap"
    etl.export(db, out)
    before = {t: _n(out, t) for t in etl.TABLES}
    files = sorted(p.name for p in out.rglob("*.parquet"))
    again = etl.export(db, out)
    assert again.total == 0 and again.files == 0
    assert {t: _n(out, t) for t in etl.TABLES} == before
    assert sorted(p.name for p in out.rglob("*.parquet")) == files

    conn = sqlite3.connect(db)
    _add_job(conn, 6, MIDNIGHT + 1000)
    _add_job(conn, 7, MIDNIGHT + 900)  # same instant as an already exported job (tie)
    conn.commit()
    conn.close()
    third = etl.export(db, out)
    assert third.rows["jobs"] == 2
    ids = [r["id"] for r in etl.read_table(out, "jobs")]
    assert len(ids) == len(set(ids)) == 8


def test_partitions_split_across_day_boundary(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "olap"
    etl.export(db, out)
    d1, d2 = out / "jobs" / "dt=2026-03-01", out / "jobs" / "dt=2026-03-02"
    assert d1.is_dir() and d2.is_dir()
    assert sorted(p.name for p in (out / "jobs").iterdir()) == ["dt=2026-03-01", "dt=2026-03-02"]
    by_day = {
        d: sorted(r["id"] for r in etl.read_table(out, "jobs") if r["dt"] == d)
        for d in ("2026-03-01", "2026-03-02")
    }
    assert by_day["2026-03-01"] == ["j0", "j1", "j2"]
    assert by_day["2026-03-02"] == ["j3", "j4", "j5"]
    assert list(d1.glob("part-*.parquet")) and list(d2.glob("part-*.parquet"))
    assert (out / "feedback" / "dt=2026-03-01").is_dir()


def test_since_and_corrections(db: Path, tmp_path: Path) -> None:
    corr = tmp_path / "c.json"
    corr.write_text(
        json.dumps(
            [
                {
                    "job_id": "j3",
                    "action": "reject",
                    "detection_ids": ["a"],
                    "consent": True,
                    "timestamp": "2026-03-02T01:00:00+00:00",
                },
                {
                    "job_id": "j0",
                    "action": "merge",
                    "detection_ids": ["a", "b"],
                    "timestamp": "2026-03-01T13:00:00",
                },
            ]
        )
    )
    out = tmp_path / "olap"
    s = etl.export(db, out, since="2026-03-02", corrections=corr)
    assert s.rows["jobs"] == 3 and s.rows["corrections"] == 1
    assert not (out / "jobs" / "dt=2026-03-01").exists()
    # without --since the older correction now arrives, once
    assert etl.export(db, out, corrections=corr).rows["corrections"] == 0  # behind watermark


def test_drift_reads_export(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "olap"
    etl.export(db, out)
    rows = drift.load_rows(out)
    assert len(rows) == 4 and rows[0]["prompt"] == "chair"
    assert rows[0]["detection_density"] == pytest.approx(0.3)
    rep = drift.run_drift(out, out)
    assert rep["status"] == "ok"


def test_cli(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "olap"
    assert main(["etl", "export", "--db", str(db), "--out", str(out), "--since", "2026-01-01"]) == 0
    assert "exported" in capsys.readouterr().out
    assert main(["etl", "export", "--db", str(db), "--out", str(out)]) == 0
    assert "exported 0 new rows" in capsys.readouterr().out
    assert main(["admin-stats", "--export", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["n_jobs"] == 6
    assert main(["admin-stats", "--db", str(db), "--json"]) == 0
    assert main(["etl", "export", "--db", str(tmp_path / "nope.db"), "--out", str(out)]) == 2
