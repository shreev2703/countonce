"""Pilot report: metrics on a seeded store, empty store renders only TODO, CLI writes the file."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.corrections import Correction
from countonce.jobs.store import SqliteJobStore
from countonce.monitor.pilot import TODO, compute_pilot, render_pilot

DAY = 86400.0
T0 = datetime(2026, 3, 2, 12, tzinfo=UTC).timestamp()


def _seed(db: Path) -> None:
    SqliteJobStore(db)
    # (id, user, created offset, duration, site, captured offset, count)
    jobs = [
        ("j1", "alice", 0, 10, None, None, 10),
        ("j2", "alice", DAY, 20, None, None, 20),  # alice: two days -> repeat
        ("j3", "bob", 0, 30, "plot", 0, 30),
        ("j4", "bob", 100, 40, "plot", 50, 40),  # bob: same day, 2 visits -> repeat
        ("j5", "carol", 0, 50, None, None, 50),
        ("j6", "carol", 100, 60, None, None, 60),  # carol: same day, no site -> not repeat
    ]
    conn = sqlite3.connect(db)
    for jid, user, off, dur, site, cap, cnt in jobs:
        conn.execute(
            "INSERT INTO jobs (id, user_id, prompt, status, site_id, captured_at, created_at, "
            "finished_at) VALUES (?, ?, 'x', 'done', ?, ?, ?, ?)",
            (jid, user, site, None if cap is None else T0 + cap, T0 + off, T0 + off + dur),
        )
        conn.execute("INSERT INTO results (job_id, count) VALUES (?, ?)", (jid, cnt))
    conn.commit()
    conn.close()


def test_metrics_on_seeded_store(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    _seed(db)
    corr = [
        Correction("j1", "reject", ("a",)),
        Correction("j1", "reject", ("b",)),
        Correction("j3", "split", ("c",)),
        Correction("nope", "reject", ("d",)),  # unknown job ignored
    ]
    m = compute_pilot(db, corrections=corr)
    assert m["active_users"] == 3
    assert m["jobs_per_user"]["value"] == 2.0
    assert m["repeat_use_rate"]["value"] == pytest.approx(2 / 3)
    assert m["time_p50_s"]["value"] == pytest.approx(35.0)
    assert m["time_p95_s"]["value"] == pytest.approx(57.5)
    assert m["time_p50_s"]["ci"] is not None  # n=6 >= 5
    assert m["jobs_per_user"]["ci"] is None  # n=3 users
    assert m["corrected_share"]["value"] == pytest.approx(2 / 6)
    assert m["rejects_per_job"]["value"] == pytest.approx(2 / 6)
    assert m["corrections_per_instance"]["value"] == pytest.approx(3 / 210)
    since = compute_pilot(db, since=T0 + DAY / 2)
    assert since["n_jobs"] == 1 and since["active_users"] == 1


def test_empty_store_renders_only_todo(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    SqliteJobStore(db)
    text = render_pilot(compute_pilot(db))
    cells = [ln for ln in text.splitlines() if ln.startswith("| ") and "Metric" not in ln]
    assert len(cells) == 8
    for ln in cells:
        assert ln.split("|")[2].strip() == TODO
        assert ln.split("|")[4].strip() == TODO


def test_cli_writes_file(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    _seed(db)
    cj = tmp_path / "c.json"
    cj.write_text(json.dumps([Correction("j1", "reject", ("a",)).to_dict()]))
    out = tmp_path / "docs" / "PILOT.md"
    rc = main(["pilot-report", "--db", str(db), "--corrections", str(cj), "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "Active users | 3" in text and "Repeat-use rate" in text
    assert main(["pilot-report", "--db", str(tmp_path / "missing.db"), "--out", str(out)]) == 2
