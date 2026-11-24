"""Admin metrics and alerts over a seeded sqlite job store (exact values)."""

import dataclasses
import json
import sqlite3
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.jobs.store import SqliteJobStore
from countonce.monitor import admin

DAY = 86400.0
T0 = 1_700_000_000.0 - (1_700_000_000.0 % DAY)  # a UTC midnight
NOW = T0 + DAY + 3 * 3600.0


def _seed(path: Path) -> SqliteJobStore:
    store = SqliteJobStore(path)
    conn = sqlite3.connect(path)

    def job(i, status, created, start=None, end=None, code=None):
        conn.execute(
            "INSERT INTO jobs (id, user_id, prompt, status, error_code, created_at, started_at,"
            " finished_at) VALUES (?, 'u', 'p', ?, ?, ?, ?, ?)",
            (f"j{i}", status, code, created, start, end),
        )

    def result(i, hit, rung, vram, timings):
        conn.execute(
            "INSERT INTO results (job_id, count, result_json, recon_cache_hit, oom_fallback_rung,"
            " peak_vram_mb) VALUES (?, 3, ?, ?, ?, ?)",
            (f"j{i}", json.dumps({"timings": timings}), hit, rung, vram),
        )

    # Day 0: four done jobs with durations 10, 20, 30, 40 s (baseline window).
    for i, dur in enumerate([10, 20, 30, 40]):
        s = NOW - 20000 + 100 * i
        job(i, "done", s, s, s + dur)
        result(i, int(i < 2), [0, 0, 1, 2][i], 1000.0 * (i + 1), {"recon": dur / 2, "detect": 1.0})
    # Day 2, last hour: 3 done at 100 s, 1 dead letter, 1 failed (recent window).
    for k in range(3):
        s = NOW - 3000 + 10 * k
        job(10 + k, "done", s - 5, s, s + 100)
        result(10 + k, 0, 0, 500.0, {"recon": 50.0})
    job(20, "dead_letter", NOW - 2000, NOW - 1990, NOW - 1900, "ERR_INVALID_INPUT")
    job(21, "failed", NOW - 1800, NOW - 1790, NOW - 1700, "ERR_WORKER_TIMEOUT")
    # Two queued jobs: ages 600 s and 1800 s.
    job(30, "queued", NOW - 600)
    job(31, "queued", NOW - 1800)
    conn.commit()
    conn.close()
    return store


@pytest.fixture
def store(tmp_path: Path) -> SqliteJobStore:
    return _seed(tmp_path / "jobs.db")


def test_exact_metrics(store: SqliteJobStore) -> None:
    m = admin.compute_metrics(store, NOW)
    assert m["n_jobs"] == 11
    assert m["by_status"] == {"done": 7, "dead_letter": 1, "failed": 1, "queued": 2}
    assert m["jobs_per_day"] == {
        admin.datetime.fromtimestamp(T0, admin.UTC).strftime("%Y-%m-%d"): 4,
        admin.datetime.fromtimestamp(T0 + DAY, admin.UTC).strftime("%Y-%m-%d"): 7,
    }
    assert m["success_rate"] == pytest.approx(7 / 9)
    assert m["dead_letter_rate"] == pytest.approx(1 / 9)
    assert m["dead_letter_by_code"] == {"ERR_INVALID_INPUT": 1}
    # totals: 10,20,30,40,100,100,100 -> p50 = 40, p90 = 100, p95 = 100
    assert m["time_s"]["total"] == {"p50": 40.0, "p90": 100.0, "p95": 100.0}
    assert m["stage_time_s"]["detect"] == {"p50": 1.0, "p90": 1.0, "p95": 1.0}
    assert m["stage_time_s"]["recon"]["p50"] == 20.0  # 5,10,15,20,50,50,50
    assert m["peak_vram_mb"] == 4000.0
    assert m["recon_cache_hit_rate"] == pytest.approx(2 / 7)
    assert m["oom_rung_histogram"] == {"0": 5, "1": 1, "2": 1}
    assert m["queue_depth"] == 2
    assert m["oldest_queued_age_s"] == 1800.0
    assert m["feedback"] is None


def test_feedback_accuracy_once_rows_exist(store: SqliteJobStore) -> None:
    conn = sqlite3.connect(store.path)
    for i, ok in enumerate([1, 1, 0, 1]):
        conn.execute(
            "INSERT INTO feedback (id, job_id, correct) VALUES (?, 'j0', ?)", (f"f{i}", ok)
        )
    conn.commit()
    conn.close()
    assert admin.compute_metrics(store, NOW)["feedback"] == {"n": 4, "accuracy": 0.75}


def test_empty_store_has_no_rates(tmp_path: Path) -> None:
    m = admin.compute_metrics(SqliteJobStore(tmp_path / "e.db"), NOW)
    assert m["success_rate"] is None and m["oldest_queued_age_s"] is None
    assert m["time_s"]["total"]["p95"] is None


def test_alerts_fire_and_stay_quiet(store: SqliteJobStore) -> None:
    rules = [
        dataclasses.replace(r, min_jobs=3) if r.type == "p95_regression" else r
        for r in admin.load_rules("countonce/configs/alerts.yaml")
    ]
    alerts = {a.rule: a for a in admin.evaluate_alerts(store, rules, NOW)}
    # recent: 3 done + 2 failed -> 40% vs baseline 0% -> fires
    assert alerts["failure_rate_spike"].value == pytest.approx(0.4)
    assert alerts["failure_rate_spike"].severity == "critical"
    # oldest queued 1800 s > 900 s
    assert alerts["queue_age"].value == 1800.0 and alerts["queue_age"].threshold == 900.0
    # recent p95 = 100 s > 1.5 * baseline p95 (38.5 s)
    assert alerts["p95_regression"].value == pytest.approx(100.0)
    assert alerts["p95_regression"].threshold == pytest.approx(1.5 * 38.5)

    loose = [
        admin.AlertRule("a", "failure_rate_spike", min_jobs=10),
        admin.AlertRule("b", "queue_age", max_age_s=3600.0),
        admin.AlertRule("c", "p95_regression", min_jobs=10),
    ]
    assert admin.evaluate_alerts(store, loose, NOW) == []


def test_bad_rule_rejected(tmp_path: Path) -> None:
    f = tmp_path / "a.yaml"
    f.write_text("rules:\n  - {id: x, type: nope}\n")
    with pytest.raises(ValueError, match="unknown type"):
        admin.load_rules(f)


def test_html_and_cli(store: SqliteJobStore, tmp_path: Path, capsys) -> None:
    html_path = tmp_path / "admin.html"
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    for d, vals in ((ref, [1, 2, 3, 4]), (cur, [1, 2, 3, 4])):
        for i, n in enumerate(vals):
            (d / f"r{i}").mkdir(parents=True)
            (d / f"r{i}" / "result.json").write_text(
                json.dumps({"prompt": "apple", "per_frame_counts": {str(k): 1 for k in range(n)}})
            )
    rc = main(
        [
            "admin-stats",
            "--db",
            str(store.path),
            "--json",
            "--html",
            str(html_path),
            "--reference",
            str(ref),
            "--current",
            str(cur),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert payload["metrics"]["queue_depth"] == 2
    assert {a["rule"] for a in payload["alerts"]} >= {"queue_age"}
    assert payload["drift"]["status"] == "ok"
    page = html_path.read_text()
    assert "CountOnce admin" in page and "queue_age" in page and "PSI" in page


def test_cli_missing_db(tmp_path: Path) -> None:
    assert main(["admin-stats", "--db", str(tmp_path / "nope.db")]) == 2
