"""Cost per job and per-stage time by worker kind over a seeded store (exact arithmetic)."""

import json
import sqlite3
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.jobs.store import SqliteJobStore
from countonce.monitor import admin

NOW = 1_700_000_000.0
RATES = {"laptop": 3.6, "cloud": 7.2}  # per hour -> 0.001 / 0.002 per GPU-second


@pytest.fixture
def store(tmp_path: Path) -> SqliteJobStore:
    path = tmp_path / "jobs.db"
    st = SqliteJobStore(path)
    conn = sqlite3.connect(path)
    for wid, kind in (("w1", "laptop"), ("w2", "cloud")):
        conn.execute("INSERT INTO workers (id, kind) VALUES (?, ?)", (wid, kind))

    def job(i, wid, timings, frames):
        end = NOW - 100 + i
        conn.execute(
            "INSERT INTO jobs (id, user_id, prompt, status, created_at, started_at, finished_at,"
            " worker_id) VALUES (?, 'u', 'p', 'done', ?, ?, ?, ?)",
            (f"j{i}", end - 50, end - 40, end, wid),
        )
        doc = {"timings": timings, "per_frame_counts": {str(k): 1 for k in range(frames)}}
        conn.execute(
            "INSERT INTO results (job_id, count, result_json) VALUES (?, 1, ?)",
            (f"j{i}", json.dumps(doc)),
        )

    job(0, "w1", {"recon": 10.0, "detect": 2.0}, 10)  # 12 s
    job(1, "w1", {"recon": 30.0, "detect": 6.0}, 30)  # 36 s
    job(2, "w2", {"recon": 20.0, "detect": 5.0}, 25)  # 25 s
    conn.commit()
    conn.close()
    return st


def test_arithmetic(store: SqliteJobStore) -> None:
    c = admin.compute_metrics(store, NOW, RATES)["cost"]["by_worker_kind"]
    lap = c["laptop"]
    assert lap["n_jobs"] == 2 and lap["gpu_seconds_total"] == 48.0
    assert lap["gpu_seconds_per_job"]["p50"] == 24.0 and lap["gpu_seconds_per_job"]["mean"] == 24.0
    assert lap["cost_per_job_mean"] == pytest.approx(24 * 0.001)
    assert lap["cost_total"] == pytest.approx(0.048)
    assert lap["cost_per_1000_frames"] == pytest.approx(0.048 / 40 * 1000)
    assert lap["stage_time_s"]["recon"] == {"p50": 20.0, "p95": pytest.approx(29.0)}
    cloud = c["cloud"]
    assert cloud["cost_per_job_mean"] == pytest.approx(25 * 0.002)
    assert cloud["cost_per_1000_frames"] == pytest.approx(0.05 / 25 * 1000)
    assert cloud["stage_time_s"]["detect"] == {"p50": 5.0, "p95": 5.0}


def test_null_rate_gives_null_not_zero(store: SqliteJobStore) -> None:
    c = admin.compute_metrics(store, NOW)["cost"]["by_worker_kind"]
    for kind in ("laptop", "cloud"):
        assert c[kind]["cost_per_hour"] is None
        assert c[kind]["cost_per_job_mean"] is None
        assert c[kind]["cost_total"] is None
        assert c[kind]["cost_per_1000_frames"] is None
        assert c[kind]["gpu_seconds_total"] > 0  # GPU time is still reported
    page = admin.render_html(admin.compute_metrics(store, NOW), [])
    assert "TODO: set rate" in page


def test_config_ships_null_rates() -> None:
    assert admin.load_cost_rates("configs/cost.yaml") == {"laptop": None, "cloud": None}


def test_bad_rates_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("cost_per_hour: {tpu: 1}\n")
    with pytest.raises(ValueError, match="unknown worker kind"):
        admin.load_cost_rates(f)
    f.write_text("cost_per_hour: {laptop: -1}\n")
    with pytest.raises(ValueError, match=">= 0"):
        admin.load_cost_rates(f)


def test_cost_alert(store: SqliteJobStore) -> None:
    rule = admin.AlertRule("cost", "cost_per_job", window_s=3600, max_cost=0.03, min_jobs=1)
    got = admin.evaluate_alerts(store, [rule], NOW, RATES)
    # costs: 0.012, 0.036, 0.05 -> mean 0.0326 > 0.03
    assert len(got) == 1 and got[0].value == pytest.approx((0.012 + 0.036 + 0.05) / 3)
    assert got[0].threshold == 0.03
    high = admin.AlertRule("cost", "cost_per_job", max_cost=1.0, min_jobs=1)
    assert admin.evaluate_alerts(store, [high], NOW, RATES) == []
    # null rate or no threshold or too few jobs: quiet, never a 0-cost comparison
    assert admin.evaluate_alerts(store, [rule], NOW, None) == []
    assert admin.evaluate_alerts(store, [admin.AlertRule("c", "cost_per_job")], NOW, RATES) == []
    few = admin.AlertRule("c", "cost_per_job", max_cost=0.0, min_jobs=10)
    assert admin.evaluate_alerts(store, [few], NOW, RATES) == []


def test_shipped_rule_is_disabled() -> None:
    rule = next(r for r in admin.load_rules("configs/alerts.yaml") if r.type == "cost_per_job")
    assert rule.max_cost is None


def test_cli_json_has_cost(store: SqliteJobStore, tmp_path: Path, capsys) -> None:
    assert main(["admin-stats", "--db", str(store.path), "--json"]) == 0
    out = capsys.readouterr().out
    cost = json.loads(out[out.index("{") :])["metrics"]["cost"]
    assert cost["cost_per_hour"] == {"cloud": None, "laptop": None}
    assert cost["by_worker_kind"]["laptop"]["cost_per_job_mean"] is None
