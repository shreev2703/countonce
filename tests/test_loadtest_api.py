"""HTTP API load test: tiny smoke (no lost jobs, no double-queues), CSV schema, config checks."""

import csv
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

from countonce.cli import main  # noqa: E402
from countonce.loadtest_api import CSV_COLUMNS, ApiLoadConfig, run_api_load  # noqa: E402


def _cfg(tmp_path: Path, **kw: object) -> ApiLoadConfig:
    base: dict[str, object] = {
        "n_clients": 2,
        "jobs_per_client": 3,  # 2 clients x 3 = 6 jobs (>= the 5-job smoke)
        "n_users": 2,
        "n_workers": 2,
        "stage_latency_s": {"a": 0.01, "b": 0.02},
        "duplicate_rate": 1.0,  # every job gets a concurrent duplicate create and submit
        "poll_s": 0.05,
        "sse_poll_s": 0.03,
        "timeout_s": 90,
        "work_dir": str(tmp_path / "work"),
        "results_csv": str(tmp_path / "api_load.csv"),
        "summary_json": str(tmp_path / "summary.json"),
    }
    base.update(kw)
    return ApiLoadConfig(**base)  # type: ignore[arg-type]


@pytest.mark.loadtest
def test_smoke_no_lost_jobs_no_double_queue(tmp_path: Path) -> None:
    s = run_api_load(_cfg(tmp_path))
    assert s["failure_samples"] == []
    assert s["jobs_in_db"] == s["jobs_expected"] == 6  # duplicate creates made no extra jobs
    assert s["jobs_done"] == 6 and s["jobs_lost"] == 0
    assert s["double_queued"] == 0
    assert s["dup_pairs"] == 6 and s["dup_replays"] == 12  # one replay each for create and submit
    assert s["reconciled"]
    with (tmp_path / "api_load.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == CSV_COLUMNS
    assert {r["route"] for r in rows} >= {"ALL", "POST /jobs", "POST /jobs/{id}/submit"}
    assert all(r["host"] and r["cpu_count"] for r in rows)


@pytest.mark.loadtest
def test_rate_limit_produces_429_and_still_reconciles(tmp_path: Path) -> None:
    s = run_api_load(
        _cfg(
            tmp_path,
            n_clients=4,
            jobs_per_client=1,
            n_users=1,
            # One user shared by 4 clients: burst 4, refill 4 req/s. The 4 jobs need ~50 requests
            # and the clients can issue them faster than that, so 429s are unavoidable. (The old
            # 20 per 2 s bucket refilled at 10 req/s, about what 2 clients can send on a slow
            # request path, so it only sometimes tripped.)
            request_rate_limit=4,
            request_rate_window_s=1.0,
            retry_after_cap_s=0.2,
            follow="poll",
            duplicate_rate=0.0,
        )
    )
    assert s["routes"]["ALL"]["n_429"] > 0
    assert s["reconciled"], s["failure_samples"]


def test_config_validation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("nope: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        ApiLoadConfig.from_yaml(bad)
    bad.write_text("follow: carrier-pigeon\n")
    with pytest.raises(ValueError, match="follow"):
        ApiLoadConfig.from_yaml(bad)
    assert main(["loadtest-api", "--config", str(tmp_path / "missing.yaml")]) == 2


def test_default_config_loads() -> None:
    cfg = ApiLoadConfig.from_yaml("configs/loadtest_api.yaml")
    assert cfg.n_clients >= 1 and cfg.request_rate_limit
