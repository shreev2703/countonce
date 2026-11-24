"""Queue load-test harness: tiny run reconciles, CSV schema is stable."""

import csv
import json
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.loadtest import CSV_COLUMNS, LoadTestConfig, append_csv, run_load_test


def _tiny(tmp_path: Path, **kw: object) -> LoadTestConfig:
    base: dict[str, object] = {
        "n_jobs": 30,
        "n_submitters": 2,
        "n_workers": 3,
        "stage_latency_s": {"a": 0.01, "b": 0.02},
        "fail_rate": 0.1,
        "crash_rate": 1.0,
        "max_crashes": 1,
        "heartbeat_interval_s": 0.1,
        "heartbeat_timeout_s": 0.8,
        "timeout_s": 60,
        "work_dir": str(tmp_path / "work"),
        "results_csv": str(tmp_path / "load_test.csv"),
        "summary_json": str(tmp_path / "summary.json"),
    }
    base.update(kw)
    return LoadTestConfig(**base)  # type: ignore[arg-type]


@pytest.mark.loadtest
def test_tiny_run_reconciles(tmp_path: Path) -> None:
    cfg = _tiny(tmp_path)
    s = run_load_test(cfg)
    assert s["crashes"] == 1
    assert s["submitted"] == 30
    assert s["unfinished"] == 0 and s["failed"] == 0
    assert s["submitted"] == s["done"] + s["dead_letter"]  # nothing lost
    assert s["duplicate_completions"] == 0  # nothing done twice
    assert s["reconciled"] is True
    assert s["completed_jps"] > 0 and s["queue_wait_p95_s"] >= s["queue_wait_p50_s"]
    assert json.loads(Path(cfg.summary_json).read_text())["run_id"] == s["run_id"]
    with Path(cfg.results_csv).open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and list(rows[0]) == CSV_COLUMNS


def test_csv_schema_stable(tmp_path: Path) -> None:
    p = tmp_path / "r.csv"
    append_csv(p, {"run_id": "a", "extra_key": 1})
    append_csv(p, {"run_id": "b"})
    lines = p.read_text().splitlines()
    assert lines[0].split(",") == CSV_COLUMNS and len(lines) == 3
    p.write_text("run_id,other\n1,2\n")
    with pytest.raises(ValueError, match="different header"):
        append_csv(p, {"run_id": "c"})


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("n_jobz: 3\n")
    with pytest.raises(ValueError, match="unknown loadtest"):
        LoadTestConfig.from_yaml(f)
    assert main(["loadtest", "--config", str(f)]) == 2


def test_shipped_config_loads() -> None:
    cfg = LoadTestConfig.from_yaml("configs/loadtest.yaml")
    assert cfg.n_jobs > 0
