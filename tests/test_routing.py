"""Shadow routing with fake components: deterministic buckets, isolation, report."""

import json
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.config import Config
from countonce.routing import (
    RoutingTable,
    bucket,
    load_log,
    run_with_shadow,
    summarize,
)
from countonce.types import CountResult, Interval, ModelVersions


def _result(count: int, width: int = 1) -> CountResult:
    return CountResult(
        prompt="p",
        count=count,
        interval=Interval(max(count - width, 0), count + width, 0.9, "sensitivity"),
        model_versions=ModelVersions(detector="d", backbone="b", associator="a"),
        input_hash="h",
        timings={"total": 0.1},
    )


class FakeRunner:
    def __init__(self, counts: dict[str, int], fail: set[str] = frozenset()):
        self.counts, self.fail, self.calls = counts, fail, []

    def __call__(self, inputs, prompt, config, **kw):
        self.calls.append((config, kw))
        if config.assoc in self.fail:
            raise RuntimeError("shadow exploded")
        return _result(self.counts[config.assoc])


def _table(tmp_path, percent=100.0, **shadow):
    return RoutingTable(
        live={"detector": "countgd", "backbone": "vggt", "assoc": "v1"},
        shadows=[{"name": "s", "percent": percent, "assoc": "v2", **shadow}],
        log_path=tmp_path / "shadow_log.jsonl",
    )


def test_bucket_deterministic_and_percentage_respected():
    assert bucket("job-1", "s") == bucket("job-1", "s")
    hit = sum(bucket(f"j{i}", "s") < 20 for i in range(2000))
    assert 300 < hit < 500
    t = RoutingTable(shadows=[{"name": "s", "percent": 0}])
    assert all(t.select_shadow(f"j{i}") is None for i in range(50))
    t = RoutingTable(shadows=[{"name": "s", "percent": 100}])
    assert all(t.select_shadow(f"j{i}") is not None for i in range(50))


def test_yaml_load(tmp_path):
    f = tmp_path / "r.yaml"
    f.write_text(
        "model_routing:\n  live: {assoc: v2}\n  shadows:\n    - {name: m3, percent: 5, assoc: v3}\n"
    )
    t = RoutingTable.from_yaml(f)
    assert t.live.assoc == "v2" and t.shadows[0].percent == 5


def test_live_result_returned_and_shadow_logged(tmp_path):
    runner = FakeRunner({"v1": 5, "v2": 7})
    res = run_with_shadow(
        [Path("a.jpg")], "p", _table(tmp_path), config=Config(out_dir=tmp_path / "o"),
        job_id="j1", runner=runner,
    )  # fmt: skip
    assert res.count == 5
    (rec,) = load_log(tmp_path / "shadow_log.jsonl")
    assert rec["job"] == "j1" and rec["counts"] == {"live": 5, "shadow": 7}
    assert rec["intervals"]["shadow"] == [6, 8] and rec["error"] is None
    assert set(rec) >= {"job", "live", "shadow", "counts", "intervals", "timings"}
    live_cfg, live_kw = runner.calls[0]
    sh_cfg, sh_kw = runner.calls[1]
    assert sh_cfg.assoc == "v2" and live_cfg.assoc == "v1"
    assert sh_cfg.out_dir != live_cfg.out_dir and sh_kw["export"] is False
    assert "export" not in live_kw


def test_shadow_failure_leaves_live_output_unchanged(tmp_path):
    ok = run_with_shadow(
        "x", "p", _table(tmp_path / "a", percent=0), job_id="j", runner=FakeRunner({"v1": 5})
    )
    bad_runner = FakeRunner({"v1": 5}, fail={"v2"})
    res = run_with_shadow("x", "p", _table(tmp_path), job_id="j", runner=bad_runner)
    assert res == ok and len(bad_runner.calls) == 2
    (rec,) = load_log(tmp_path / "shadow_log.jsonl")
    assert rec["counts"] == {"live": 5, "shadow": None}
    assert "shadow exploded" in rec["error"]


def test_log_write_failure_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    table = _table(tmp_path)
    table.log_path = blocker / "sub" / "log.jsonl"  # parent is a file -> OSError
    res = run_with_shadow("x", "p", table, job_id="j", runner=FakeRunner({"v1": 3, "v2": 4}))
    assert res.count == 3


def test_live_failure_propagates(tmp_path):
    with pytest.raises(RuntimeError):
        run_with_shadow("x", "p", _table(tmp_path), runner=FakeRunner({}, fail={"v1"}))
    assert not (tmp_path / "shadow_log.jsonl").exists()


def test_matching_stages_share_components_and_cache(tmp_path):
    runner = FakeRunner({"v1": 5, "v2": 5})
    det, rec = object(), object()
    other = object()
    run_with_shadow(
        "x", "p", _table(tmp_path), config=Config(cache_dir=tmp_path / "c"), job_id="j",
        runner=runner, detector=det, reconstructor=rec, shadow_reconstructor=other,
    )  # fmt: skip
    (lc, lk), (sc, sk) = runner.calls
    assert sk["detector"] is det and sk["reconstructor"] is rec  # assoc-only shadow shares both
    assert lc.cache_dir == sc.cache_dir == tmp_path / "c"
    assert json.loads((tmp_path / "shadow_log.jsonl").read_text())["shared_stages"] == [
        "detector",
        "backbone",
    ]
    runner2 = FakeRunner({"v1": 5, "v2": 5})
    run_with_shadow(
        "x", "p", _table(tmp_path, backbone="colmap"), job_id="j2", runner=runner2,
        detector=det, reconstructor=rec, shadow_reconstructor=other,
    )  # fmt: skip
    assert runner2.calls[1][1]["reconstructor"] is other  # backbone differs: not shared


def test_summary_and_cli(tmp_path, capsys):
    table = _table(tmp_path)
    for i in range(6):
        run_with_shadow(
            "x", "p", table, job_id=f"j{i}", runner=FakeRunner({"v1": 10, "v2": 10 + i % 2})
        )
    run_with_shadow("x", "p", table, job_id="jf", runner=FakeRunner({"v1": 1}, fail={"v2"}))
    rep = summarize(load_log(table.log_path), n_boot=200)["s"]
    assert rep["n"] == 7 and rep["n_ok"] == 6 and rep["n_failed"] == 1
    m = rep["mean_abs_delta_count"]
    assert m["mean"] == pytest.approx(0.5) and m["ci95"][0] <= 0.5 <= m["ci95"][1]
    assert rep["interval_overlap"]["mean"] == 1.0
    assert main(["shadow-report", str(table.log_path), "--boot", "100"]) == 0
    assert "mean |d count|" in capsys.readouterr().out
    assert main(["shadow-report", str(tmp_path / "missing.jsonl")]) == 2
