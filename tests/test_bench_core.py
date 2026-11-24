"""Toy-input tests for bench/: metrics, baselines, statistics, splits and the runner."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from bench import baselines, metrics, runner, splits, stats


def test_count_metrics_toy():
    pred, gt = [10, 22, 5], [10, 20, 10]
    assert metrics.mae(pred, gt) == pytest.approx(7 / 3)
    assert metrics.mape(pred, gt) == pytest.approx(100 * (0 + 0.1 + 0.5) / 3)
    assert metrics.within_rate(pred, gt) == pytest.approx(2 / 3)


def test_mape_skips_zero_gt_and_within_zero():
    assert metrics.mape([3, 5], [0, 10]) == pytest.approx(50.0)
    assert math.isnan(metrics.mape([3], [0]))
    assert list(metrics.within_tolerance([0, 1], [0, 0])) == [True, False]


def test_metric_input_validation():
    with pytest.raises(ValueError):
        metrics.mae([1, 2], [1])
    with pytest.raises(ValueError):
        metrics.interval_coverage([5], [3], [4])


def test_interval_coverage_closed():
    assert metrics.interval_coverage([1, 5, 8], [3, 5, 9], [3, 5, 7]) == pytest.approx(2 / 3)


# Detections: cluster ids vs ground-truth object ids.
CL = ["a", "a", "b", "b", "c", "d"]
GT = [1, 1, 1, 2, 2, None]


def test_pairwise_prf_hand_computed():
    # labelled: (a,1)(a,1)(b,1)(b,2)(c,2). pred pairs: a:1, b:1 -> 2. true pairs: gt1: C(3,2)=3,
    # gt2: C(2,2)=1 -> 4. tp: (a,1)x2 -> 1.
    p, r, f1 = metrics.pairwise_prf(CL, GT)
    assert p == pytest.approx(1 / 2)
    assert r == pytest.approx(1 / 4)
    assert f1 == pytest.approx(2 * 0.5 * 0.25 / 0.75)


def test_pairwise_prf_perfect_and_degenerate():
    assert metrics.pairwise_prf(["x", "x", "y"], [1, 1, 2]) == (1.0, 1.0, 1.0)
    p, r, f1 = metrics.pairwise_prf(["x", "y"], [1, 2])  # nothing positive anywhere
    assert (p, r) == (1.0, 1.0) and f1 == 1.0
    assert metrics.pairwise_prf(["x", "y"], [1, 1]) == (1.0, 0.0, 0.0)  # all split


def test_purity():
    # a: {1:2} b: {1:1,2:1} c: {2:1} -> (2 + 1 + 1) / 5
    assert metrics.cluster_purity(CL, GT) == pytest.approx(4 / 5)
    assert math.isnan(metrics.cluster_purity(["a"], [None]))


def test_duplicate_and_miss_rate():
    # majorities: a->1, b->1 (tie, smallest repr), c->2. Clusters 3, distinct gt 2 -> 1/3 dup.
    assert metrics.duplicate_rate(CL, GT) == pytest.approx(1 / 3)
    assert metrics.miss_rate(CL, GT) == 0.0
    assert metrics.miss_rate(CL, GT, gt_universe=[1, 2, 3, 4]) == pytest.approx(0.5)
    assert math.isnan(metrics.duplicate_rate(["a"], [None]))


def test_baselines():
    c = [4, 7, 5, 6]
    assert baselines.sum_baseline(c) == 22
    assert baselines.max_baseline(c) == 7
    assert baselines.median_baseline(c) == 5.5
    assert set(baselines.BASELINES) == {"sum", "max", "median"}
    assert all(f([]) == 0.0 for f in baselines.BASELINES.values())


def test_bootstrap_ci_properties():
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = stats.bootstrap_ci(v, n_boot=2000, seed=1)
    assert ci.estimate == 3.0 and ci.low <= 3.0 <= ci.high and ci.n == 5
    assert stats.bootstrap_ci(v, n_boot=2000, seed=1) == ci  # seeded
    const = stats.bootstrap_ci([2.0] * 6, n_boot=200)
    assert const.low == const.high == 2.0
    nan = stats.bootstrap_ci([1.0, float("nan"), 3.0], n_boot=200)
    assert nan.n == 2 and nan.estimate == 2.0
    med = stats.bootstrap_ci(v, statistic=np.median, n_boot=500)
    assert med.estimate == 3.0
    with pytest.raises(ValueError):
        stats.bootstrap_ci([float("nan")])


def test_paired_compare_detects_consistent_gap():
    b = np.arange(1.0, 21.0)
    a = b - 2.0 - 0.01 * np.arange(20)  # A uniformly lower error
    r = stats.paired_compare(a, b, n_boot=2000, seed=0)
    assert r.diff < 0 and r.ci_high < 0
    assert r.p_bootstrap < 0.01 and r.p_wilcoxon < 0.01 and r.n == 20


def test_paired_compare_identical_is_null():
    x = [1.0, 2.0, 3.0, 4.0]
    r = stats.paired_compare(x, x, n_boot=200)
    assert r.diff == 0.0 and r.p_wilcoxon == 1.0 and r.p_bootstrap == 1.0
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1.0], [1.0, 2.0])


def _make_scenes(root, names):
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        (d / "gt.json").write_text(json.dumps({"count": len(n)}))
        (d / "img.bin").write_bytes(n.encode() * 3)


def test_scene_hash_sensitive_to_content_and_name(tmp_path):
    _make_scenes(tmp_path, ["s1"])
    h = splits.scene_hash(tmp_path / "s1")
    assert h == splits.scene_hash(tmp_path / "s1")
    (tmp_path / "s1" / "img.bin").write_bytes(b"x")
    assert splits.scene_hash(tmp_path / "s1") != h
    with pytest.raises(FileNotFoundError):
        splits.scene_hash(tmp_path / "nope")


def test_assign_splits_deterministic_and_order_free():
    ids = [f"s{i}" for i in range(10)]
    a = splits.assign_splits(ids, 0.3, seed=7)
    assert a == splits.assign_splits(ids[::-1], 0.3, seed=7)
    assert sum(v == "test" for v in a.values()) == 3
    two = splits.assign_splits(["a", "b"], 0.3)
    assert sorted(two.values()) == ["test", "val"]
    with pytest.raises(ValueError):
        splits.assign_splits(["a", "a"])


def test_splits_write_verify_freeze(tmp_path):
    names = [f"scene{i}" for i in range(6)]
    _make_scenes(tmp_path / "scenes", names)
    hashes = {n: splits.scene_hash(tmp_path / "scenes" / n) for n in names}
    path = tmp_path / "splits.json"
    out = splits.write_splits(hashes, path)
    assert set(out["val"]) | set(out["test"]) == set(names)
    assert not set(out["val"]) & set(out["test"])
    assert splits.verify_splits(hashes, path) == []
    with pytest.raises(FileExistsError):
        splits.write_splits(hashes, path)
    assert splits.split_of(next(iter(out["test"])), path) == "test"
    assert splits.split_of("ghost", path) is None

    tampered = dict(hashes, scene0="0" * 64)
    del tampered["scene1"]
    tampered["new"] = "f" * 64
    problems = "\n".join(splits.verify_splits(tampered, path))
    assert "scene0: hash changed" in problems
    assert "scene1: recorded" in problems and "missing" in problems
    assert "new: not in any split" in problems


def test_verify_flags_overlap(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": 1, "splits": {"val": {"a": "h"}, "test": {"a": "h"}}}))
    assert any("both" in p for p in splits.verify_splits({"a": "h"}, path))


def test_runner_resumes_and_records_provenance(tmp_path):
    csv_path = tmp_path / "out" / "bench.csv"
    jobs = [runner.Job(f"s{i}", "sum", n_views=5, sample="even", config={"k": 1}) for i in range(3)]
    calls: list[str] = []

    def fn(job):
        calls.append(job.scene_id)
        return runner.RunOutput(pred=float(len(calls)), gt=2.0, low=1.0, high=3.0)

    assert runner.run_jobs(jobs[:2], fn, csv_path) == (2, 0, 0)
    assert runner.run_jobs(jobs, fn, csv_path) == (1, 2, 0)
    assert calls == ["s0", "s1", "s2"]
    rows = runner.read_rows(csv_path)
    assert len(rows) == 3 and rows[0]["method"] == "sum" and rows[0]["n_views"] == "5"
    assert rows[0]["timestamp_utc"] and "git_commit" in rows[0]
    assert json.loads(rows[0]["config_json"]) == {"k": 1}
    assert csv_path.read_text().count("run_key") == 1  # header written once


def test_runner_retries_errors_and_config_changes_key(tmp_path):
    csv_path = tmp_path / "bench.csv"
    job = runner.Job("s", "m")
    state = {"fail": True}

    def fn(j):
        if state["fail"]:
            raise RuntimeError("boom\nline2")
        return runner.RunOutput(pred=1.0)

    assert runner.run_jobs([job], fn, csv_path) == (0, 0, 1)
    row = runner.read_rows(csv_path)[0]
    assert row["status"] == "error" and "boom" in row["error"] and "\n" not in row["error"]
    state["fail"] = False
    assert runner.run_jobs([job], fn, csv_path) == (1, 0, 0)
    assert runner.completed_keys(csv_path) == {job.key()}
    assert runner.Job("s", "m", config={"a": 1}).key() != job.key()


def test_runner_rejects_foreign_csv(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        runner.run_jobs([runner.Job("s", "m")], lambda j: runner.RunOutput(pred=0.0), p)
