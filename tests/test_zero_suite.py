"""Zero-count / distractor suite on CPU with the fake plane reconstructor (PLAN 9.8b)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from bench.runner import COLUMNS, Job, RunOutput, read_rows, run_jobs
from bench.sweep import BenchConfig, build_jobs, run_bench
from bench.zero_suite import (
    build_zero_jobs,
    generate_scenes,
    make_zero_executor,
    run_zero_suite,
    summarize_zero,
    wrong_prompt_scenes,
    zero_summary_markdown,
)


def _rows(tmp_path: Path, options: dict, n: int = 6, max_score: float = 0.45):
    scenes = generate_scenes(tmp_path / "scenes", n, seed=1, max_distractor_score=max_score)
    jobs = build_zero_jobs("m", options, scenes, "test", 1)
    fn = make_zero_executor(scenes, tmp_path / "runs")
    csv_path = tmp_path / "bench.csv"
    run_jobs(jobs, fn, csv_path)
    return scenes, read_rows(csv_path)


def test_generator_is_seeded_and_has_both_kinds(tmp_path):
    a = generate_scenes(tmp_path / "a", 10, seed=3)
    b = generate_scenes(tmp_path / "b", 10, seed=3)
    assert [s["gt"] for s in a] == [s["gt"] for s in b]
    assert {s["gt"]["synthetic_zero"] for s in a} == {"empty", "distractor"}
    assert all(s["gt"]["count"] == 0 for s in a)


@pytest.mark.parametrize("assoc", ["v1", "v2"])
def test_empty_scene_false_positive_rate_zero(tmp_path, assoc):
    scenes = [s for s in generate_scenes(tmp_path / "s", 4, seed=0) if s["gt"]["script"] == {}]
    jobs = build_zero_jobs("m", {"assoc": assoc}, scenes, "test", 0)
    run_jobs(jobs, make_zero_executor(scenes, tmp_path / "r"), tmp_path / "b.csv")
    (row,) = summarize_zero(tmp_path / "b.csv")
    assert row["false_positive_rate"][0] == 0.0
    assert row["mean_false_count"][0] == 0.0
    assert row["interval_includes_zero_rate"][0] == 1.0


def test_distractor_only_high_threshold_gives_zero_and_interval_includes_zero(tmp_path):
    # scores are <= 0.45 and the threshold is 0.5, so every look-alike is filtered out
    _, rows = _rows(tmp_path, {"assoc": "v2", "score_threshold": 0.5})
    assert len(rows) == 6 and all(r["status"] == "ok" and r["gt"] == "0.0" for r in rows)
    assert all(float(r["pred"]) == 0 and float(r["low"]) <= 0 <= float(r["high"]) for r in rows)
    (s,) = summarize_zero(tmp_path / "bench.csv")
    assert s["false_positive_rate"][0] == 0.0 and s["interval_includes_zero_rate"][0] == 1.0
    assert "| m | 6 |" in zero_summary_markdown([s])


def test_low_threshold_exposes_false_positives(tmp_path):
    _, rows = _rows(tmp_path, {"assoc": "v1", "score_threshold": 0.0}, n=6, max_score=0.45)
    (s,) = summarize_zero(tmp_path / "bench.csv")
    assert s["false_positive_rate"][0] > 0 and s["mean_false_count"][0] > 0


def test_wrong_prompt_scenes_from_gt(tmp_path):
    gt = {"count": 12, "prompt": "apples", "wrong_prompts": ["oranges", "red bikes"]}
    real = [{"id": "orchard", "dir": tmp_path, "tier": "easy", "gt": gt}]
    wp = wrong_prompt_scenes(real)
    assert [w["id"] for w in wp] == ["orchard@wrong-oranges", "orchard@wrong-red-bikes"]
    assert all(w["gt"]["count"] == 0 for w in wp) and wp[0]["gt"]["prompt"] == "oranges"
    assert wrong_prompt_scenes([{"id": "x", "dir": tmp_path, "gt": {"count": 1}}]) == []


def test_sweep_entry_appends_rows_and_summary(tmp_path):
    cfg = BenchConfig(
        scenes_dir=tmp_path / "none",
        csv=tmp_path / "bench.csv",
        runs_dir=tmp_path / "runs",
        cache_dir=None,
        views=[3],
        random_samples=0,
        methods=[
            {
                "name": "zero_v1",
                "kind": "zero_suite",
                "config": {
                    "options": {"assoc": "v1", "score_threshold": 0.5},
                    "n_synthetic": 10,
                    "scenes_dir": str(tmp_path / "zs"),
                },
            }
        ],
    )
    assert build_jobs(cfg, [{"id": "s"}]) == []  # not part of the view grid
    ran, _, err = run_bench(cfg, [])
    assert ran == 10 and err == 0
    rows = read_rows(cfg.csv)
    assert {r["gt"] for r in rows} == {"0.0"} and {r["domain"] for r in rows} == {"zero_suite"}
    assert (tmp_path / "zero_suite.md").read_text().count("zero_v1") == 1
    assert run_bench(cfg, [])[0] == 0  # resumable


def test_run_zero_suite_wrong_prompt_uses_injected_backends(tmp_path):
    from countonce.detect.base import FakeDetector
    from tests.test_pipeline import PlaneRecon, make_scene

    img_dir, _ = make_scene(tmp_path)
    scene_dir = tmp_path / "real"
    scene_dir.mkdir()
    (img_dir).rename(scene_dir / "images")
    gt = {"count": 5, "prompt": "widgets", "wrong_prompts": ["bikes"]}
    real = [{"id": "real", "dir": scene_dir, "tier": "easy", "gt": gt}]
    ran, _, err = run_zero_suite(
        tmp_path / "b.csv", "m", {"assoc": "v1"}, tmp_path / "zs", real, n_synthetic=0,
        runs_dir=tmp_path / "runs", n_views=4,
        detector_factory=lambda c: FakeDetector({}), reconstructor_factory=lambda c: PlaneRecon(),
    )  # fmt: skip
    assert (ran, err) == (1, 0)
    (r,) = read_rows(tmp_path / "b.csv")
    assert r["scene_id"] == "real@wrong-bikes" and r["pred"] == "0.0"


def test_ghost_column_round_trips_with_old_rows(tmp_path):
    assert "ghost_duplicate_rate" in COLUMNS
    old = tmp_path / "old.csv"
    old_cols = COLUMNS[: COLUMNS.index("ghost_duplicate_rate")]
    with old.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_cols)
        w.writeheader()
        w.writerow({c: "" for c in old_cols} | {"run_key": "old", "status": "ok"})
    assert read_rows(old)[0]["ghost_duplicate_rate"] == ""

    jobs = [Job("a", "m", 3, config={"i": 1}), Job("b", "m", 3, config={"i": 2}),
            Job("c", "m", 3, config={"i": 3})]  # fmt: skip
    outs = {"a": RunOutput(1.0, 0.0, ghost_duplicate_rate=0.25),
            "b": RunOutput(1.0, 0.0, ghost_duplicate_rate=math.nan),
            "c": RunOutput(1.0, 0.0)}  # fmt: skip
    run_jobs(jobs, lambda j: outs[j.scene_id], old)  # upgrades the old header in place
    rows = {r["scene_id"]: r for r in read_rows(old)}
    assert rows["a"]["ghost_duplicate_rate"] == "0.25"
    assert rows["b"]["ghost_duplicate_rate"] == "" and rows[""]["run_key"] == "old"
    assert json.loads(json.dumps(list(rows))) and rows["c"]["ghost_duplicate_rate"] == ""
