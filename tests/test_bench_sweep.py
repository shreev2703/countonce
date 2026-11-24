"""Sweep grid, figures and failure gallery, smoke-tested on a synthetic bench.csv in tmp."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.figures import load_rows, make_all
from bench.gallery import worst_fraction, write_gallery
from bench.runner import Job, RunOutput, read_rows, run_jobs
from bench.sweep import BenchConfig, build_jobs, discover_scenes, run_bench, select_views
from countonce.cli import main

CFG = Path(__file__).parents[1] / "configs" / "bench.yaml"


def test_repo_config_loads_and_grid_shape():
    cfg = BenchConfig.from_yaml(CFG)
    assert cfg.views == [3, 5, 10, 20, 40]
    jobs = build_jobs(cfg, [{"id": "s1", "tier": "easy"}])
    n_methods = len([m for m in cfg.all_methods() if m.kind != "zero_suite"])
    assert len(jobs) == n_methods * 5 * 4 * 2 * 3  # 4 samples, 2 orders, 3 spreads
    assert len({j.key() for j in jobs}) == len(jobs)
    assert any("+merge_k=" in m.name for m in cfg.all_methods())
    assert all(m.name != "co_v2_vggt+backbone=vggt" for m in cfg.all_methods())


def test_view_selection_same_frames_ordered_vs_shuffled():
    paths = [Path(f"{i:03d}.jpg") for i in range(40)]
    a = select_views(paths, 10, 1, 0, ordered=True)
    b = select_views(paths, 10, 1, 0, ordered=False)
    assert sorted(a) == sorted(b) and a == sorted(a) and a != b
    assert select_views(paths, 5, None, 0, True) == select_views(paths, 5, None, 0, True)


def _synthetic_csv(path: Path) -> None:
    rng = np.random.default_rng(0)
    jobs, truth = [], {}
    tiers = ["easy", "medium", "hard"]
    for s in range(12):
        gt = 10 + 5 * s
        truth[f"s{s}"] = gt
        for m in ("sum", "co_v2"):
            for n in (3, 10):
                for smp in ("random0", "even"):
                    for ordered in (True, False):
                        jobs.append(Job(f"s{s}", m, n, smp, ordered, "test",
                                        {"tier": tiers[s % 3]}))  # fmt: skip

    def fn(j: Job) -> RunOutput:
        gt = truth[j.scene_id]
        sd = 0.8 if j.method == "sum" else 0.1
        pred = gt * (1 + rng.normal(0, sd))
        has_iv = j.method == "co_v2"
        return RunOutput(pred, gt, pred - 0.2 * gt if has_iv else None,
                         pred + 0.2 * gt if has_iv else None)  # fmt: skip

    run_jobs(jobs, fn, path)


def test_figures_and_gallery_on_synthetic_csv(tmp_path):
    pytest.importorskip("matplotlib")
    csv_path = tmp_path / "bench.csv"
    _synthetic_csv(csv_path)
    rows = load_rows(csv_path)
    assert {r.tier for r in rows} == {"easy", "medium", "hard"}
    figs = make_all(csv_path, tmp_path / "figs")
    assert {f.name for f in figs} >= {
        "mape_vs_views.png",
        "ordered_vs_unordered.png",
        "tier_error.png",
        "coverage.png",
    }
    assert all(f.exists() and f.stat().st_size > 1000 for f in figs)

    worst = write_gallery(csv_path, tmp_path / "gallery.md")
    assert (tmp_path / "gallery.md").read_text().startswith("# Failure gallery")
    methods = {r.method for r in rows}
    for m in methods:
        n = sum(r.method == m for r in rows)
        assert sum(r.method == m for r in worst) == -(-n // 10)


def test_worst_fraction_is_ranked_and_deterministic(tmp_path):
    csv_path = tmp_path / "bench.csv"
    _synthetic_csv(csv_path)
    rows = load_rows(csv_path)
    a = worst_fraction(rows, 0.1, per_method=False)
    assert a == worst_fraction(list(reversed(rows)), 0.1, per_method=False)
    errs = [abs(r.pred - r.gt) / r.gt for r in a]
    assert errs == sorted(errs, reverse=True)
    assert min(errs) >= np.quantile([abs(r.pred - r.gt) / r.gt for r in rows], 0.88)
    with pytest.raises(ValueError):
        worst_fraction(rows, 0)


def test_run_bench_resumes_and_cli(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    scenes = tmp_path / "scenes" / "a"
    (scenes / "images").mkdir(parents=True)
    (scenes / "gt.json").write_text(json.dumps({"count": 7, "tier": "hard"}))
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(
        f"scenes_dir: {tmp_path / 'scenes'}\ncsv: {tmp_path / 'out' / 'bench.csv'}\n"
        "views: [3]\nrandom_samples: 1\nmethods:\n  - {name: sum, kind: baseline, baseline: sum}\n"
    )
    assert main(["bench", "--config", str(cfg_path), "--dry-run"]) == 0
    assert "1 scenes, 4 jobs" in capsys.readouterr().out

    cfg = BenchConfig.from_yaml(cfg_path)
    sc = discover_scenes(cfg.scenes_dir)
    assert run_bench(cfg, sc, lambda j: RunOutput(7.0, 7.0)) == (4, 0, 0)
    assert run_bench(cfg, sc, lambda j: RunOutput(7.0, 7.0)) == (0, 4, 0)
    assert json.loads(read_rows(cfg.csv)[0]["config_json"])["tier"] == "hard"

    assert main(["bench", "--config", str(cfg_path), "--figures", str(tmp_path / "f")]) == 0
    assert (tmp_path / "f" / "failure_gallery.md").exists()
    assert (tmp_path / "f" / "mape_vs_views.png").exists()


def test_sweep_end_to_end_baselines_and_colmap_lifting_ok(tmp_path):
    """Baselines run FakeDetector through the detection cache; co_v1_colmap runs through the
    same executor with a fake reconstructor. Every row must be status=ok."""
    import shutil

    from test_pipeline import OBJECTS, PlaneRecon, make_scene

    from countonce.detect.base import FakeDetector

    src = tmp_path / "src"
    src.mkdir()
    img_dir, script = make_scene(src)
    script[2], script[3] = script[2][:4], script[3][:3]  # uneven per-view counts (5, 5, 4, 3)
    script[0] = [*script[0], (1.0, 1.0, 5.0, 5.0, 0.1)]  # below score_threshold: must be dropped
    scene = tmp_path / "scenes" / "s1"
    scene.mkdir(parents=True)
    shutil.move(str(img_dir), scene / "images")
    (scene / "gt.json").write_text(json.dumps({"count": len(OBJECTS), "tier": "easy"}))

    counts = [sum(1 for b in script[i] if b[4] >= 0.3) for i in range(len(script))]
    assert len(set(counts)) > 1  # sum, max and median are distinguishable

    cfg = BenchConfig(
        scenes_dir=tmp_path / "scenes",
        csv=tmp_path / "bench.csv",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        views=[4],
        order=["ordered"],  # FakeDetector is positional; a content-keyed cache needs one order
        random_samples=1,
        even_sample=True,
        methods=[
            {"name": "sum", "kind": "baseline", "baseline": "sum"},
            {"name": "max", "kind": "baseline", "baseline": "max"},
            {"name": "median", "kind": "baseline", "baseline": "median"},
            {
                "name": "co_v1_colmap",
                "kind": "countonce",
                "config": {"assoc": "v1", "backbone": "colmap", "max_frames": 0},
            },
        ],
    )
    detectors: list[FakeDetector] = []

    def make_det(conf):
        d = FakeDetector(script)
        detectors.append(d)
        return d

    ran, skipped, errored = run_bench(
        cfg,
        discover_scenes(cfg.scenes_dir),
        detector_factory=make_det,
        reconstructor_factory=lambda conf: PlaneRecon(),
    )
    assert (ran, skipped, errored) == (8, 0, 0)  # 4 methods x (random0 + even)
    rows = read_rows(cfg.csv)
    assert {r["status"] for r in rows} == {"ok"}
    expected = {
        "sum": float(sum(counts)),
        "max": float(max(counts)),
        "median": float(np.median(counts)),
        "co_v1_colmap": float(len(OBJECTS)),
    }
    for r in rows:
        assert float(r["pred"]) == expected[r["method"]], r["method"]
    assert any((cfg.cache_dir / "detect").iterdir())  # detections went through the cache
