"""Viewpoint-spread sweep axis on synth/geometry episodes with known cameras."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bench.figures import load_rows, make_all
from bench.runner import COLUMNS, Job, RunOutput, read_rows, run_jobs
from bench.slices import centers_spread
from bench.sweep import BenchConfig, build_jobs, select_spread_views, select_views
from synth.geometry import GeometryConfig, generate_episode


def _episode_centers(seed: int = 3, n: int = 40) -> tuple[list[Path], np.ndarray]:
    ep = generate_episode(seed, GeometryConfig(n_views=(n, n)))
    centers = -np.einsum("nji,nj->ni", ep.R, ep.t)  # camera centres C = -R^T t
    return [Path(f"{i:03d}.jpg") for i in range(len(centers))], centers


def _idx(paths: list[Path], chosen: list[Path]) -> list[int]:
    pos = {p: i for i, p in enumerate(paths)}
    return [pos[c] for c in chosen]


def test_narrow_has_smaller_spread_than_wide():
    for seed in range(5):
        paths, centers = _episode_centers(seed)
        for n in (4, 8):
            for smp in (None, 0, 1, 2):
                nar = select_spread_views(paths, centers, n, "narrow", smp, 0)
                wid = select_spread_views(paths, centers, n, "wide", smp, 0)
                assert len(nar) == len(wid) == n
                sn = centers_spread(centers[_idx(paths, nar)])
                sw = centers_spread(centers[_idx(paths, wid)])
                assert sn < sw, (seed, n, smp)


def test_selection_deterministic_and_all_matches_plain():
    paths, centers = _episode_centers()
    a = select_spread_views(paths, centers, 6, "narrow", 1, 7)
    assert a == select_spread_views(paths, centers, 6, "narrow", 1, 7)
    assert select_views(paths, 6, 1, 7, True, "all", centers) == select_views(paths, 6, 1, 7, True)
    assert _idx(paths, a) == sorted(_idx(paths, a))


def test_build_jobs_spread_axis_and_stable_keys():
    cfg = BenchConfig(methods=[{"name": "sum", "kind": "baseline", "baseline": "sum"}],
                      views=[3], random_samples=1, spread=["narrow", "wide", "all"])  # fmt: skip
    jobs = build_jobs(cfg, [{"id": "s"}])
    assert len(jobs) == 1 * 2 * 2 * 3 and len({j.key() for j in jobs}) == len(jobs)
    plain = build_jobs(cfg.model_copy(update={"spread": ["all"]}), [{"id": "s"}])
    assert {j.key() for j in plain} <= {j.key() for j in jobs}
    assert (
        Job("s", "m", 3, "even", True, "t", {"a": 1}).key()
        == Job("s", "m", 3, "even", True, "t", {"a": 1}, spread="all").key()
    )


def test_csv_columns_deterministic_and_old_csv_loads(tmp_path):
    paths, centers = _episode_centers()

    def fn(j: Job) -> RunOutput:
        chosen = select_views(paths, j.n_views or 0, 0, 0, True, j.spread, centers)
        sp = centers_spread(centers[_idx(paths, chosen)])
        return RunOutput(pred=10.0, gt=10.0 + sp, baseline_spread=sp)

    jobs = [Job("s", "m", 6, "random0", True, "test", {"tier": "easy"}, spread=s)
            for s in ("narrow", "wide", "all")]  # fmt: skip
    runs = []
    for k in range(2):
        csv_path = tmp_path / f"b{k}.csv"
        run_jobs(jobs, fn, csv_path)
        runs.append({r["spread"]: r["baseline_spread"] for r in read_rows(csv_path)})
    assert runs[0] == runs[1]
    assert float(runs[0]["narrow"]) < float(runs[0]["wide"])
    assert load_rows(tmp_path / "b0.csv")[0].baseline_spread is not None

    # An old CSV (pre-spread columns) still loads, and resuming upgrades the header.
    old = tmp_path / "old.csv"
    old_cols = COLUMNS[: COLUMNS.index("spread")]
    with old.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_cols)
        w.writeheader()
        w.writerow({c: "" for c in old_cols} | {"run_key": "k", "status": "ok", "scene_id": "s",
                   "method": "m", "n_views": "3", "ordered": "1", "pred": "1", "gt": "1",
                   "config_json": "{}"})  # fmt: skip
    rows = load_rows(old)
    assert rows[0].spread == "" and rows[0].baseline_spread is None
    run_jobs(jobs[:1], fn, old)
    assert read_rows(old)[0]["spread"] == "" and len(read_rows(old)) == 2


def test_error_vs_spread_figure(tmp_path):
    paths, centers = _episode_centers()
    rng = np.random.default_rng(0)

    def fn(j: Job) -> RunOutput:
        chosen = select_views(paths, j.n_views or 0, 0, 0, True, j.spread, centers)
        sp = centers_spread(centers[_idx(paths, chosen)])
        return RunOutput(pred=10 * (1 + rng.normal(0, 0.2 / (1 + sp))), gt=10.0, baseline_spread=sp)

    jobs = [
        Job(f"s{i}", m, 6, "random0", True, "test", {"tier": "easy"}, spread=s)
        for i in range(4)
        for m in ("sum", "co")
        for s in ("narrow", "wide", "all")
    ]
    run_jobs(jobs, fn, tmp_path / "b.csv")
    out = make_all(tmp_path / "b.csv", tmp_path / "figs")
    assert any(p.name == "error_vs_spread.png" and p.stat().st_size > 0 for p in out)
