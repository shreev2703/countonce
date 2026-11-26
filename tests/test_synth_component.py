"""M5 component test helpers: finite noisy depth, tuning, count-first scoring."""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from bench.metrics import decomposition_error
from bench.synth_component import (
    COUNT_COLUMNS,
    DepthModel,
    Prepared,
    episode_inputs_for,
    evaluate,
    make_v1,
    mean_abs_error,
    render_depth,
    score_episode,
    summarize_counts,
    tune,
)
from countonce.merge.base import ClusterResult
from countonce.merge.match import V2Associator, _Views
from countonce.types import Instance
from synth.geometry import Episode, GeometryConfig, generate_episode, write_dataset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_synth_component.py"


def _prep(seed: int, finite: bool) -> Prepared:
    ep = generate_episode(seed, GeometryConfig(n_instances=(8, 12), n_views=(3, 4)))
    lifted, recon = episode_inputs_for(ep, finite)
    return Prepared(ep, lifted, recon)


def test_finite_depth_is_finite_and_deterministic():
    ep = _prep(1, True).ep
    d = render_depth(ep)
    assert d.shape[0] == ep.n_views and np.isfinite(d).all() and (d > 0).all()
    assert np.array_equal(d, render_depth(ep))


def test_nan_variant_keeps_depth_inert():
    assert np.isnan(_prep(1, False).recon.depth).all()


def test_visible_instance_depth_matches_detection_depth():
    """Where an instance is the nearest surface the map agrees with its true camera depth."""
    p = _prep(2, True)
    ep = p.ep
    cam = ep.points3d @ ep.R[0].T + ep.t[0]
    z = cam[:, 2]
    near = int(np.argmin(np.where(z > 0.1, z, np.inf)))
    uv = (ep.K[0] @ cam[near])[:2] / z[near] * (64 / 640)
    if 0 <= uv[0] < 64 and 0 <= uv[1] < 64:
        got = p.recon.depth[0, int(uv[1]), int(uv[0])]
        assert abs(got - z[near]) < 0.25 * z[near]


def test_decomposition_identity_and_fp_counted():
    data = [_prep(s, True) for s in (3, 4, 5)]
    rows = evaluate(make_v1({"k": 0.5}), data)
    for r in rows:
        assert (
            r["duplicates"] + r["false_positives"] - r["missed"] - r["wrong_merges"]
            == (r["signed_err"])
        )
    assert "injected_fp_dets" in summarize_counts(rows)


def test_tune_never_worse_than_default_on_val():
    data = [_prep(s, False) for s in (6, 7, 8)]
    res = tune("v1", make_v1, {"k": [0.25, 0.5, 1.0]}, {"k": 0.5}, data)
    assert res.val_mae <= res.default_val_mae + 1e-9
    assert mean_abs_error(evaluate(make_v1(res.best), data)) == res.val_mae


def test_occlusion_check_rejects_an_occluded_candidate(monkeypatch):
    """On a fixed seed v2 drops a track that is in the image but behind a nearer surface."""
    original = _Views.project
    calls: list[tuple[np.ndarray, float, int, float, bool, np.ndarray, float]] = []

    def spy(self, xyz, size, view, depth_tol):
        vis, uv, z = original(self, xyz, size, view, depth_tol)
        calls.append((np.array(xyz), size, view, depth_tol, vis, uv, z))
        return vis, uv, z

    ep = generate_episode(11, GeometryConfig(n_instances=(40, 40), n_views=(4, 4)))
    lifted, finite = episode_inputs_for(ep, True, DepthModel(noise=0.0))
    _, nan = episode_inputs_for(ep, False)
    monkeypatch.setattr(_Views, "project", spy)
    V2Associator().associate(lifted, finite, None)
    monkeypatch.undo()

    occluded = []
    for xyz, size, view, tol, vis, uv, z in calls:
        h, w = finite.depth.shape[1:]
        if vis or z <= 0 or not (0 <= uv[0] < w and 0 <= uv[1] < h):
            continue
        if finite.depth[view, int(uv[1]), int(uv[0])] < z - tol * size:  # something in front
            occluded.append((xyz, size, view, tol))
    assert occluded
    # the very same candidates pass when depth is NaN, so only the occlusion test removed them
    geo = _Views(nan, None, None)
    assert all(geo.project(*c)[0] for c in occluded)


def _toy() -> tuple[Episode, list, ClusterResult]:
    """Objects 0-3 detected; two false positives (identity -1). One of each error type."""
    identity = np.array([0, 0, 1, 2, 3, -1, -1])
    n = identity.size
    ep = Episode(
        seed=0,
        K=np.array([[[500.0, 0, 320], [0, 500.0, 320], [0, 0, 1]]]),
        R=np.eye(3)[None],
        t=np.zeros((1, 3)),
        points3d=np.array([[0.0, 0, 3]] * 4),
        view=np.zeros(n, dtype=np.int64),
        uv=np.full((n, 2), 320.0),
        depth=np.full(n, 3.0),
        score=np.full(n, 0.9),
        identity=identity,
    )
    lifted, _ = episode_inputs_for(ep, False)
    ids = [ld.det.id for ld in lifted]
    # object 0 twice (duplicate); 1 and 2 merged; object 3 rejected (missed);
    # one false positive becomes an instance, the other is rejected.
    groups = [[0], [1], [2, 3], [5]]
    instances = [
        Instance(
            id=k,
            xyz=np.zeros(3),
            size=0.1,
            members=[ids[i] for i in g],
            n_views=1,
            conf=1.0,
        )
        for k, g in enumerate(groups)
    ]
    return ep, lifted, ClusterResult(instances=instances)


def test_toy_decomposition_counts_injected_false_positives():
    ep, lifted, res = _toy()
    row = score_episode(ep, lifted, res)
    assert row["injected_fp_dets"] == 2
    got = {k: int(row[k]) for k in ("missed", "duplicates", "false_positives", "wrong_merges")}
    assert got == {"missed": 1, "duplicates": 1, "false_positives": 1, "wrong_merges": 1}
    assert decomposition_error(got) == row["pred"] - row["gt"] == 0


def test_zero_scene_false_count():
    base = {k: 0.0 for k in summarize_counts([score_episode(*_toy())])}
    rows = [dict(base, pred=3.0, gt=0.0), dict(base, pred=10.0, gt=10.0)]
    s = summarize_counts(rows)
    assert s["n_zero"] == 1 and s["zero_scene_false_count"] == 3.0
    assert s["mae"] == 1.5 and s["signed_bias"] == 1.5 and s["within_10pct"] == 0.5
    assert math.isnan(summarize_counts(rows[1:])["zero_scene_false_count"])


def test_help_documents_both_depth_variants():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True
    ).stdout
    assert "nan_depth" in out and "finite_depth" in out


def test_script_writes_count_first_csv(tmp_path, monkeypatch):
    """End to end on three tiny CPU episodes, with v1 standing in for v3 (no checkpoint)."""
    spec = importlib.util.spec_from_file_location("eval_synth_component", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "V3Associator", lambda _: make_v1({"k": 0.5}))
    for name in ("V1_GRID", "V2_GRID", "SYNC_GRID"):
        monkeypatch.setattr(mod, name, {k: v[:1] for k, v in getattr(mod, name).items()})
    cfg = GeometryConfig(n_instances=(5, 6), n_views=(3, 3))
    for split in ("val", "test"):
        write_dataset(tmp_path / split, 3, cfg=cfg)
    out = tmp_path / "out" / "summary.csv"
    argv = ["--data", str(tmp_path), "--n-boot", "20", "--out-csv", str(out)]
    argv += ["--out-json", str(tmp_path / "out" / "r.json")]
    argv += ["--out-rows-csv", str(tmp_path / "out" / "rows.csv")]
    assert mod.main(argv) == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert {r["variant"] for r in rows} == {"nan_depth", "finite_depth"}
    assert tuple(rows[0])[4:8] == COUNT_COLUMNS
    assert all(float(r["mae"]) >= abs(float(r["signed_bias"])) - 1e-9 for r in rows)
