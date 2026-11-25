"""M6 failure-source harness: conditions, identical upstream inputs, tuning, both fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench import failure_sources as fs
from synth.geometry import GeometryConfig, generate_episode

TINY = Path(__file__).resolve().parents[1] / "examples" / "tiny_scene"


def _synth(seed: int) -> fs.SceneBundle:
    ep = generate_episode(seed, GeometryConfig(n_instances=(6, 10), n_views=(4, 5)))
    return fs.synth_bundle(ep, seed=seed)


@pytest.fixture(scope="module")
def tiny() -> fs.SceneBundle:
    return fs.tiny_scene_bundle(TINY)


def test_tiny_oracle_detections_see_each_sphere(tiny):
    dets, ident = fs.oracle_detections(tiny)
    assert {v for v in ident.values()} == set(tiny.gt_ids)
    assert len(dets) == len(ident) > 0


def test_tiny_runs_all_conditions_and_methods(tiny):
    rows = fs.run_scene(tiny, fs.build_associators())
    assert {r["condition"] for r in rows} == set(fs.CONDITIONS)
    assert {r["method"] for r in rows} == {"v1", "v2", "sync"}
    assert all(r["gt"] == 5 for r in rows)


def test_decomposition_identity(tiny):
    for r in fs.run_scene(tiny, fs.build_associators()):
        assert (
            r["duplicates"] + r["false_positives"] - r["missed"] - r["wrong_merges"]
            == (r["signed_err"])
        )


def test_identical_upstream_for_every_associator(tiny):
    rows = fs.run_scene(tiny, fs.build_associators())
    for cond in fs.CONDITIONS:
        hashes = {r["input_hash"] for r in rows if r["condition"] == cond}
        assert len(hashes) == 1
    assert len({r["input_hash"] for r in rows}) > 1  # conditions do differ


def test_oracle_ceiling_on_tiny_with_tuned_settings(tiny):
    """With perfect detections and geometry, tuned associators recover the true count; the
    defaults do not (front-surface lifting puts one sphere's views ~0.2 apart)."""
    tuned = fs.tune_associators([tiny], "oracle_det__oracle_geom")
    rows = fs.run_scene(tiny, fs.build_associators(tuned), ["oracle_det__oracle_geom"])
    assert [r["abs_err"] for r in rows] == [0.0, 0.0, 0.0]


def test_perturb_recon_is_deterministic_and_changes_geometry(tiny):
    a = fs.perturb_recon(tiny.oracle_recon, seed=3)
    b = fs.perturb_recon(tiny.oracle_recon, seed=3)
    assert np.array_equal(a.extrinsics, b.extrinsics)
    assert not np.allclose(a.extrinsics, tiny.oracle_recon.extrinsics)
    assert np.array_equal(np.nan_to_num(a.world_points), np.nan_to_num(b.world_points))


def test_synth_fixture_runs_and_decomposes():
    bundles = [_synth(s) for s in (1, 2)]
    rows = fs.run_bundles(bundles, fs.build_associators())
    assert len(rows) == 2 * len(fs.CONDITIONS) * 3
    for r in rows:
        assert (
            r["missed"] + r["duplicates"] + r["false_positives"] - r["wrong_merges"]
            == (r["signed_err"])
        )
    summ = fs.summarize(rows)
    assert {s["condition"] for s in summ} == set(fs.CONDITIONS)
    att = fs.attribution(summ)
    assert {a["method"] for a in att} == {"v1", "v2", "sync"}


def test_synth_oracle_detections_have_no_false_positives():
    b = _synth(4)
    up = fs.build_upstream(b, "oracle_det__oracle_geom")
    assert all(v is not None for v in up.identity.values())


def test_tune_returns_grid_points_and_not_worse_than_default():
    val = [_synth(s) for s in (10, 11)]
    best = fs.tune_associators(val, methods=("v1",))
    assert best["v1"]["k"] in fs.V1_GRID["k"]
    up = [fs.build_upstream(b, "full") for b in val]

    def mae(assoc):
        return np.mean(
            [
                fs.score(b, u, assoc.associate(u.lifted, u.recon, b.frames))["abs_err"]
                for b, u in zip(val, up, strict=True)
            ]
        )

    assert mae(fs.make_v1(best["v1"])) <= mae(fs.make_v1({})) + 1e-9


def test_unknown_condition_rejected(tiny):
    with pytest.raises(ValueError):
        fs.build_upstream(tiny, "nope")
