"""M5 component test helpers: finite noisy depth, tuning, count-first scoring."""

from __future__ import annotations

import numpy as np

from bench.synth_component import (
    Prepared,
    episode_inputs_for,
    evaluate,
    make_v1,
    mean_abs_error,
    render_depth,
    summarize_counts,
    tune,
)
from synth.geometry import GeometryConfig, generate_episode


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
        assert r["duplicates"] + r["false_positives"] - r["missed"] - r["wrong_merges"] == (
            r["signed_err"]
        )
    assert "injected_fp_dets" in summarize_counts(rows)


def test_tune_never_worse_than_default_on_val():
    data = [_prep(s, False) for s in (6, 7, 8)]
    res = tune("v1", make_v1, {"k": [0.25, 0.5, 1.0]}, {"k": 0.5}, data)
    assert res.val_mae <= res.default_val_mae + 1e-9
    assert mean_abs_error(evaluate(make_v1(res.best), data)) == res.val_mae
