"""Tests for the synthetic episode generator and the dustbin Sinkhorn (PLAN §18)."""

from __future__ import annotations

import numpy as np
import pytest

from synth.geometry import (
    EpisodeDataset,
    GeometryConfig,
    generate_episode,
    load_episode,
    save_episode,
    write_dataset,
)


def _same(a, b):
    return all(
        np.array_equal(getattr(a, k), getattr(b, k))
        for k in ("K", "R", "t", "points3d", "view", "uv", "depth", "score", "identity")
    )


def test_deterministic_per_seed():
    assert _same(generate_episode(3), generate_episode(3))
    assert not _same(generate_episode(3), generate_episode(4))


def test_episode_labels_consistent():
    ep = generate_episode(7)
    assert ep.uv.shape == (ep.n_detections, 2)
    assert ep.identity.max() < ep.n_instances
    for v in range(ep.n_views):
        ids = ep.identity[ep.view_slice(v)]
        real = ids[ids >= 0]
        assert real.size == np.unique(real).size  # one detection per instance per view
    assert 0 <= ep.true_count() <= ep.n_instances


def test_false_positives_and_drops_present():
    cfg = GeometryConfig(drop_prob=0.5, false_positive_rate=0.5, n_instances=(40, 40))
    ep = generate_episode(1, cfg)
    assert (ep.identity == -1).any()
    clean = generate_episode(1, GeometryConfig(drop_prob=0.0, false_positive_rate=0.0,
                                               n_instances=(40, 40)))
    assert ep.n_detections != clean.n_detections or (ep.identity == -1).any()
    assert (clean.identity >= 0).all()


def test_npz_roundtrip_and_dataset(tmp_path):
    ep = generate_episode(11)
    save_episode(ep, tmp_path / "x.npz")
    back = load_episode(tmp_path / "x.npz")
    assert back.seed == 11
    assert np.allclose(back.uv, ep.uv, atol=1e-3)
    assert np.array_equal(back.identity, ep.identity)

    write_dataset(tmp_path / "ds", 5, seed0=20)
    ds = EpisodeDataset(tmp_path / "ds")
    assert len(ds) == 5
    for i in range(len(ds)):
        s = ds[i]
        na, nb = s["feat_a"].shape[0], s["feat_b"].shape[0]
        assert s["feat_a"].shape[1] == s["feat_b"].shape[1] == 4
        assert s["match"].shape == (na, nb)
        assert s["unmatched_a"].shape == (na,) and s["unmatched_b"].shape == (nb,)
        assert s["match"].sum(axis=1).max(initial=0) <= 1
        assert np.array_equal(s["unmatched_a"], ~s["match"].any(axis=1))
        assert np.array_equal(ds[i]["match"], s["match"])  # deterministic pair choice


torch = pytest.importorskip("torch")

from countonce.match.sinkhorn import DustbinSinkhorn, log_sinkhorn  # noqa: E402


@pytest.mark.parametrize("shape", [(5, 7), (1, 4), (6, 6)])
def test_sinkhorn_marginals(shape):
    m, n = shape
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(2, m, n, generator=g, dtype=torch.float64)
    p = log_sinkhorn(scores, torch.tensor(0.5, dtype=torch.float64), iters=200).exp()
    assert p.shape == (2, m + 1, n + 1)
    assert torch.allclose(p[:, :m].sum(-1), torch.ones(2, m, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(p[:, :, :n].sum(-2), torch.ones(2, n, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(p[:, m].sum(-1), torch.full((2,), float(n), dtype=torch.float64),
                          atol=1e-6)
    assert torch.allclose(p[:, :, n].sum(-1), torch.full((2,), float(m), dtype=torch.float64),
                          atol=1e-6)


def test_dustbin_learnable_and_absorbs_outliers():
    mod = DustbinSinkhorn(iters=30)
    scores = torch.full((3, 4), -10.0)
    scores[0, 0] = 10.0
    out = mod(scores).exp()
    assert out[0, 0] > 0.9
    assert out[1, :4].sum() < out[1, 4]  # unmatched row goes to the dustbin
    out.sum().backward()
    assert mod.bin_score.grad is not None
