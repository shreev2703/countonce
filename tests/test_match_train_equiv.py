"""The batched (padded) training path equals the per-pair reference path on CPU."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from countonce.match.model import episode_inputs, make_model  # noqa: E402
from countonce.match.train import TrainConfig, _total, episode_terms, episode_terms_loop  # noqa: E402
from synth.geometry import generate_episode  # noqa: E402


def _grads(model, terms, cfg):
    model.zero_grad()
    _total(terms, cfg).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


@pytest.mark.parametrize("cycle_weight", [0.0, 0.5])
def test_padded_path_matches_loop_path(cycle_weight):
    cfg = TrainConfig(count_weight=0.1, cycle_weight=cycle_weight, max_triplets=4)
    torch.manual_seed(0)
    model = make_model(width=32, layers=2, heads=2, sinkhorn_iters=10).train()
    eps = [generate_episode(seed) for seed in range(8)]
    n_checked = 0
    for ep in eps:
        d = episode_inputs(ep)
        new = episode_terms(model, ep, d, cfg, np.random.default_rng(1))
        old = episode_terms_loop(model, ep, d, cfg, np.random.default_rng(1))
        assert (new is None) == (old is None)
        if new is None:
            continue
        assert new.keys() == old.keys()
        for k in new:
            assert float(new[k].detach()) == pytest.approx(float(old[k].detach()), abs=1e-5), k
        g_new, g_old = _grads(model, new, cfg), None
        old = episode_terms_loop(model, ep, d, cfg, np.random.default_rng(1))
        g_old = _grads(model, old, cfg)
        assert torch.allclose(g_new, g_old, atol=1e-5, rtol=1e-4)
        n_checked += 1
    assert n_checked >= 4


def test_match_probs_matches_loop_path():
    torch.manual_seed(0)
    model = make_model(width=32, layers=2, heads=2, sinkhorn_iters=10).eval()
    ep = generate_episode(3)
    d = episode_inputs(ep)
    from countonce.match.sinkhorn import log_sinkhorn

    p = torch.zeros(d.n, d.n)
    with torch.no_grad():
        for _, _, ia, ib, s in model._pairs(d):
            fwd = log_sinkhorn(s, model.bin_score, model.iters)[:-1, :-1].exp()
            bwd = log_sinkhorn(s.T, model.bin_score, model.iters)[:-1, :-1].exp().T
            core = (fwd + bwd) / 2
            p[np.ix_(ia, ib)] = core
            p[np.ix_(ib, ia)] = core.T
        got = model.match_probs(d)
    assert torch.allclose(got, p, atol=1e-5)
