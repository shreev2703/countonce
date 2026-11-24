"""CountOnce-Match v3: model shapes, overfit, permutation, cannot-link (PLAN §10.3, §18)."""

import time

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from countonce.config import Config
from countonce.match.instances import form_instances
from countonce.merge.base import Associator
from countonce.merge.v3 import V3Associator, require_checkpoint
from synth.geometry import GeometryConfig, generate_episode


def _labels_to_partition(labels):
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), set()).add(i)
    return {frozenset(g) for g in groups.values()}


# ---------------------------------------------------------------- instances (no torch)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(1, 14),
    n_views=st.integers(1, 4),
    seed=st.integers(0, 10_000),
    thr=st.floats(0.05, 0.95),
)
def test_cannot_link_never_violated(n, n_views, seed, thr):
    rng = np.random.default_rng(seed)
    views = rng.integers(0, n_views, n)
    probs = rng.random((n, n))  # dense, adversarial: same-view pairs also get high scores
    labels = form_instances(probs, views, thr)
    assert labels.shape == (n,)
    for lab in np.unique(labels):
        v = views[labels == lab]
        assert len(v) == len(set(v.tolist()))


def test_components_and_split():
    views = np.array([0, 1, 2, 0])
    p = np.zeros((4, 4))
    for i, j in [(0, 1), (1, 2), (3, 1), (3, 2)]:
        p[i, j] = p[j, i] = 0.9
    p[0, 2] = p[2, 0] = 0.9
    p[3, 1] = p[1, 3] = 0.6  # detection 3 shares a view with 0; it is the weaker link
    labels = form_instances(p, views, 0.5)
    assert labels[0] != labels[3]
    assert labels[0] == labels[1] == labels[2]
    # without conflicts the components are plain connected components
    l2 = form_instances(p[:3, :3], views[:3], 0.5)
    assert len(set(l2.tolist())) == 1
    assert form_instances(np.zeros((0, 0)), [], 0.5).shape == (0,)


# ---------------------------------------------------------------- model (torch)
torch = pytest.importorskip("torch")

from countonce.match.model import (  # noqa: E402
    BASE_FEATURES,
    episode_inputs,
    episode_loss,
    load_checkpoint,
    make_model,
    save_checkpoint,
)

SMALL = GeometryConfig(n_views=(3, 4), n_instances=(5, 10))


def test_output_shapes_and_marginals():
    ep = generate_episode(1, SMALL)
    d = episode_inputs(ep)
    assert d.features().shape == (ep.n_detections, BASE_FEATURES)
    model = make_model().eval()
    assert model.hparams["width"] == 128 and model.hparams["layers"] == 4
    assert model.encode(d).shape == (ep.n_detections, 128)
    logp = model.pair_log_assignments(d)
    assert len(logp) == ep.n_views * (ep.n_views - 1) // 2
    ids = sorted(set(ep.view.tolist()))
    for (a, b), lp in logp.items():
        na, nb = len(ep.view_slice(ids[a])), len(ep.view_slice(ids[b]))
        assert lp.shape == (na + 1, nb + 1)
        assert torch.allclose(lp[:-1, :].exp().sum(-1), torch.ones(na), atol=1e-3)
    p = model.match_probs(d)
    assert p.shape == (ep.n_detections, ep.n_detections)
    assert torch.allclose(p, p.T)
    same = torch.as_tensor(ep.view[:, None] == ep.view[None, :])
    assert float(p[same].detach().abs().max()) == 0.0


def test_appearance_slot_and_single_view():
    ep = generate_episode(2, SMALL)
    d = episode_inputs(ep)
    d.appearance = np.random.default_rng(0).normal(size=(d.n, 8))
    model = make_model(width=32, layers=2, appearance_dim=8)
    assert model.match_probs(d).shape == (d.n, d.n)
    one = make_model(width=32, layers=2)
    d1 = episode_inputs(ep)
    keep = np.flatnonzero(d1.view == d1.view[0])
    p = one.match_probs(d1.permuted(keep))  # a single view: nothing to match, nothing breaks
    assert p.shape == (len(keep), len(keep)) and float(p.detach().abs().max()) == 0.0


def _pair_stats(model, eps):
    tp = fp = fn = correct = total = 0
    with torch.no_grad():
        for ep in eps:
            p = (model.match_probs(episode_inputs(ep)) > 0.5).numpy()
            gt = (ep.identity[:, None] == ep.identity[None, :]) & (ep.identity[:, None] >= 0)
            cross = ep.view[:, None] != ep.view[None, :]
            iu = np.triu(cross, 1)
            tp += int((p & gt & iu).sum())
            fp += int((p & ~gt & iu).sum())
            fn += int((~p & gt & iu).sum())
            correct += int((p == gt)[iu].sum())
            total += int(iu.sum())
    return correct / total, tp, fp, fn


@pytest.mark.slow
def test_overfits_ten_episodes():
    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    eps = [generate_episode(s, SMALL) for s in range(10)]
    inputs = [episode_inputs(e) for e in eps]
    model = make_model()  # width 128, 4 layers, as specified
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    t0 = time.perf_counter()
    acc = 0.0
    for epoch in range(300):
        model.train()
        for ep, d in zip(eps, inputs, strict=True):
            opt.zero_grad()
            episode_loss(model, ep, d).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch % 5 == 4:
            model.eval()
            acc, tp, fp, fn = _pair_stats(model, eps)
            if acc >= 0.999 and fp + fn <= 0.01 * max(tp, 1):
                break
        assert time.perf_counter() - t0 < 55, f"too slow, pair accuracy {acc:.4f}"
    model.eval()
    acc, tp, fp, fn = _pair_stats(model, eps)
    assert acc >= 0.995, acc
    assert tp / max(tp + fp + fn, 1) >= 0.95, (tp, fp, fn)


def test_view_permutation_leaves_instances_unchanged():
    torch.manual_seed(1)
    ep = generate_episode(5, SMALL)
    d = episode_inputs(ep)
    model = make_model(width=32, layers=4).eval()
    rng = np.random.default_rng(0)
    # relabel views and shuffle the detection order: instances must be the same partition
    relabel = rng.permutation(ep.n_views) + 10
    d2 = episode_inputs(ep)
    d2.view = relabel[d2.view]
    order = rng.permutation(d.n)
    d2 = d2.permuted(order)
    with torch.no_grad():
        p1 = model.match_probs(d).numpy()
        p2 = model.match_probs(d2).numpy()
    inv = np.argsort(order)
    np.testing.assert_allclose(p2[np.ix_(inv, inv)], p1, atol=1e-4)
    # an untrained model gives near-tied probabilities; cut in the widest gap among the top
    # values so float noise (~1e-7) cannot flip an edge
    top = np.unique(p1[p1 > 0])[-40:]
    gap = int(np.argmax(np.diff(top)))
    thr = float((top[gap] + top[gap + 1]) / 2)
    a = _labels_to_partition(form_instances(p1, d.view, thr))
    b = form_instances(p2, d2.view, thr)[inv]
    assert a == _labels_to_partition(b)


def test_checkpoint_roundtrip_and_associator(tmp_path):
    from countonce.types import Detection, LiftedDetection, Recon

    model = make_model(width=32, layers=2)
    ck = tmp_path / "m.pt"
    save_checkpoint(model, ck)
    assert load_checkpoint(ck).hparams == model.hparams

    h, w, nv = 40, 60, 3
    k = np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]])
    ext = np.zeros((nv, 3, 4))
    for v in range(nv):
        ext[v, :, :3] = np.eye(3)
        ext[v, 0, 3] = -0.2 * v
    recon = Recon(
        backbone="vggt",
        intrinsics=np.tile(k, (nv, 1, 1)),
        extrinsics=ext,
        depth=np.full((nv, h, w), 5.0),
        depth_conf=np.ones((nv, h, w)),
        world_points=np.zeros((nv, h, w, 3)),
        world_points_conf=np.ones((nv, h, w)),
        scene_scale=5.0,
    )
    lifted = []
    for v in range(nv):
        for j in range(3):
            box = np.array([10.0 + 12 * j, 10, 18.0 + 12 * j, 20])
            det = Detection(id=f"{v}:{j}", frame_id=v, score=0.9, box_xyxy=box)
            lifted.append(LiftedDetection(det, np.array([float(j), 0, 5.0]), 0.3, 0.9))
    assoc = V3Associator(ck)
    assert isinstance(assoc, Associator)
    res = assoc.associate(lifted, recon, None)
    members = sorted(m for i in res.instances for m in i.members)
    assert members == sorted(ld.det.id for ld in lifted)
    for inst in res.instances:  # cannot-link holds end to end
        frames = [int(m.split(":")[0]) for m in inst.members]
        assert len(frames) == len(set(frames))
    with pytest.raises(ValueError, match="reconstruction"):
        assoc.associate(lifted, None, None)


def test_missing_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="v3_checkpoint"):
        require_checkpoint(None)
    with pytest.raises(FileNotFoundError, match="not found"):
        V3Associator(tmp_path / "nope.pt")
    cfg = Config(assoc="v3", v3_checkpoint=tmp_path / "nope.pt")
    from countonce.pipeline import run

    with pytest.raises(FileNotFoundError, match="not found"):
        run([tmp_path], "apple", cfg)
