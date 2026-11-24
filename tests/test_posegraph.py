"""Pose-graph chunk stitching: synthetic chunks in random Sim(3) frames with per-chunk noise."""

import numpy as np
import pytest

from countonce.config import Config
from countonce.recon.align import camera_centers, transform_extrinsics, umeyama
from countonce.recon.chunked import ChunkAlignmentError, ChunkedReconstructor
from countonce.recon.posegraph import _chain_init, build_edges, solve_pose_graph
from tests.test_chunked_recon import SimFake, _frames, _rand_sim

N_FRAMES = 24


def _loop_truth(n):
    """Cameras on a circle looking at the centre (a closed trajectory)."""
    ext = np.zeros((n, 3, 4))
    for i in range(n):
        th = 2 * np.pi * i / n
        c = np.array([5 * np.cos(th), 5 * np.sin(th), 0.3 * np.sin(3 * th)])
        z = -c / np.linalg.norm(c)
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        ext[i, :3, :3] = R
        ext[i, :3, 3] = -R @ c
    return ext


def _noisy_chunks(seed, noise):
    rng = np.random.default_rng(seed)
    truth = _loop_truth(N_FRAMES)
    # chunks of 6 frames, 3 shared with the next; the last wraps to close the loop
    idx = [np.arange(s, s + 6) % N_FRAMES for s in range(0, N_FRAMES, 3)]
    exts = []
    for ix in idx:
        e = truth[ix].copy()
        c = camera_centers(e) + rng.normal(scale=noise, size=(len(ix), 3))
        e[:, :, 3] = -np.einsum("nij,nj->ni", e[:, :3, :3], c)
        exts.append(transform_extrinsics(e, _rand_sim(rng)))
    return truth, exts, idx


def _global_error(truth, exts, idx, sims):
    est = np.zeros((N_FRAMES, 3))
    seen = np.zeros(N_FRAMES, bool)
    for e, ix, s in zip(exts, idx, sims, strict=True):
        c = s.apply(camera_centers(e))
        for f, p in zip(ix, c, strict=True):
            if not seen[f]:
                est[f], seen[f] = p, True
    gt = camera_centers(truth)
    fit = umeyama(est, gt)  # remove the free global gauge
    return float(np.sqrt(((fit.apply(est) - gt) ** 2).sum(1).mean()))


def _chain(exts, idx):
    edges = [e for e in build_edges(exts, idx) if e.j == e.i + 1]
    return _chain_init(len(exts), edges)


def test_pose_graph_beats_chain_with_loop_closure():
    chain_err, graph_err = [], []
    for seed in range(6):
        truth, exts, idx = _noisy_chunks(seed, noise=0.05)
        chain_err.append(_global_error(truth, exts, idx, _chain(exts, idx)))
        res = solve_pose_graph(exts, idx)
        assert not res.is_tree
        graph_err.append(_global_error(truth, exts, idx, res.sims))
    assert np.mean(graph_err) < np.mean(chain_err)
    assert sum(g < c for g, c in zip(graph_err, chain_err, strict=True)) >= 5


def test_noise_free_graph_is_exact():
    truth, exts, idx = _noisy_chunks(0, noise=0.0)
    res = solve_pose_graph(exts, idx)
    assert _global_error(truth, exts, idx, res.sims) < 1e-6
    assert res.sims[0].s == 1.0


def test_tree_graph_is_chain_exactly():
    truth, exts, idx = _noisy_chunks(1, noise=0.05)
    exts, idx = exts[:-1], idx[:-1]  # drops the wrap chunk: a path graph
    res = solve_pose_graph(exts, idx)
    assert res.is_tree
    for a, b in zip(res.sims, _chain(exts, idx), strict=True):
        assert a.s == b.s and np.array_equal(a.R, b.R) and np.array_equal(a.t, b.t)


def test_disconnected_graph_raises_stitch_error():
    _, exts, idx = _noisy_chunks(2, noise=0.0)
    idx = [ix + 100 * (k // 3) for k, ix in enumerate(idx)]  # two islands, no shared ids
    with pytest.raises(ChunkAlignmentError, match="disconnected"):
        solve_pose_graph(exts, idx)


def _stitch(fake, mode, n=12):
    r = ChunkedReconstructor(fake, chunk_size=6, overlap=2, stitch=mode)
    return r.reconstruct(_frames(n))


def test_reconstructor_tree_identical_in_both_modes():
    a = _stitch(SimFake(12, seed=3), "chain")
    b = _stitch(SimFake(12, seed=3), "pose_graph")
    assert np.array_equal(a.extrinsics, b.extrinsics)
    assert np.array_equal(a.world_points, b.world_points)


def test_reconstructor_pose_graph_with_redundant_edges_is_consistent():
    # overlap 4 of cap 6: chunk i also shares frames with chunk i+2, so the graph has cycles
    fake = SimFake(12, seed=4)
    out = ChunkedReconstructor(fake, 6, 4, stitch="pose_graph").reconstruct(_frames(12))
    fit = umeyama(camera_centers(out.extrinsics), camera_centers(fake.ext))
    err = np.abs(fit.apply(camera_centers(out.extrinsics)) - camera_centers(fake.ext)).max()
    assert err < 1e-6


def test_config_default_and_validation():
    assert Config().chunk_stitch == "chain"
    assert Config(chunk_stitch="pose_graph").chunk_stitch == "pose_graph"
    with pytest.raises(ValueError):
        Config(chunk_stitch="bogus")
    with pytest.raises(ValueError, match="stitch"):
        ChunkedReconstructor(SimFake(4), 3, 2, stitch="bogus")
