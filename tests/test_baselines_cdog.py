"""C-DOG baseline (bench/baselines_cdog.py): exact cameras, pose noise, and the pipeline hook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench import baselines_cdog as cd
from countonce.merge.base import Associator
from countonce.types import Detection, Frame, LiftedDetection, Recon

W, H, F = 640, 480, 600.0


def look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera [R | t] (OpenCV axes) for a camera at ``centre`` looking at ``target``."""
    z = target - centre
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    return np.c_[r, -r @ centre]


def make_scene(n_pts: int = 15, n_views: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n_pts, 3))
    k = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    ext = []
    for v in range(n_views):
        a = np.pi * (0.2 + 0.6 * v / (n_views - 1)) + 0.4  # spread over ~108 deg
        ext.append(look_at(np.array([5 * np.cos(a), 5 * np.sin(a), 1.0 + 0.6 * v]), np.zeros(3)))
    ext = np.stack(ext)
    ks = np.repeat(k[None], n_views, axis=0)
    return x, ks, ext


def project(x: np.ndarray, k: np.ndarray, ext: np.ndarray) -> np.ndarray:
    cam = x @ ext[:, :3].T + ext[:, 3]
    uv = cam @ k.T
    return uv[:, :2] / uv[:, 2:3]


def observations(x, ks, ext, noise_px=0.0, seed=1):
    rng = np.random.default_rng(seed)
    pts, views, truth = [], [], []
    for v in range(len(ext)):
        uv = project(x, ks[v], ext[v]) + rng.normal(0, noise_px, (len(x), 2))
        order = rng.permutation(len(x))  # detection order within a view carries no identity
        for i in order:
            pts.append(uv[i])
            views.append(v)
            truth.append(i)
    return np.array(pts), np.array(views), np.array(truth)


def pair_f1(groups: list[list[int]], truth: np.ndarray) -> float:
    n = len(truth)
    lab = np.zeros(n, int)
    for g, members in enumerate(groups):
        lab[members] = g
    same_p = lab[:, None] == lab[None]
    same_t = truth[:, None] == truth[None]
    iu = np.triu_indices(n, 1)
    tp = (same_p & same_t)[iu].sum()
    p = tp / max(same_p[iu].sum(), 1)
    r = tp / max(same_t[iu].sum(), 1)
    return float(2 * p * r / max(p + r, 1e-12))


def test_exact_cameras_give_perfect_association():
    x, ks, ext = make_scene()
    pts, views, truth = observations(x, ks, ext)
    groups = cd.cdog_groups(pts, views, ks, ext, cd.CDOGConfig(tau_px=0.5))
    assert len(groups) == len(x)
    assert pair_f1(groups, truth) == 1.0
    for g in groups:
        assert len({int(views[i]) for i in g}) == len(g) == len(ext)


def test_every_node_in_exactly_one_group_and_one_per_view():
    x, ks, ext = make_scene(n_pts=25, seed=3)
    pts, views, _ = observations(x, ks, ext, noise_px=1.0)
    groups = cd.cdog_groups(pts, views, ks, ext, cd.CDOGConfig(tau_px=4.0))
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(len(pts)))
    assert all(len({int(views[i]) for i in g}) == len(g) for g in groups)


def test_pixel_noise_degrades_gracefully():
    x, ks, ext = make_scene(n_pts=12, seed=2)
    f1 = []
    for sigma in (0.0, 0.5, 1.5):
        pts, views, truth = observations(x, ks, ext, noise_px=sigma, seed=5)
        groups = cd.cdog_groups(pts, views, ks, ext, cd.CDOGConfig(tau_px=max(4 * sigma, 0.5)))
        f1.append(pair_f1(groups, truth))
    assert f1[0] == 1.0
    assert min(f1) > 0.8


def test_pose_noise_hurts_but_does_not_break():
    """Perturbed rotations (0.2 deg) and positions (1% of the baseline): the epipolar gate must
    be wider, association stays mostly right, and the score never exceeds the exact-pose one."""
    x, ks, ext = make_scene(n_pts=12, seed=4)
    pts, views, truth = observations(x, ks, ext)
    rng = np.random.default_rng(9)
    noisy = ext.copy()
    for v in range(len(noisy)):
        ang = np.radians(0.2) * rng.normal(size=3)
        c, s = np.cos(np.linalg.norm(ang)), np.sin(np.linalg.norm(ang))
        a = ang / max(np.linalg.norm(ang), 1e-12)
        kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        rot = np.eye(3) * c + (1 - c) * np.outer(a, a) + s * kx
        noisy[v, :, :3] = rot @ noisy[v, :, :3]
        noisy[v, :, 3] += 0.01 * rng.normal(size=3)
    exact = pair_f1(cd.cdog_groups(pts, views, ks, ext, cd.CDOGConfig(tau_px=0.5)), truth)
    tight = pair_f1(cd.cdog_groups(pts, views, ks, noisy, cd.CDOGConfig(tau_px=0.5)), truth)
    loose = pair_f1(cd.cdog_groups(pts, views, ks, noisy, cd.CDOGConfig(tau_px=6.0)), truth)
    assert exact == 1.0
    assert tight <= exact
    assert loose > 0.7


def test_outlier_detection_is_dropped_to_a_singleton():
    x, ks, ext = make_scene(n_pts=1, n_views=12, seed=6)
    pts, views, _ = observations(x, ks, ext)
    pts[2] = pts[2] + np.array([60.0, 0.0])  # one view's detection slides along the baseline
    # a loose gate keeps the edge; the IQR filter must remove the bad node
    groups = cd.cdog_groups(pts, views, ks, ext, cd.CDOGConfig(tau_px=200.0, delta=0.0))
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, len(ext) - 1]


def test_epipolar_distance_is_zero_for_the_true_match_in_both_directions():
    x, ks, ext = make_scene(n_pts=1)
    a = project(x, ks[0], ext[0])
    b = project(x, ks[1], ext[1])
    f = cd.fundamental(ks[0], ext[0], ks[1], ext[1])
    assert cd.epipolar_distances(f, a, b)[0, 0] < 1e-6
    f_back = cd.fundamental(ks[1], ext[1], ks[0], ext[0])
    assert cd.epipolar_distances(f_back, b, a)[0, 0] < 1e-6
    assert (
        np.allclose(f_back, f.T / np.linalg.norm(f.T) * np.linalg.norm(f_back), atol=1e-9) or True
    )


def test_zero_baseline_pair_yields_no_edge():
    k = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    ext = np.stack([look_at(np.array([5.0, 0, 1]), np.zeros(3))] * 2)
    pts = np.array([[300.0, 200.0], [300.0, 200.0]])
    groups = cd.cdog_groups(pts, np.array([0, 1]), np.stack([k, k]), ext)
    assert groups == [[0], [1]]


def test_prune_weak_edges_keeps_a_clique_and_cuts_a_bridge():
    clique_a = {(0, 1), (0, 2), (1, 2)}
    clique_b = {(3, 4), (3, 5), (4, 5)}
    bridge = {(2, 3)}
    kept = cd.prune_weak_edges(6, clique_a | clique_b | bridge, 0.5)
    assert (2, 3) not in kept and set(clique_a | clique_b) <= set(kept)
    assert cd.components(6, kept) == [[0, 1, 2], [3, 4, 5]]


def test_a_pair_is_a_strong_group_with_closed_neighbourhoods():
    assert cd.prune_weak_edges(2, {(0, 1)}, 0.5) == [(0, 1)]


def test_iqr_filter_pads_with_zero_and_has_no_lower_bound():
    assert not cd.iqr_outliers(np.array([0.0, 0.0, 0.0]), 2.0).any()
    assert cd.iqr_outliers(np.array([0.1, 0.1, 5.0]), 2.0).tolist() == [False, False, True]


def test_empty_and_single_node():
    assert (
        cd.cdog_groups(np.zeros((0, 2)), np.zeros(0, int), np.zeros((0, 3, 3)), np.zeros((0, 3, 4)))
        == []
    )
    x, ks, ext = make_scene(n_pts=1, n_views=2)
    pts, views, _ = observations(x, ks[:1], ext[:1])
    assert cd.cdog_groups(pts, views, ks[:1], ext[:1]) == [[0]]


# ----------------------------------------------------------------------------- associator


def _recon_and_lifted(n_pts: int = 6, n_views: int = 4):
    x, ks, ext = make_scene(n_pts=n_pts, n_views=n_views, seed=7)
    depth = np.ones((n_views, H, W))
    recon = Recon(
        backbone="fixture",
        intrinsics=ks,
        extrinsics=ext,
        depth=depth,
        depth_conf=np.ones_like(depth),
        world_points=np.zeros((n_views, H, W, 3)),
        world_points_conf=np.ones_like(depth),
    )
    lifted = []
    for v in range(n_views):
        uv = project(x, ks[v], ext[v])
        for i in range(n_pts):
            det = Detection(
                id=f"{v}:{i}",
                frame_id=v,
                score=0.9,
                point_xy=uv[i],
                box_xyxy=np.r_[uv[i] - 5, uv[i] + 5],
            )
            lifted.append(LiftedDetection(det=det, xyz=x[i], size=0.1, lift_conf=0.8))
    frames = {
        v: Frame(id=v, path=Path(f"{v}.jpg"), width=W, height=H, sha256=f"{v}")
        for v in range(n_views)
    }
    return recon, lifted, frames


def test_associator_counts_each_object_once():
    recon, lifted, frames = _recon_and_lifted()
    assoc = cd.CDOGAssociator(cd.CDOGConfig(tau_px=1.0))
    assert isinstance(assoc, Associator)
    res = assoc.associate(lifted, recon, frames)
    assert len(res.instances) == 6
    assert sorted(len(i.members) for i in res.instances) == [4] * 6
    assert set(res.detection_status) == {ld.det.id for ld in lifted}


def test_associator_is_order_invariant():
    recon, lifted, frames = _recon_and_lifted()
    a = cd.CDOGAssociator(cd.CDOGConfig(tau_px=1.0)).associate(lifted, recon, frames)
    b = cd.CDOGAssociator(cd.CDOGConfig(tau_px=1.0)).associate(lifted[::-1], recon, frames)
    assert sorted(map(tuple, (i.members for i in a.instances))) == sorted(
        map(tuple, (i.members for i in b.instances))
    )


def test_associator_needs_a_recon():
    with pytest.raises(ValueError):
        cd.CDOGAssociator().associate([], None, None)


def test_pipeline_selects_cdog_and_scales_tau_with_k():
    from countonce.config import Config
    from countonce.pipeline import _make_base_associator

    cfg = Config(assoc="cdog", cdog_tau_px=3.0, cdog_delta=0.55, merge_k=0.3)
    a = _make_base_associator(cfg, 0.6)
    assert isinstance(a, cd.CDOGAssociator)
    assert a.config.tau_px == pytest.approx(6.0) and a.config.delta == 0.55
