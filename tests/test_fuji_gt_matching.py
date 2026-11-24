"""Sim(3) alignment and GT identity matching on a synthetic fixture (known transform, known ids)."""

from __future__ import annotations

import numpy as np
import pytest

from bench.datasets.fuji_sfm import FujiScene, align_recon_to_gt, load_fuji_sfm
from bench.datasets.validate import DataQualityError
from bench.gt_matching import match_detections_to_gt, project_points
from countonce.recon.align import (
    Sim3,
    camera_centers,
    transform_extrinsics,
    umeyama,
)
from countonce.types import Detection


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera (3,4), +z forward, y down."""
    f = target - center
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return np.concatenate([R, (-R @ center)[:, None]], axis=1)


SIM = Sim3(2.5, _rot(np.array([0.3, 1.0, -0.4]), 0.8), np.array([1.0, -2.0, 3.0]))
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
SIZE = (640, 480)


def _scene(n_gt: int = 12, n_cam: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    gt = rng.uniform(-1, 1, (n_gt, 3))
    ang = np.linspace(0, np.pi, n_cam)
    ext = np.stack([_look_at(5 * np.array([np.cos(a), np.sin(a), 0.4]), np.zeros(3)) for a in ang])
    return gt, ext


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    est = umeyama(src, SIM.apply(src))
    assert est.s == pytest.approx(SIM.s, rel=1e-9)
    np.testing.assert_allclose(est.R, SIM.R, atol=1e-9)
    np.testing.assert_allclose(est.t, SIM.t, atol=1e-9)
    assert np.linalg.det(est.R) == pytest.approx(1.0)


def test_umeyama_noise_reflection_and_errors():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(200, 3))
    est = umeyama(src, SIM.apply(src) + rng.normal(scale=0.01, size=src.shape))
    assert est.s == pytest.approx(SIM.s, rel=1e-2)
    # mirrored data must still yield a proper rotation
    assert np.linalg.det(umeyama(src, src * [1, 1, -1]).R) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        umeyama(src[:2], src[:2])
    with pytest.raises(ValueError):
        umeyama(np.ones((5, 3)), np.ones((5, 3)))


def test_sim3_inverse_compose_matrix():
    pts = np.random.default_rng(3).normal(size=(5, 3))
    np.testing.assert_allclose(SIM.inverse().apply(SIM.apply(pts)), pts, atol=1e-9)
    np.testing.assert_allclose(SIM.compose(SIM.inverse()).apply(pts), pts, atol=1e-9)
    h = np.c_[pts, np.ones(5)] @ SIM.matrix().T
    np.testing.assert_allclose(h[:, :3], SIM.apply(pts), atol=1e-9)


def test_transform_extrinsics_matches_point_transform():
    gt, ext = _scene()
    new = transform_extrinsics(ext, SIM)
    for i in range(len(ext)):
        uv_src, z_src = project_points(gt, K, ext[i])
        uv_dst, z_dst = project_points(SIM.apply(gt), K, new[i])
        np.testing.assert_allclose(uv_dst, uv_src, atol=1e-6)
        np.testing.assert_allclose(z_dst, SIM.s * z_src, atol=1e-6)
    np.testing.assert_allclose(camera_centers(new), SIM.apply(camera_centers(ext)), atol=1e-9)


def _dets_for(gt: np.ndarray, ext: np.ndarray, frame: int, idx: list[int], jitter: float = 2.0):
    uv, _ = project_points(gt[idx], K, ext[frame])
    rng = np.random.default_rng(frame)
    return [
        Detection(
            id=f"{frame}:{k}",
            frame_id=frame,
            score=0.9,
            point_xy=uv[k] + rng.normal(scale=jitter, size=2),
        )
        for k in range(len(idx))
    ]


def _maps(ext: np.ndarray):
    n = len(ext)
    return ({i: K for i in range(n)}, {i: ext[i] for i in range(n)}, {i: SIZE for i in range(n)})


def test_matching_recovers_known_identities():
    gt, ext = _scene()
    ids = [f"g{i}" for i in range(len(gt))]
    Ks, Es, Ss = _maps(ext)
    dets, truth = [], {}
    for f in range(len(ext)):
        uv, z = project_points(gt, K, ext[f])
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < SIZE[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE[1])
        vis = [int(i) for i in np.flatnonzero((z > 0) & inside)]
        for d, i in zip(_dets_for(gt, ext, f, vis), vis, strict=True):
            dets.append(d)
            truth[d.id] = ids[i]
    assert len(dets) > 30
    got = match_detections_to_gt(dets, ids, gt, Ks, Es, Ss, threshold_px=15)
    assert got == truth


def test_matching_threshold_box_center_and_missing_camera():
    gt, ext = _scene(n_gt=3, n_cam=3)
    ids = ["a", "b", "c"]
    Ks, Es, Ss = _maps(ext)
    uv, _ = project_points(gt, K, ext[0])
    far = Detection(id="0:far", frame_id=0, score=1, point_xy=uv[0] + 100)
    box = Detection(id="0:box", frame_id=0, score=1, box_xyxy=np.r_[uv[1] - 5, uv[1] + 5])
    nocam = Detection(id="9:x", frame_id=9, score=1, point_xy=uv[0])
    got = match_detections_to_gt([far, box, nocam], ids, gt, Ks, Es, Ss, threshold_px=20)
    assert got == {"0:far": None, "0:box": "b", "9:x": None}


def test_matching_ignores_points_behind_camera():
    gt = np.array([[0.0, 0.0, -5.0]])  # behind a camera at the origin looking along +z
    ext = np.eye(4)[:3]
    d = Detection(id="0:0", frame_id=0, score=1, point_xy=np.array([320.0, 240.0]))
    got = match_detections_to_gt([d], ["a"], gt, {0: K}, {0: ext}, {0: SIZE})
    assert got == {"0:0": None}


def test_duplicate_detections_share_identity():
    gt, ext = _scene(n_gt=1, n_cam=1)
    Ks, Es, Ss = _maps(ext)
    uv, _ = project_points(gt, K, ext[0])
    dets = [Detection(id=f"0:{k}", frame_id=0, score=1, point_xy=uv[0] + k) for k in range(2)]
    got = match_detections_to_gt(dets, ["a"], gt, Ks, Es, Ss)
    assert set(got.values()) == {"a"}


def test_align_recon_to_gt_end_to_end():
    """Recon lives in an arbitrary Sim(3) of the GT frame; alignment recovers GT-frame matching."""
    gt, ext = _scene()
    recon_ext = transform_extrinsics(ext, SIM.inverse())  # recon world = SIM^-1 (GT world)
    scene = FujiScene(
        root=None,  # type: ignore[arg-type]
        image_paths=[],
        gt_ids=[f"g{i}" for i in range(len(gt))],
        gt_xyz=gt,
        extrinsics={i: ext[i] for i in range(len(ext))},
    )
    frame_ids = list(range(len(ext)))
    sim, moved = align_recon_to_gt(scene, recon_ext, frame_ids)
    assert sim.s == pytest.approx(SIM.s, rel=1e-6)
    np.testing.assert_allclose(sim.R, SIM.R, atol=1e-6)
    for f in frame_ids:
        np.testing.assert_allclose(moved[f], ext[f], atol=1e-6)
    with pytest.raises(ValueError):
        align_recon_to_gt(scene, recon_ext[:2], [0, 1])


def test_loader_missing_and_synthetic_layout(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fuji_sfm(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        load_fuji_sfm(tmp_path)
    (tmp_path / "images").mkdir()
    for n in ("a", "b"):
        (tmp_path / "images" / f"{n}.jpg").write_bytes(b"x")  # fake bytes: fail the gates
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "b.png").write_bytes(b"x")
    np.save(tmp_path / "gt_apples.npy", np.zeros((4, 3)))
    with pytest.raises(DataQualityError):
        load_fuji_sfm(tmp_path)
    from PIL import Image

    for i, n in enumerate(("a", "b")):
        Image.fromarray(np.full((8, 8, 3), 40 * (i + 1), dtype=np.uint8)).save(
            tmp_path / "images" / f"{n}.jpg"
        )
    s = load_fuji_sfm(tmp_path)
    assert len(s.image_paths) == 2 and len(s.gt_ids) == 4 and list(s.mask_paths) == [1]
