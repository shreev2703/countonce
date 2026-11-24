"""Synthetic geometry and cannot-link tests (docs/PLAN.md §18)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from countonce.lift import compute_scene_scale, lift_detections
from countonce.merge.cluster import cluster_detections
from countonce.types import Detection, Frame, LiftedDetection, Recon


def _plane_recon(h=60, w=80, f=100.0, z=5.0, n_views=2):
    """Camera(s) at identity looking down +z at the plane z=`z`."""
    ys, xs = np.mgrid[0:h, 0:w]
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    depth = np.full((n_views, h, w), z)
    pts = np.stack([(xs + 0.5 - w / 2) * z / f, (ys + 0.5 - h / 2) * z / f, np.full((h, w), z)], -1)
    ext = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (n_views, 1, 1))
    return Recon(
        backbone="vggt",
        intrinsics=np.tile(k, (n_views, 1, 1)),
        extrinsics=ext,
        depth=depth,
        depth_conf=np.full((n_views, h, w), 5.0),
        world_points=np.tile(pts, (n_views, 1, 1, 1)),
        world_points_conf=np.full((n_views, h, w), 5.0),
    )


def test_lift_box_center_and_size_on_plane():
    recon = _plane_recon()
    det = Detection(id="0:0", frame_id=0, score=0.9, box_xyxy=np.array([20.0, 10.0, 60.0, 50.0]))
    (ld,) = lift_detections([det], recon)
    assert ld.ok
    expected = np.array([(40 - 40) * 5 / 100, (30 - 30) * 5 / 100, 5.0])
    np.testing.assert_allclose(ld.xyz, expected, atol=0.05)
    assert ld.size == pytest.approx(40 * 5.0 / 100.0)  # 2 m wide box
    assert 0 < ld.lift_conf <= 1


def test_lift_point_mask_and_failures():
    recon = _plane_recon()
    mask = np.zeros((60, 80), bool)
    mask[20:30, 50:60] = True
    dets = [
        Detection(id="0:0", frame_id=0, score=0.9, point_xy=np.array([40.0, 30.0])),
        Detection(id="0:1", frame_id=0, score=0.9, mask=mask),
        Detection(id="0:2", frame_id=0, score=0.1, point_xy=np.array([40.0, 30.0])),
    ]
    a, b, c = lift_detections(dets, recon, min_score=0.3)
    assert a.ok and b.ok
    np.testing.assert_allclose(b.xyz[:2], [(55 - 40) * 5 / 100, (25 - 30) * 5 / 100], atol=0.05)
    assert c.fail_reason == "low_conf" and c.xyz is None


def test_lift_privacy_mask_and_too_few_pixels():
    recon = _plane_recon()
    pm = np.zeros((60, 80), bool)
    pm[:, :40] = True
    frame = Frame(id=0, path="x.jpg", width=80, height=60, sha256="0", privacy_mask=pm)
    masked = Detection(id="0:0", frame_id=0, score=0.9, box_xyxy=np.array([4.0, 4.0, 30.0, 30.0]))
    tiny = Detection(id="0:1", frame_id=0, score=0.9, box_xyxy=np.array([60.0, 30.0, 61.0, 31.0]))
    a, b = lift_detections([masked, tiny], recon, frames={0: frame})
    assert a.fail_reason == "privacy_mask"
    assert b.fail_reason == "too_few_pixels"


def test_scene_scale_is_camera_distance():
    assert compute_scene_scale(_plane_recon()) == pytest.approx(5.0, rel=0.1)


def _ld(det_id, frame, xyz, size=1.0, conf=0.9):
    det = Detection(id=det_id, frame_id=frame, score=conf, point_xy=np.zeros(2))
    return LiftedDetection(det=det, xyz=np.asarray(xyz, float), size=size, lift_conf=conf)


def test_cannot_link_same_frame_even_when_coincident():
    res = cluster_detections([_ld("0:0", 0, [0, 0, 0]), _ld("0:1", 0, [0, 0, 0])])
    assert len(res.instances) == 2
    assert any(e.reason == "same_frame_cannot_link" and not e.merged for e in res.evidence)


def test_cannot_link_blocks_transitive_chain():
    # A(f0) - B(f1) - C(f0): A~B and B~C close, but A and C share a frame.
    res = cluster_detections(
        [_ld("0:0", 0, [0, 0, 0]), _ld("1:0", 1, [0.3, 0, 0]), _ld("0:1", 0, [0.6, 0, 0])], k=0.5
    )
    assert len(res.instances) == 2
    for inst in res.instances:
        frames = [m.split(":")[0] for m in inst.members]
        assert len(frames) == len(set(frames))


def test_merge_status_and_failed_lifts():
    failed = LiftedDetection(
        det=Detection(id="2:0", frame_id=2, score=0.5, point_xy=np.zeros(2)), fail_reason="low_conf"
    )
    res = cluster_detections(
        [_ld("0:0", 0, [0, 0, 0], conf=0.9), _ld("1:0", 1, [0.1, 0, 0], conf=0.5), failed]
    )
    assert len(res.instances) == 1 and res.instances[0].n_views == 2
    assert res.detection_status == {"0:0": "new", "1:0": "duplicate", "2:0": "rejected"}
    assert res.reject_reasons["2:0"] == "low_conf"


@pytest.mark.parametrize("noise", [0.0, 0.03, 0.08])
def test_synthetic_geometry_recovers_count(noise):
    """N random 3D objects seen from several views with noise, drops and false positives."""
    rng = np.random.default_rng(0)
    n_obj, n_views, size = 40, 6, 1.0
    # Well-separated objects: jittered grid so spacing >> size.
    grid = np.array([(i, j, 0) for i in range(8) for j in range(5)], float) * 4.0
    pts = grid[:n_obj] + rng.uniform(-0.5, 0.5, (n_obj, 3))
    lifted = []
    seen_any = set()
    for v in range(n_views):
        for o in range(n_obj):
            if rng.random() < 0.25:  # random drop
                continue
            seen_any.add(o)
            xyz = pts[o] + rng.normal(0, noise * size, 3)
            lifted.append(_ld(f"{v}:{len(lifted)}", v, xyz, size=size))
        for _ in range(2):  # false positives, isolated in space
            xyz = rng.uniform(-10, 40, 3) + [0, 0, 20]
            lifted.append(_ld(f"{v}:{len(lifted)}", v, xyz, size=size, conf=0.3))
    res = cluster_detections(lifted, k=0.5)
    # Isolated false positives each become one instance (counted with a low conf).
    confident = [i for i in res.instances if i.n_views >= 2]
    assert abs(len(confident) - len(seen_any)) <= 0.1 * n_obj
    for inst in res.instances:
        frames = [m.split(":")[0] for m in inst.members]
        assert len(frames) == len(set(frames))


def test_count_nonincreasing_in_k():
    rng = np.random.default_rng(1)
    lifted = [_ld(f"{i % 4}:{i}", i % 4, rng.normal(0, 2, 3)) for i in range(60)]
    counts = [len(cluster_detections(lifted, k=k).instances) for k in (0.1, 0.3, 0.5, 1.0, 2.0)]
    assert counts == sorted(counts, reverse=True)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(1, 40),
    n_frames=st.integers(1, 5),
    ks=st.lists(st.floats(0.05, 5.0), min_size=2, max_size=6),
)
def test_v1_count_never_increases_with_k(seed, n, n_frames, ks):
    """Property: a larger merge radius can only merge more, so the v1 count never grows."""
    rng = np.random.default_rng(seed)
    lifted = [
        _ld(f"{i % n_frames}:{i}", i % n_frames, rng.normal(0, 2, 3), size=rng.uniform(0.3, 1.5))
        for i in range(n)
    ]
    counts = [len(cluster_detections(lifted, k=k).instances) for k in sorted(ks)]
    assert all(a >= b for a, b in zip(counts, counts[1:], strict=False))
