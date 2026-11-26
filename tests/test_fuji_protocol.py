"""CPU tests for the Fuji Core protocol fixes (sixteenth audit): visibility, windows, scale
alignment, identity matching, interval recording and MAPE with GT = 0."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bench import fuji_core as fc
from bench import fuji_report as fr
from countonce.recon.align import Sim3, transform_extrinsics
from countonce.types import Instance, Interval

W, H = 64, 48
K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
EXT = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)  # camera at origin looking +z


def _depth(val: float) -> np.ndarray:
    return np.full((1, H, W), val)


# ----------------------------------------------------------------------------- visible_gt


def test_visible_gt_occlusion_depth_test() -> None:
    gt = np.array([[0.0, 0.0, 2.0], [0.3, 0.0, 2.0]])
    surf = _depth(2.0)
    surf[0, :, 40:] = 1.0  # foliage in front of the second apple (it projects to u=39.5 -> col 40)
    got = fc.visible_gt(gt, K[None], EXT[None], surf)
    assert got[0].tolist() == [0]


def test_visible_gt_includes_last_row_col_and_rounds() -> None:
    # u = 50*x/z + 32 = 63.0 (last column), v = 47.0 (last row)
    gt = np.array([[31.0 / 50 * 2, 23.0 / 50 * 2, 2.0]])
    assert fc.visible_gt(gt, K[None], EXT[None], _depth(2.0))[0].tolist() == [0]
    # u = 62.6 rounds to column 63, not truncated to 62
    d = _depth(9.0)
    d[0, :, 63] = 1.0  # occluder only in column 63
    gt = np.array([[30.6 / 50 * 2, 0.0, 2.0]])
    assert fc.visible_gt(gt, K[None], EXT[None], d)[0].tolist() == []


def test_visible_gt_invalid_depth_is_not_occlusion() -> None:
    gt = np.array([[0.0, 0.0, 2.0]])
    for bad in (np.nan, 0.0, np.inf, -1.0):
        assert fc.visible_gt(gt, K[None], EXT[None], _depth(bad))[0].tolist() == [0]


def test_visible_gt_range_and_behind_camera() -> None:
    gt = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -1.0], [5.0, 0.0, 2.0]])
    assert fc.visible_gt(gt, K[None], EXT[None], _depth(50.0))[0].tolist() == []


# ----------------------------------------------------------------------------- window_for_tree


def test_window_centred_and_clamped() -> None:
    cam_y = np.linspace(0, 10, 11)
    assert fc.window_for_tree(cam_y, 11, 5.0, 5) == (3, 8)
    assert fc.window_for_tree(cam_y, 11, -3.0, 5) == (0, 5)
    assert fc.window_for_tree(cam_y, 11, 99.0, 5) == (6, 11)
    assert fc.window_for_tree(cam_y, 11, 5.0, 11) == (0, 11)


def test_window_longer_than_side_raises() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        fc.window_for_tree(np.arange(3.0), 3, 1.0, 5)
    with pytest.raises(ValueError):
        fc.window_for_tree(np.arange(3.0), 3, 1.0, 0)
    with pytest.raises(ValueError, match="entries"):
        fc.window_for_tree(np.arange(4.0), 5, 1.0, 3)
    with pytest.raises(ValueError, match="non-finite"):
        fc.window_for_tree(np.array([0.0, np.nan, 2.0]), 3, 1.0, 2)


# ----------------------------------------------------------------------------- scale alignment


def _poses(n: int) -> np.ndarray:
    ext = np.repeat(EXT[None], n, axis=0).copy()
    ext[:, 0, 3] = -np.linspace(0, 1.0, n)  # camera centres move along +x
    return ext


def test_scaled_depth_matches_projected_z() -> None:
    sim = Sim3(0.4, np.eye(3), np.array([1.0, 2.0, 3.0]))
    ext_src = _poses(4)
    ext_gt = transform_extrinsics(ext_src, sim)
    x_src = np.array([0.2, -0.1, 3.0])  # a point in the reconstruction frame
    x_gt = sim.apply(x_src[None])[0]
    from bench.gt_matching import project_points

    _, z = project_points(x_gt[None], K, ext_gt[0])
    z_src = (x_src @ ext_src[0, :, :3].T + ext_src[0, :, 3])[2]
    assert float(fc.scaled_depth(np.array([z_src]), sim)[0]) == pytest.approx(float(z[0]))


def test_check_alignment_flags_bad_scale_and_bad_fit() -> None:
    ext = _poses(5)
    sim = Sim3(2.0, np.eye(3), np.zeros(3))
    gt_ext = transform_extrinsics(ext, sim)
    info = fc.check_alignment(gt_ext, ext, sim)
    assert info["cam_rms_m"] < 1e-9 and info["cam_span_m"] == pytest.approx(2.0)
    with pytest.raises(fc.AlignmentError, match="misfit"):
        fc.check_alignment(gt_ext, ext, Sim3(0.2, np.eye(3), np.zeros(3)))
    with pytest.raises(fc.AlignmentError, match="non-positive"):
        fc.check_alignment(gt_ext, ext, Sim3(-1.0, np.eye(3), np.zeros(3)))
    assert info["cam_rot_deg"] < 1e-4
    # a roll about the camera path leaves every centre in place, so only the rotation gate sees it
    roll = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    straight = np.tile(np.eye(3, 4), (5, 1, 1))
    straight[:, 0, 3] = -np.linspace(0, 1.0, 5)
    with pytest.raises(fc.AlignmentError, match="rotation"):
        fc.check_alignment(straight, straight, Sim3(1.0, roll, np.zeros(3)))


# ----------------------------------------------------------------------------- evaluate_window


def _record(i: int, uv: tuple[float, float], xyz: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(id=f"d{i}", view=0, uv=uv, xyz=xyz, score=1.0)


def _window_result(interval: Interval | None, count: int, insts: list, dets: list):
    return SimpleNamespace(
        instances=insts,
        detections=dets,
        count=count,
        interval=interval,
    )


def _recon(depth_val: float):
    return SimpleNamespace(
        extrinsics=EXT[None],
        intrinsics=K[None],
        depth=_depth(depth_val),
        n_views=1,
    )


def _inst(i: int, xyz: tuple[float, float, float], members: list[str]) -> Instance:
    inst = SimpleNamespace(id=i, xyz=np.array(xyz), members=members, n_views=1)
    return inst  # type: ignore[return-value]


BOX = (np.array([-10.0, -10, 0]), np.array([10.0, 10, 10]))


def test_identity_not_stolen_by_visible_neighbour() -> None:
    # Apple A (occluded) projects at u=32; apple B (visible) at u=36, 4 px away. A detection on
    # A's pixel must stay unlabeled, not be assigned to B.
    gt = np.array([[0.0, 0.0, 2.0], [0.16, 0.0, 2.0]])  # u = 32, 36
    depth = _depth(2.0)
    depth[0, :, 30:34] = 1.0  # foliage over A only
    recon = SimpleNamespace(
        extrinsics=EXT[None],
        intrinsics=K[None],
        depth=depth,
        n_views=1,
    )
    dets = [_record(0, (32.0, 24.0), (0.0, 0.0, 2.0))]
    res = _window_result(None, 1, [_inst(0, (0.0, 0.0, 2.0), ["d0"])], dets)
    ev = fc.evaluate_window(res, recon, Sim3(1.0, np.eye(3), np.zeros(3)), ["A", "B"], gt, BOX)
    assert ev["gt"] == 1  # only B visible
    assert ev["n_labeled_dets"] == 0


def test_identity_labels_visible_apple() -> None:
    gt = np.array([[0.0, 0.0, 2.0], [0.16, 0.0, 2.0]])
    dets = [_record(0, (32.0, 24.0), (0.0, 0.0, 2.0)), _record(1, (36.0, 24.0), (0.16, 0.0, 2.0))]
    insts = [_inst(0, (0.0, 0.0, 2.0), ["d0"]), _inst(1, (0.16, 0.0, 2.0), ["d1"])]
    res = _window_result(None, 2, insts, dets)
    ev = fc.evaluate_window(
        res, _recon(2.0), Sim3(1.0, np.eye(3), np.zeros(3)), ["A", "B"], gt, BOX
    )
    assert (ev["pred"], ev["gt"], ev["n_labeled_dets"]) == (2, 2, 2)
    assert ev["duplicate_rate"] == 0 and ev["miss_rate"] == 0


def test_interval_recorded_and_shifted_for_roi() -> None:
    gt = np.array([[0.0, 0.0, 2.0]])
    dets = [_record(0, (32.0, 24.0), (0.0, 0.0, 2.0))]
    far = (50.0, 0.0, 2.0)  # outside the ROI box
    insts = [_inst(0, (0.0, 0.0, 2.0), ["d0"]), _inst(1, far, [])]
    iv = Interval(low=1, high=4, level=0.9, method="sensitivity")
    res = _window_result(iv, 2, insts, dets)
    ev = fc.evaluate_window(res, _recon(2.0), Sim3(1.0, np.eye(3), np.zeros(3)), ["A"], gt, BOX)
    assert ev["pred"] == 1
    assert (ev["low"], ev["high"]) == (0.0, 3.0)
    assert ev["low"] <= ev["pred"] <= ev["high"]
    out = fc.run_output(ev, "countonce_v1", SimpleNamespace(peak_vram_mb=0.0, timings={}), 1.0)
    assert (out.low, out.high) == (0.0, 3.0)
    base = fc.run_output(
        {**ev, "baselines": {"sum": 3}}, "sum", SimpleNamespace(peak_vram_mb=0.0, timings={}), 1.0
    )
    assert base.low is None and base.high is None


def test_interval_none_when_pipeline_has_none() -> None:
    gt = np.array([[0.0, 0.0, 2.0]])
    res = _window_result(None, 0, [], [])
    ev = fc.evaluate_window(res, _recon(2.0), Sim3(1.0, np.eye(3), np.zeros(3)), ["A"], gt, BOX)
    assert ev["low"] is None and ev["high"] is None


def test_evaluate_window_checks_alignment_when_given() -> None:
    gt = np.array([[0.0, 0.0, 2.0]])
    res = _window_result(None, 0, [], [])
    sim = Sim3(0.1, np.eye(3), np.zeros(3))
    recon = SimpleNamespace(
        extrinsics=_poses(5),
        intrinsics=np.repeat(K[None], 5, 0),
        depth=np.full((5, H, W), 2.0),
        n_views=5,
    )
    gt_ext = transform_extrinsics(_poses(5), Sim3(1.0, np.eye(3), np.zeros(3)))
    with pytest.raises(fc.AlignmentError):
        fc.evaluate_window(res, recon, sim, ["A"], gt, BOX, gt_ext)


# ----------------------------------------------------------------------------- MAPE, GT = 0


def _row(tree: str, side: str, pred: float, gt: float, method="countonce_v1", k=5) -> dict:
    return {
        "status": "ok",
        "method": method,
        "n_views": str(k),
        "scene_id": tree,
        "sample": side,
        "pred": str(pred),
        "gt": str(gt),
        "assoc_f1": "",
        "duplicate_rate": "",
        "miss_rate": "",
        "split": "test",
        "config_json": json.dumps({"protocol": fr.PROTOCOL}),
    }


def _all(gt_of) -> list[dict]:
    return [_row(t, s, 4, gt_of(t, s)) for t in fr.TEST_TREES for s in fr.SIDES]


def test_mape_skips_gt_zero_slices() -> None:
    rows = _all(lambda t, s: 0 if (t, s) == ("Tree01", "east") else 2)
    v = fr.tree_values(rows, "countonce_v1", 5, "mape")
    assert v is not None
    assert v["Tree01"] == pytest.approx(100.0)  # only the west side (|4-2|/2) counts
    assert v["Tree02"] == pytest.approx(100.0)
    assert len(v) == 8


def test_mape_drops_tree_with_gt_zero_on_both_sides() -> None:
    rows = _all(lambda t, s: 0 if t == "Tree03" else 2)
    v = fr.tree_values(rows, "countonce_v1", 5, "mape")
    assert v is not None and "Tree03" not in v and len(v) == 7
    cells = fr.compute(rows)["cells"]
    assert cells["countonce_v1|5|mape"]["n_trees"] == 7
    assert cells["countonce_v1|5|mae"]["n_trees"] == 8  # MAE still uses GT = 0 slices
    assert fr.tree_values(rows, "countonce_v1", 5, "mae")["Tree03"] == pytest.approx(4.0)


def test_mape_none_when_row_missing_or_all_gt_zero() -> None:
    rows = _all(lambda t, s: 2)[:-1]  # missing Tree08/west
    assert fr.tree_values(rows, "countonce_v1", 5, "mape") is None
    assert fr.tree_values(_all(lambda t, s: 0), "countonce_v1", 5, "mape") is None
