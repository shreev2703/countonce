"""CPU tests for ``bench/fuji_core.py`` and ``scripts/run_fuji_core.py`` on synthetic data."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bench import fuji_core as fc
from countonce.recon.align import Sim3
from countonce.types import DetectionRecord, Instance

REPO = Path(__file__).resolve().parents[1]

# Registration tolerances (stated, not tuned per case): scale 2 %, rotation 2 deg, translation 5 cm.
SCALE_RTOL = 0.02
ROT_DEG = 2.0
TRANS_M = 0.05


def _true_sim() -> Sim3:
    rot = Rotation.from_euler("zyx", [40, -25, 70], degrees=True).as_matrix()
    return Sim3(0.37, rot, np.array([3.0, -2.0, 1.5]))


def _assert_close(est: Sim3, true: Sim3) -> None:
    assert est.s == pytest.approx(true.s, rel=SCALE_RTOL)
    ang = np.degrees(np.arccos(np.clip((np.trace(est.R @ true.R.T) - 1) / 2, -1, 1)))
    assert ang < ROT_DEG
    assert np.linalg.norm(est.t - true.t) < TRANS_M


def _skewed_blob(rng: np.random.Generator, n: int) -> np.ndarray:
    """Anisotropic, asymmetric cloud (distinct PCA axes, skewed so signs are identifiable)."""
    x = rng.gamma(2.0, 1.5, n)
    y = rng.normal(0, 0.8, n) + 0.2 * x
    z = rng.normal(0, 0.3, n)
    return np.stack([x, y, z], axis=1)


def test_register_to_gt_recovers_sim3_with_noise_and_outliers() -> None:
    rng = np.random.default_rng(0)
    gt = _skewed_blob(rng, 120) * 2.0
    true = _true_sim()
    pick = rng.choice(len(gt), 90, replace=False)
    src = true.inverse().apply(gt[pick] + rng.normal(0, 0.01, (90, 3)))
    # 6 spurious instances (~6 %) near the tree. The PCA initialisation breaks down at ~12 %.
    c = gt.mean(0)
    lo, hi = c + (gt.min(0) - c) * 1.2, c + (gt.max(0) - c) * 1.2
    src = np.vstack([src, true.inverse().apply(rng.uniform(lo, hi, (6, 3)))])
    reg = fc.register_to_gt(src, gt)
    _assert_close(reg.sim, true)
    assert reg.n_src == len(src)
    assert reg.n_inliers >= 80
    assert reg.median_err_m < 0.03


def test_register_dense_recovers_sim3_with_noise_and_outliers() -> None:
    rng = np.random.default_rng(1)
    dst = _skewed_blob(rng, 6000)
    true = _true_sim()
    pick = rng.choice(len(dst), 2500, replace=False)
    src = true.inverse().apply(dst[pick] + rng.normal(0, 0.005, (2500, 3)))
    # 25 stray points (1 %); PCA-initialised trimmed ICP is not robust beyond a few percent.
    outliers = true.inverse().apply(rng.uniform(dst.min(0), dst.max(0), (25, 3)))
    src = np.vstack([src, outliers])
    reg = fc.register_dense(src, dst, iters=40)
    _assert_close(reg.sim, true)
    assert reg.inlier_frac > 0.5


def test_register_dense_is_memory_bounded() -> None:
    rng = np.random.default_rng(2)
    dst = _skewed_blob(rng, 20000)
    true = _true_sim()
    src = true.inverse().apply(dst[rng.choice(len(dst), 6000, replace=False)])
    # Cap far below the input size: still registers, because both clouds are thinned first.
    reg = fc.register_dense(src, dst, max_points=3000, iters=40)
    assert reg.n_src <= 3000
    assert reg.sim.s == pytest.approx(true.s, rel=0.05)


def test_subsample_cloud_caps_points_and_uses_float32() -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, (50_000, 3))
    out = fc.subsample_cloud(pts, voxel=0.0, max_points=1000)
    assert out.dtype == np.float32
    assert out.shape == (1000, 3)
    thinned = fc.subsample_cloud(np.repeat(pts[:100], 5, axis=0), voxel=0.01, max_points=10**6)
    assert len(thinned) <= 100  # duplicates share a voxel
    assert fc.subsample_cloud(np.zeros((0, 3))).shape == (0, 3)


def test_load_gt_cloud_streams_stride_and_caps(tmp_path: Path) -> None:
    d = tmp_path / "3-3D_data"
    d.mkdir()
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 5, (4000, 3))
    rows = [f"{x:.4f} {y:.4f} {z:.4f} 10 20 30" for x, y, z in pts]
    (d / "Fuji_apple_trees_point_cloud.txt").write_text("\n".join(rows) + "\n")
    cloud = fc.load_gt_cloud(tmp_path, every=4, voxel=0.0, max_points=500)
    assert cloud.dtype == np.float32
    assert cloud.shape == (500, 3)
    full = fc.load_gt_cloud(tmp_path, every=4, voxel=0.0)
    assert full.shape == (1000, 3)
    np.testing.assert_allclose(full[0], pts[0], atol=1e-3)
    np.testing.assert_allclose(full[1], pts[4], atol=1e-3)


# ------------------------------------------------------------------ windows / ROI / poses


def test_window_for_tree_centres_and_clamps() -> None:
    cam_y = np.arange(0.0, 20.0)  # 20 side-local frames at y = index
    assert fc.window_for_tree(cam_y, 20, 10.2, 4) == (8, 12)
    assert fc.window_for_tree(cam_y, 20, 10.2, 5) == (8, 13)
    assert fc.window_for_tree(cam_y, 20, -5.0, 5) == (0, 5)  # clamp at start
    assert fc.window_for_tree(cam_y, 20, 99.0, 5) == (15, 20)  # clamp at end
    assert fc.window_for_tree(cam_y, 20, 7.0, 20) == (0, 20)


def test_roi_box_and_in_roi() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 1.0]])
    lo, hi = fc.roi_box(gt, margin=0.5)
    np.testing.assert_allclose(lo, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(hi, [2.5, 4.5, 1.5])
    pts = np.array([[0, 0, 0], [2.5, 4.5, 1.5], [2.6, 0, 0], [0, -0.51, 0], [1, 1, 1.6]])
    assert fc.in_roi(pts, (lo, hi)).tolist() == [True, True, False, False, False]
    assert fc.in_roi(np.array([1.0, 1.0, 1.0]), (lo, hi)).tolist() == [True]  # single point
    default_lo, _ = fc.roi_box(gt)
    np.testing.assert_allclose(default_lo, -fc.ROI_MARGIN_M)


def _ext(rot: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """World-to-camera [R|t] for camera-to-world rotation ``rot`` at ``centre``."""
    rc = rot.T
    return np.concatenate([rc, (-rc @ centre)[:, None]], axis=1)


def test_interpolate_poses_linear_centres_slerp_rotations_and_clamping() -> None:
    r0 = np.eye(3)
    r1 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    c0, c1 = np.zeros(3), np.array([10.0, 0.0, 0.0])
    ext = np.stack([_ext(r0, c0), _ext(r1, c1)])
    out = fc.interpolate_poses(np.array([0, 10]), ext, np.array([0, 5, 10, 15, -3]))
    assert out.shape == (5, 3, 4)
    np.testing.assert_allclose(out[0], ext[0], atol=1e-9)
    np.testing.assert_allclose(out[2], ext[1], atol=1e-9)
    mid_rot = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    np.testing.assert_allclose(out[1], _ext(mid_rot, np.array([5.0, 0, 0])), atol=1e-9)
    np.testing.assert_allclose(out[3], ext[1], atol=1e-9)  # beyond range -> nearest end pose
    np.testing.assert_allclose(out[4], ext[0], atol=1e-9)


def test_visible_gt_depth_occlusion_frustum_and_range() -> None:
    k = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    ext = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    depth = np.full((1, 100, 100), 2.0)
    gt = np.array(
        [
            [0.0, 0.0, 1.0],  # 0 visible, in front of surface
            [0.0, 0.0, 3.0],  # 1 occluded (beyond surface + tol)
            [0.0, 0.0, -1.0],  # 2 behind camera
            [10.0, 0.0, 1.0],  # 3 outside image
            [0.0, 0.0, 2.1],  # 4 visible within tolerance of the surface
            [0.0, 0.0, 5.0],  # 5 beyond max depth
        ]
    )
    deep = np.full((1, 100, 100), 9.0)
    (vis,) = fc.visible_gt(gt, k[None], ext[None], depth)
    assert vis.tolist() == [0, 4]
    # Surface far behind everything: occlusion no longer applies, only the depth range does.
    (vis_deep,) = fc.visible_gt(gt, k[None], ext[None], deep)
    assert vis_deep.tolist() == [0, 1, 4]  # point 5 (z=5) is past GT_MAX_DEPTH_M
    (vis_cap,) = fc.visible_gt(gt, k[None], ext[None], deep, max_depth=2.5)
    assert vis_cap.tolist() == [0, 4]
    assert 5.0 > fc.GT_MAX_DEPTH_M


# ------------------------------------------------------------------ evaluate_window


def test_evaluate_window_exact_assoc_duplicate_and_miss() -> None:
    k = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    ext = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    recon = SimpleNamespace(
        n_views=2,
        intrinsics=np.stack([k, k]),
        extrinsics=np.stack([ext, ext]),
        depth=np.full((2, 100, 100), 1.0),
    )
    gt_ids = ["A", "B", "C", "D"]
    gt_xyz = np.array([[0, 0, 1.0], [0.2, 0, 1.0], [-0.2, 0, 1.0], [0, 0.3, 1.0]])
    px = {"A": (50, 50), "B": (70, 50), "C": (30, 50), "far": (60, 90)}

    def rec(i: str, view: int, name: str, xyz: tuple[float, float, float]) -> DetectionRecord:
        return DetectionRecord(
            id=i, frame_id=view, view=view, xyz=list(xyz), uv=[float(v) for v in px[name]]
        )

    inside = (0.0, 0.0, 1.0)
    dets = [
        rec("d0", 0, "A", inside),
        rec("d1", 0, "B", inside),
        rec("d5", 0, "C", inside),
        rec("d2", 1, "A", inside),
        rec("d3", 1, "B", inside),
        rec("d4", 1, "C", inside),
        rec("d6", 0, "far", (50.0, 50.0, 50.0)),  # outside the ROI
    ]

    def inst(i: int, members: list[str], xyz: tuple[float, float, float]) -> Instance:
        return Instance(i, np.array(xyz), 0.1, members, len(members), 0.9)

    instances = [
        inst(0, ["d0", "d2"], (0.0, 0.0, 1.0)),  # A
        inst(1, ["d1", "d3"], (0.2, 0.0, 1.0)),  # B
        inst(2, ["d5"], (-0.2, 0.0, 1.0)),  # C
        inst(3, ["d4"], (-0.2, 0.0, 1.0)),  # C again: a duplicate
        inst(4, ["d6"], (50.0, 50.0, 50.0)),  # outside ROI: dropped
    ]
    result = SimpleNamespace(detections=dets, instances=instances, count=7)
    box = (np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 3.0]))
    ev = fc.evaluate_window(result, recon, Sim3(1.0, np.eye(3), np.zeros(3)), gt_ids, gt_xyz, box)

    assert ev["pred"] == 4
    assert ev["gt"] == 4  # A, B, C, D all visible in view 0 or 1
    assert ev["raw_count"] == 7
    assert ev["n_labeled_dets"] == 6
    # pairs: predicted positive 2 (clusters of size 2), truly positive 3, hit 2 -> P=1, R=2/3
    assert ev["assoc_f1"] == pytest.approx(0.8)
    assert ev["duplicate_rate"] == pytest.approx(0.25)  # 4 clusters, GT A,B,C,C
    assert ev["miss_rate"] == pytest.approx(0.25)  # D never covered
    assert ev["per_image"] == [3, 3]
    assert ev["baselines"] == {"sum": 6, "max": 3, "median": 3}


def test_evaluate_window_nan_when_too_few_labels() -> None:
    k = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    ext = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    recon = SimpleNamespace(
        n_views=1, intrinsics=k[None], extrinsics=ext[None], depth=np.full((1, 100, 100), 1.0)
    )
    result = SimpleNamespace(detections=[], instances=[], count=0)
    box = (np.full(3, -1.0), np.full(3, 3.0))
    ev = fc.evaluate_window(
        result, recon, Sim3(1.0, np.eye(3), np.zeros(3)), ["A"], np.array([[0, 0, 1.0]]), box
    )
    assert ev["pred"] == 0
    assert np.isnan(ev["assoc_f1"])
    assert ev["miss_rate"] == pytest.approx(1.0)
    assert ev["per_image"] == [0]


def test_make_job_and_run_output() -> None:
    job = fc.make_job("Tree01", "test", "east", 5, "countonce_v1", {"a": 1})
    assert (job.scene_id, job.sample, job.n_views, job.split) == ("Tree01", "east", 5, "test")
    ev = {
        "pred": 7,
        "gt": 9,
        "assoc_f1": 0.5,
        "duplicate_rate": 0.1,
        "miss_rate": 0.2,
        "baselines": {"sum": 12.0},
    }
    res = SimpleNamespace(peak_vram_mb=100.0, timings={"total": 1.5})
    pipe = fc.run_output(ev, "countonce_v1", res, 2.0)
    assert (pipe.pred, pipe.count_error, pipe.assoc_f1) == (7.0, 2.0, 0.5)
    base = fc.run_output(ev, "sum", res, 2.0)
    assert (base.pred, base.count_error, base.assoc_f1) == (12.0, 3.0, None)


# ------------------------------------------------------------------ script


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location(
        "run_fuji_core", REPO / "scripts" / "run_fuji_core.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_help_lists_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    with pytest.raises(SystemExit) as e:
        script.main(["--help"])  # type: ignore[attr-defined]
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "overview" in out and "bench" in out


def test_script_help_subprocess() -> None:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_fuji_core.py"), "--help"],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "bench" in r.stdout


def test_script_arg_parsing() -> None:
    script = _load_script()
    p = script.build_parser()  # type: ignore[attr-defined]
    o = p.parse_args(["overview"])
    assert (o.cmd, o.sides, o.force) == ("overview", ["east", "west"], False)
    assert o.fn is script.cmd_overview  # type: ignore[attr-defined]
    o2 = p.parse_args(["overview", "--sides", "west", "--force"])
    assert (o2.sides, o2.force) == (["west"], True)
    b = p.parse_args(["bench"])
    assert (b.split, b.ks, b.sides) == ("val", [3, 5, 10, 20], ["east", "west"])
    assert (b.score_threshold, b.budget_min) == (0.3, 90.0)
    assert b.fn is script.cmd_bench  # type: ignore[attr-defined]
    b2 = p.parse_args(
        ["bench", "--split", "test", "--ks", "5", "10", "--score-threshold", "0.25",
         "--budget-min", "12.5", "--csv", "x.csv"]
    )  # fmt: skip
    assert (b2.split, b2.ks, b2.score_threshold, b2.budget_min) == ("test", [5, 10], 0.25, 12.5)
    assert b2.csv == Path("x.csv")
    for bad in (["bench", "--split", "train"], [], ["nope"]):
        with pytest.raises(SystemExit) as e:
            p.parse_args(bad)
        assert e.value.code == 2
