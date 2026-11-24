"""Toy-input tests for zero-count metrics and error slicing (no real data)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bench import metrics, slices
from countonce.types import CountResult, Detection, Instance, Interval, ModelVersions, Recon


def test_zero_count_metrics():
    pred = [0, 3, 0, 1, 7]
    gt = [0, 0, 0, 0, 5]
    low = [0, 0, 0, 1, 4]
    high = [1, 5, 0, 2, 9]
    m = metrics.zero_count_metrics(pred, gt, low, high)
    assert m["n_zero"] == 4
    assert m["false_positive_rate"] == pytest.approx(0.5)
    assert m["mean_false_count"] == pytest.approx(1.0)
    assert m["interval_includes_zero_rate"] == pytest.approx(0.75)
    assert metrics.false_positive_rate(pred, gt) == pytest.approx(0.5)
    assert metrics.mean_false_count(pred, gt) == pytest.approx(1.0)
    assert metrics.interval_includes_zero_rate(low, high, pred, gt) == pytest.approx(0.75)


def test_zero_count_metrics_edge_cases():
    m = metrics.zero_count_metrics([2, 3], [2, 3])
    assert m["n_zero"] == 0 and math.isnan(m["false_positive_rate"])
    with pytest.raises(ValueError):
        metrics.zero_count_metrics([0], [0], low=[0])
    with pytest.raises(ValueError):
        metrics.zero_count_metrics([0], [0], [2], [1])


def _result(n_inst: int) -> CountResult:
    insts = [
        Instance(id=i, xyz=np.zeros(3), size=1.0, members=[], n_views=2, conf=0.5)
        for i in range(n_inst)
    ]
    return CountResult(
        prompt="x",
        count=n_inst,
        interval=Interval(n_inst, n_inst, 0.9, "sensitivity"),
        model_versions=ModelVersions("d", "b", "v1"),
        input_hash="h",
        instances=insts,
    )


def test_scene_metadata_from_toy_inputs():
    n, h, w = 2, 10, 10
    recon = Recon(
        backbone="vggt",
        intrinsics=np.tile(np.eye(3), (n, 1, 1)),
        extrinsics=np.tile(np.eye(3, 4), (n, 1, 1)),
        depth=np.ones((n, h, w)),
        depth_conf=np.full((n, h, w), 0.4),
        world_points=np.zeros((n, h, w, 3)),
        world_points_conf=np.ones((n, h, w)),
    )
    dets = [
        Detection(id=f"0:{k}", frame_id=0, score=s, point_xy=np.zeros(2))
        for k, s in enumerate([0.2, 0.6])
    ]
    imgs = [np.full((4, 4, 3), 255, np.uint8), np.zeros((4, 4, 3), np.uint8)]
    m = slices.scene_metadata(_result(3), gt_count=4, recon=recon, detections=dets, images=imgs)
    assert m.n_views == 2 and m.gt_count == 4
    assert m.mean_depth_conf == pytest.approx(0.4)
    assert m.density == pytest.approx(3 / (200 / 1e6))
    assert m.luminance == pytest.approx(0.5)
    assert m.score_mean == pytest.approx(0.4) and m.score_min == pytest.approx(0.2)
    assert math.isnan(m.blur_score)


def _records(n_good: int, n_bad: int) -> list[slices.SceneRecord]:
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n_good):  # low counts, small error, good F1
        meta = slices.SceneMeta(gt_count=10, n_views=10, mean_depth_conf=0.9)
        recs.append(
            slices.make_record(f"g{i}", meta, 10 + rng.integers(0, 2), [0, 0, 1], [0, 0, 1])
        )
    for i in range(n_bad):  # high counts, big error, bad F1
        meta = slices.SceneMeta(gt_count=80, n_views=10, mean_depth_conf=0.9)
        recs.append(
            slices.make_record(f"b{i}", meta, 80 + 20 + rng.integers(0, 3), [0, 0, 0], [0, 1, 2])
        )
    return recs


def test_slice_report_flags_worse_slice_only():
    res = {r.name: r for r in slices.slice_report(_records(20, 20), min_scenes=5)}
    assert res["count>50"].worse_error and res["count>50"].worse_f1
    assert not res["count<20"].worse_error and not res["count<20"].worse_f1
    assert res["count<20"].err_diff < 0 < res["count>50"].err_diff
    assert slices.flagged_slices(list(res.values())) == ["count>50"]
    assert res["count>50"].assoc_f1 == pytest.approx(0.0)


def test_small_slices_not_flagged_and_empty_slices_dropped():
    out = slices.slice_report(_records(20, 3), min_scenes=5)
    names = {r.name: r for r in out}
    assert not names["count>50"].flagged and names["count>50"].n == 3
    assert "dark(lum<0.25)" not in names  # luminance is NaN -> empty slice


def test_slice_report_rejects_empty():
    with pytest.raises(ValueError):
        slices.slice_report([])


def test_unlabelled_scenes_have_nan_f1():
    r = slices.make_record("s", slices.SceneMeta(gt_count=3, n_views=2), 5)
    assert math.isnan(r.assoc_f1) and r.abs_error == 2
