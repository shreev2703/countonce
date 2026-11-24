"""Sensitivity range and quality warnings (PLAN 8.8 Core, 8.9; tests per section 18)."""

import numpy as np
import pytest

from countonce.confidence import k_grid, sensitivity_range, threshold_grid
from countonce.merge.cluster import cluster_detections
from countonce.types import Detection, Instance, LiftedDetection, Recon
from countonce.warnings import (
    LOW_DEPTH_CONFIDENCE,
    LOW_OVERLAP,
    MANY_FAILED_LIFTS,
    TOO_FEW_FRAMES,
    WIDE_RANGE,
    assess_quality,
)


def scene(noise: float, n_obj=15, n_views=4, seed=0):
    """Objects 4 units apart (size 1); each view sees each object with xyz noise, random scores."""
    rng = np.random.default_rng(seed)
    centers = np.stack([np.arange(n_obj) * 4.0, np.zeros(n_obj), np.zeros(n_obj)], axis=1)
    out = []
    for v in range(n_views):
        for i, c in enumerate(centers):
            det = Detection(
                id=f"{v}:{i}", frame_id=v, score=float(rng.uniform(0.2, 0.9)), point_xy=np.zeros(2)
            )
            out.append(
                LiftedDetection(det, xyz=c + rng.normal(0, noise, 3), size=1.0, lift_conf=1.0)
            )
    return out


def range_for(noise: float, seed=0):
    lifted = scene(noise, seed=seed)

    def count_fn(k, thr):
        kept = [ld for ld in lifted if ld.det.score >= thr]
        return len(cluster_detections(kept, k=k).instances)

    return sensitivity_range(count_fn, k_grid(0.5), threshold_grid(0.3), 0.5, 0.3)


def test_range_contains_count_and_is_labelled_sensitivity():
    r = range_for(0.05)
    assert r.low <= r.count <= r.high
    iv = r.to_interval()
    assert iv.method == "sensitivity" and (iv.low, iv.high) == (r.low, r.high)


def test_range_widens_with_noise():
    widths = [np.mean([range_for(s, seed).width for seed in range(3)]) for s in (0.02, 0.1, 0.2)]
    assert widths[0] < widths[1] <= widths[2]
    assert widths[2] > widths[0]


def test_grid_always_includes_defaults():
    assert 0.3 in threshold_grid(0.3)
    calls = []
    sensitivity_range(lambda k, t: calls.append((k, t)) or 3, [1.0], [0.5], 2.0, 0.1)
    assert (2.0, 0.1) in calls


def inst(n_views):
    return Instance(id=0, xyz=np.zeros(3), size=1.0, members=[], n_views=n_views, conf=1.0)


def codes(**kw):
    base = dict(n_frames=8, count=10)
    return {w["code"] for w in assess_quality(**{**base, **kw})}


def test_no_warnings_on_healthy_input():
    assert codes(instances=[inst(3)] * 10, interval_low=9, interval_high=11) == set()


def test_each_warning_triggers_with_message_and_fix():
    recon = Recon(
        backbone="fake",
        intrinsics=np.zeros((1, 3, 3)),
        extrinsics=np.zeros((1, 3, 4)),
        depth=np.ones((1, 2, 2)),
        depth_conf=np.full((1, 2, 2), 1.0),
        world_points=np.zeros((1, 2, 2, 3)),
        world_points_conf=np.ones((1, 2, 2)),
    )
    ok = LiftedDetection(
        Detection(id="0:0", frame_id=0, score=1.0, point_xy=np.zeros(2)), xyz=np.zeros(3), size=1.0
    )
    bad = LiftedDetection(
        Detection(id="0:1", frame_id=0, score=1.0, point_xy=np.zeros(2)), fail_reason="low_conf"
    )
    cases = {
        TOO_FEW_FRAMES: dict(n_frames=2),
        LOW_OVERLAP: dict(instances=[inst(1)] * 10),
        LOW_DEPTH_CONFIDENCE: dict(recon=recon),
        MANY_FAILED_LIFTS: dict(lifted=[ok, bad, bad, bad]),
        WIDE_RANGE: dict(interval_low=6, interval_high=14),
    }
    for code, kw in cases.items():
        found = assess_quality(**{**dict(n_frames=8, count=10), **kw})
        hit = [w for w in found if w["code"] == code]
        assert hit, code
        assert len(hit[0]["message"]) > 20 and len(hit[0]["fix"]) > 20


def test_narrow_range_is_not_flagged():
    assert WIDE_RANGE not in codes(count=2, interval_low=1, interval_high=2)


@pytest.mark.parametrize("n_frames", [3, 8])
def test_enough_frames_not_flagged(n_frames):
    assert TOO_FEW_FRAMES not in codes(n_frames=n_frames)
