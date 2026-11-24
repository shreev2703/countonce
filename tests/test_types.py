import json
from pathlib import Path

import numpy as np
import pytest

from countonce.types import (
    SCHEMA_VERSION,
    CountResult,
    Detection,
    Frame,
    Instance,
    Interval,
    LiftedDetection,
    ModelVersions,
    Recon,
)


def _recon(n=2, h=4, w=5):
    return Recon(
        backbone="vggt",
        intrinsics=np.tile(np.eye(3), (n, 1, 1)),
        extrinsics=np.zeros((n, 3, 4)),
        depth=np.zeros((n, h, w)),
        depth_conf=np.zeros((n, h, w)),
        world_points=np.zeros((n, h, w, 3)),
        world_points_conf=np.zeros((n, h, w)),
    )


def _result(**kw):
    base = dict(
        prompt="apples",
        count=1,
        interval=Interval(1, 2, 0.9, "sensitivity"),
        model_versions=ModelVersions("det@abc", "vggt", "v1"),
        input_hash="h",
    )
    base.update(kw)
    return CountResult(**base)


def test_frame_validation():
    f = Frame(0, "a.jpg", 5, 4, "x", privacy_mask=np.zeros((4, 5), bool))
    assert f.path == Path("a.jpg")
    with pytest.raises(ValueError):
        Frame(0, "a.jpg", 5, 4, "x", privacy_mask=np.zeros((5, 5), bool))


def test_detection_validation():
    d = Detection(Detection.make_id(3, 1), 3, 0.9, box_xyxy=np.zeros(4))
    assert d.id == "3:1"
    with pytest.raises(ValueError):
        Detection("0:0", 0, 0.5)
    with pytest.raises(ValueError):
        Detection("0:0", 0, 0.5, box_xyxy=np.zeros(3))


def test_recon_shapes():
    assert _recon().n_views == 2
    r = _recon()
    r.depth_conf = np.zeros((2, 1, 1))
    with pytest.raises(ValueError):
        r.__post_init__()


def test_lifted_detection():
    det = Detection("0:0", 0, 0.9, point_xy=np.zeros(2))
    ok = LiftedDetection(det, xyz=np.zeros(3), size=0.1, lift_conf=0.8)
    assert ok.ok and not ok.possibly_moving
    bad = LiftedDetection(det, fail_reason="low_conf")
    assert not bad.ok
    with pytest.raises(ValueError):
        LiftedDetection(det, fail_reason="nope")


def test_instance_and_interval_validation():
    with pytest.raises(ValueError):
        Instance(0, np.zeros(3), 0.1, [], 1, conf=1.5)
    with pytest.raises(ValueError):
        Interval(3, 1, 0.9, "sensitivity")
    with pytest.raises(ValueError):
        Interval(1, 3, 0.9, "bogus")


def test_count_result_defaults_and_fields():
    r = _result()
    assert r.schema_version == SCHEMA_VERSION == "1.0"
    assert r.confidence_method == "heuristic"
    assert r.recon_cache_key is None
    assert r.n_possibly_moving == 0
    # mutable defaults are not shared
    assert _result().warnings is not r.warnings


def test_count_result_validation():
    with pytest.raises(ValueError):
        _result(count=-1)
    with pytest.raises(ValueError):
        _result(confidence_method="magic")
    with pytest.raises(ValueError):
        _result(detection_status={"0:0": "maybe"})


def test_to_dict_is_json_serializable():
    inst = Instance(0, np.array([1.0, 2.0, 3.0]), 0.1, ["0:0"], 2, 0.7, possibly_moving=True)
    r = _result(
        instances=[inst],
        per_frame_counts={0: 1},
        confidence_method="temperature_scaled",
        recon_cache_key="k123",
    )
    d = r.to_dict()
    assert r.n_possibly_moving == 1
    assert d["instances"][0]["xyz"] == [1.0, 2.0, 3.0]
    assert d["instances"][0]["possibly_moving"] is True
    assert d["recon_cache_key"] == "k123"
    json.dumps(d)
