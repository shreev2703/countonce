import hashlib
import importlib

import numpy as np
import pytest

from countonce.detect._grounding import ensure_caption, logits_boxes_to_detections
from countonce.detect.base import (
    Detector,
    Exemplar,
    FakeDetector,
    cxcywh_norm_to_xyxy,
    default_box_size,
    density_to_points,
    filter_by_score,
    points_to_boxes,
    points_to_detections,
)
from countonce.detect.cache import CachedDetector, DetectionCache, detection_cache_key
from countonce.types import Frame


def mkframe(i: int, w: int = 200, h: int = 100) -> Frame:
    return Frame(
        id=i, path=f"/tmp/{i}.jpg", width=w, height=h, sha256=hashlib.sha256(b"%d" % i).hexdigest()
    )


def test_density_peaks():
    d = np.zeros((10, 12))
    d[2, 3] = 1.0
    d[7, 9] = 0.5
    d[7, 10] = 0.5  # tie: only one survives
    pts, vals = density_to_points(d, min_value=0.1)
    assert len(pts) == 2
    assert {tuple(p) for p in pts} >= {(3.0, 2.0)}
    assert sorted(vals) == [0.5, 1.0]


def test_points_to_boxes_clip_and_size():
    b = points_to_boxes(np.array([[2.0, 50.0]]), (10, 20), 200, 100)
    assert b.tolist() == [[0.0, 40.0, 7.0, 60.0]]
    assert default_box_size(200, 100) == (5.0, 5.0)
    ex = [Exemplar(0, (0, 0, 10, 20)), Exemplar(0, (0, 0, 14, 40))]
    assert default_box_size(200, 100, ex, 0) == (12.0, 30.0)


def test_points_to_detections_have_box_and_point():
    dets = points_to_detections(mkframe(3), np.array([[50.0, 50.0]]), np.array([0.9]))
    assert dets[0].id == "3:0" and dets[0].box_xyxy is not None and dets[0].point_xy is not None


def test_cxcywh_conversion():
    out = cxcywh_norm_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4]]), 200, 100)
    np.testing.assert_allclose(out, [[80, 30, 120, 70]])


def test_logits_postprocess_keeps_low_scores():
    logits = np.array([[0.9, 0.1], [0.06, 0.02], [0.01, 0.0]])
    boxes = np.array([[0.5, 0.5, 0.1, 0.1]] * 3)
    dets = logits_boxes_to_detections(logits, boxes, mkframe(1), floor=0.05)
    assert [round(d.score, 2) for d in dets] == [0.9, 0.06]
    assert len(filter_by_score(dets, 0.5)) == 1
    assert ensure_caption("Apple") == "apple ."
    assert ensure_caption("apple.") == "apple."


def test_cache_hits_and_sweep(tmp_path):
    fake = FakeDetector({0: [(0, 0, 10, 10, 0.9), (5, 5, 20, 20, 0.2)], 1: [(1, 1, 9, 9, 0.5)]})
    assert isinstance(fake, Detector)
    cd = CachedDetector(fake, DetectionCache(tmp_path))
    frames = [mkframe(0), mkframe(1)]
    first = cd.detect(frames, "apple")
    again = cd.detect(frames, "apple")
    assert fake.calls == 1 and cd.hits == 2 and cd.misses == 2
    assert [d.score for d in first] == [d.score for d in again] == [0.9, 0.2, 0.5]
    assert len(filter_by_score(again, 0.4)) == 2
    cd.detect(frames, "pear")  # different prompt -> miss
    assert fake.calls == 2
    cd.detect([mkframe(0), mkframe(2)], "apple")  # only frame 2 is sent on
    assert fake.calls == 3


def test_cache_key_sensitivity():
    f = mkframe(0)
    base = detection_cache_key(f, "a", "d", "1")
    assert base == detection_cache_key(f, "a", "d", "1")
    assert base != detection_cache_key(f, "a", "d", "2")
    assert base != detection_cache_key(f, "a", "e", "1")
    assert base != detection_cache_key(f, "b", "d", "1")
    assert base != detection_cache_key(f, "a", "d", "1", exemplars=[Exemplar(0, (0, 0, 1, 1))])


def test_corrupt_cache_is_miss(tmp_path):
    c = DetectionCache(tmp_path)
    c.path("k").write_text("{not json")
    assert c.get("k", 0) is None


def test_real_adapters_import_lazily_and_gate_on_weights():
    for mod in ("countgd", "groundingdino"):
        importlib.import_module(f"countonce.detect.{mod}")
    from countonce.detect._grounding import ModelUnavailableError
    from countonce.detect.groundingdino import GroundingDinoDetector

    with pytest.raises(ModelUnavailableError):
        GroundingDinoDetector()._load()
