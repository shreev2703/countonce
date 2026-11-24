"""Mask producers, cache and mask lifting (PLAN 8.2 masks.py, 9.6)."""

import numpy as np
import pytest
from PIL import Image

from countonce.config import Config
from countonce.detect.masks import (
    BoxFillMaskProducer,
    MaskCache,
    MaskModelUnavailable,
    MaskProducer,
    SamMaskProducer,
    attach_masks,
    build_mask_producer,
)
from countonce.lift import lift_detections
from countonce.pipeline import run
from countonce.types import Detection, Frame
from tests.test_lift_cluster import _plane_recon
from tests.test_pipeline import OBJECTS, FakeDetector, PlaneRecon, make_scene


def _frame(tmp_path, h=60, w=80, fid=0):
    p = tmp_path / f"f{fid}.png"
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(p)
    return Frame(id=fid, path=p, width=w, height=h, sha256=f"sha{fid}")


def _det(k, box, fid=0):
    return Detection(id=f"{fid}:{k}", frame_id=fid, score=0.9, box_xyxy=np.array(box, float))


def test_box_fill_aligns_with_boxes():
    img = np.zeros((60, 80, 3), np.uint8)
    boxes = [(10, 5, 30, 25), (50.4, 40, 79, 59.5)]
    ms = BoxFillMaskProducer().masks(img, boxes)
    assert isinstance(BoxFillMaskProducer(), MaskProducer)
    for m, (x0, y0, x1, y1) in zip(ms, boxes, strict=True):
        assert m.shape == (60, 80) and m.dtype == bool
        ys, xs = np.nonzero(m)
        assert xs.min() == int(np.floor(x0)) and xs.max() == min(int(np.ceil(x1)), 80) - 1
        assert ys.min() == int(np.floor(y0)) and ys.max() == min(int(np.ceil(y1)), 60) - 1


def test_attach_masks_and_cache_hits_skip_recompute(tmp_path):
    frame = _frame(tmp_path)
    dets = [_det(0, (10, 10, 30, 30)), _det(1, (40, 20, 60, 50))]

    class Counting(BoxFillMaskProducer):
        calls = 0

        def masks(self, image, boxes):
            Counting.calls += 1
            return super().masks(image, boxes)

    cache = MaskCache(tmp_path / "masks")
    stats: dict[str, int] = {}
    a = attach_masks(dets, [frame], Counting(), "det@1", cache, stats)
    b = attach_masks(dets, [frame], Counting(), "det@1", cache, stats)
    assert Counting.calls == 1 and stats == {"hits": 1, "misses": 1}
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.mask, y.mask)
    assert a[0].mask is not None and a[0].mask[10:30, 10:30].all() and a[0].mask.sum() == 400
    # a different detector name, mask version or box set is a different key
    attach_masks(dets, [frame], Counting(), "other@1", cache)
    assert Counting.calls == 2
    attach_masks(dets[:1], [frame], Counting(), "det@1", cache)
    assert Counting.calls == 3
    assert dets[0].mask is None  # inputs are not mutated


def test_sam_missing_weights_is_a_clear_error(tmp_path):
    with pytest.raises(MaskModelUnavailable, match="SAM weights not found"):
        SamMaskProducer(tmp_path / "nope.pt").masks(np.zeros((8, 8, 3), np.uint8), [(1, 1, 4, 4)])
    assert build_mask_producer("none") is None
    with pytest.raises(ValueError):
        build_mask_producer("bogus")


def test_mask_lift_is_tighter_than_box_centre():
    """Object fills the right third of a loose box: the mask centre beats the box centre."""
    recon = _plane_recon(h=60, w=80, f=100.0, z=5.0)
    box = np.array([10.0, 10.0, 70.0, 50.0])
    obj = np.zeros((60, 80), bool)
    obj[20:40, 52:68] = True
    truth_x = (59.5 + 0.0 - 40) * 5.0 / 100.0  # object centre column 59.5 -> world x
    box_det = Detection(id="0:0", frame_id=0, score=0.9, box_xyxy=box)

    class ObjectMask:
        name, version = "obj", "1"

        def masks(self, image, boxes):
            return [obj.copy()]

    mask_det = attach_masks([box_det], [_frame_stub()], ObjectMask(), "d@1", None)[0]
    (lb,) = lift_detections([box_det], recon)
    (lm,) = lift_detections([mask_det], recon)
    assert abs(lm.xyz[0] - truth_x) < abs(lb.xyz[0] - truth_x)
    assert abs(lm.xyz[0] - truth_x) < 0.05


def _frame_stub():
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    return _frame(d)


def test_pipeline_masks_switch(tmp_path):
    img_dir, script = make_scene(tmp_path)
    seen = {}

    base = Config(out_dir=tmp_path / "o1", cache_dir=tmp_path / "cache", max_frames=0)
    cfg = base.model_copy(update={"masks": "box_fill", "out_dir": tmp_path / "o2"})
    res = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    assert res.count == len(OBJECTS)
    assert any((tmp_path / "cache" / "masks").glob("*.npz"))
    # the second run hits the mask cache
    import countonce.detect.masks as m

    orig = m.BoxFillMaskProducer.masks

    def spy(self, image, boxes):
        seen["n"] = seen.get("n", 0) + 1
        return orig(self, image, boxes)

    m.BoxFillMaskProducer.masks = spy  # type: ignore[method-assign]
    try:
        run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    finally:
        m.BoxFillMaskProducer.masks = orig  # type: ignore[method-assign]
    assert "n" not in seen


def test_sam_mode_without_checkpoint_fails_clearly(tmp_path):
    img_dir, script = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "o", max_frames=0, masks="sam")
    with pytest.raises(MaskModelUnavailable):
        run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
