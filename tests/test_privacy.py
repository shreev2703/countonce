"""Privacy-mask producer: blur, mask shape through the grid map, lift rejection (PLAN 14.8)."""

import numpy as np
import pytest
from PIL import Image

from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.ingest.frames import backbone_grid_map
from countonce.ingest.privacy import (
    OpenCVRegionDetector,
    PrivacyRegionDetector,
    blur_image_file,
    blur_regions,
    build_privacy_detector,
    mask_to_grid,
)
from countonce.lift import lift_detections
from countonce.pipeline import make_frames, run
from tests.test_lift_cluster import _plane_recon
from tests.test_pipeline import PlaneRecon, make_scene


class FakeRegions:
    def __init__(self, boxes):
        self.boxes = boxes

    def regions(self, image):
        return list(self.boxes)


def _noise(h=60, w=80, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_fake_satisfies_protocol():
    assert isinstance(FakeRegions([]), PrivacyRegionDetector)


def test_blur_changes_only_masked_pixels():
    img = _noise()
    out, mask = blur_regions(img, FakeRegions([(10, 10, 40, 30)]), sigma=4.0)
    assert mask.shape == img.shape[:2] and mask.dtype == bool
    assert mask[10:30, 10:40].all() and mask.sum() == 20 * 30
    np.testing.assert_array_equal(out[~mask], img[~mask])
    # blurring noise must shrink its variance inside the region
    assert out[mask].astype(float).std() < 0.5 * img[mask].astype(float).std()


def test_no_regions_is_identity_and_boxes_are_clipped():
    img = _noise()
    out, mask = blur_regions(img, FakeRegions([]))
    assert not mask.any()
    np.testing.assert_array_equal(out, img)
    _, mask = blur_regions(img, FakeRegions([(-20, -20, 5, 5), (70, 50, 200, 200)]))
    assert mask[:5, :5].all() and mask[50:, 70:].all()


@pytest.mark.parametrize("w,h", [(80, 60), (1000, 400), (300, 900)])
def test_mask_has_grid_shape_after_resize_and_lines_up(w, h):
    mask = np.zeros((h, w), bool)
    mask[: h // 2, : w // 4] = True  # upper-left block
    for mode in ("crop", "pad"):
        gm = backbone_grid_map(w, h, mode)
        g = mask_to_grid(mask, gm)
        assert g.shape == (gm.grid_height, gm.grid_width)
        # a grid cell is masked iff the original pixel it maps to is masked
        ys, xs = np.nonzero(g)
        assert len(ys) > 0
        for v, u in list(zip(ys, xs, strict=True))[:: max(1, len(ys) // 50)]:
            x, y = gm.to_original(u + 0.5, v + 0.5)
            assert mask[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))]
    # pad mode leaves the padding unmasked
    gm = backbone_grid_map(1000, 400, "pad")
    assert not mask_to_grid(np.ones((400, 1000), bool), gm)[0].any()


def test_blur_image_file_roundtrip(tmp_path):
    src = tmp_path / "a.jpg"
    Image.fromarray(_noise()).save(src, quality=100)
    dest, mask = blur_image_file(src, tmp_path / "out", FakeRegions([(0, 0, 40, 30)]), 4.0)
    assert dest.suffix == ".png" and dest.exists()
    assert mask.shape == (60, 80) and mask[:30, :40].all()
    assert np.asarray(Image.open(dest)).shape == (60, 80, 3)


def test_detection_mostly_inside_mask_fails_with_privacy_mask(tmp_path):
    img_path = tmp_path / "f.png"
    Image.fromarray(_noise()).save(img_path)
    _, mask = blur_regions(_noise(), FakeRegions([(0, 0, 40, 60)]))
    (frame,) = make_frames([img_path], masks={img_path: mask})
    assert frame.privacy_mask is not None and frame.privacy_mask.shape == (60, 80)
    from countonce.types import Detection

    inside = Detection(id="0:0", frame_id=0, score=0.9, box_xyxy=np.array([4.0, 4.0, 30.0, 30.0]))
    outside = Detection(id="0:1", frame_id=0, score=0.9, box_xyxy=np.array([45.0, 4.0, 75.0, 40.0]))
    a, b = lift_detections([inside, outside], _plane_recon(), frames={0: frame})
    assert a.fail_reason == "privacy_mask"
    assert b.ok


def test_opencv_detector_is_lazy_and_never_downloads(tmp_path):
    pytest.importorskip("cv2")
    det = OpenCVRegionDetector(tmp_path / "missing.xml")  # constructing must not touch the disk
    with pytest.raises(FileNotFoundError):
        det.regions(_noise())
    with pytest.raises(ValueError):
        build_privacy_detector(None)


def test_pipeline_blur_off_by_default_and_on_masks_frames(tmp_path):
    assert Config().privacy_blur is False
    img_dir, script = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0, privacy_blur=True)
    res = run(
        img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon(),
        privacy_detector=FakeRegions([(0, 0, 32, 48)]),
    )  # fmt: skip
    assert len(list((tmp_path / "out" / "privacy").glob("*.png"))) == 4
    assert res.count < 5  # objects in the masked left half cannot be lifted
