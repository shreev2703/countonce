"""CPU golden synthetic-geometry test (PLAN section 18 CI).

A fixed seeded scene is pushed through the full pipeline with a fake detector and a fake
plane reconstructor, and the v1/v2 counts are pinned. Any change here is a behaviour change.
"""

import numpy as np
import pytest
from PIL import Image

from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.pipeline import run
from tests.test_pipeline import CAM_X, DEPTH, F, H, PlaneRecon, W

SEED = 1234
N_OBJECTS = 6
# Counts pinned from the seeded scene below.
GOLDEN = {"v1": N_OBJECTS, "v2": N_OBJECTS}


def seeded_scene(tmp_path):
    rng = np.random.default_rng(SEED)
    # Spread objects on a jittered grid so they stay well separated in 3D.
    xs = np.linspace(-1.0, 1.4, N_OBJECTS) + rng.uniform(-0.05, 0.05, N_OBJECTS)
    ys = rng.uniform(-0.6, 0.6, N_OBJECTS)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(len(CAM_X)):
        arr = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"f{i}.png")
    script = {}
    for i, cx in enumerate(CAM_X):
        boxes = []
        for ox, oy in zip(xs, ys, strict=True):
            u = F * (ox - cx) / DEPTH + W / 2
            v = F * oy / DEPTH + H / 2
            if 4 <= u < W - 4 and 4 <= v < H - 4:
                boxes.append((u - 3, v - 3, u + 3, v + 3, 0.9))
        script[i] = boxes
    return img_dir, script


@pytest.mark.golden
@pytest.mark.parametrize("assoc", ["v1", "v2"])
def test_golden_counts(tmp_path, assoc):
    img_dir, script = seeded_scene(tmp_path)
    assert sum(len(v) for v in script.values()) > N_OBJECTS  # duplicates exist
    cfg = Config(out_dir=tmp_path / "out", max_frames=0, assoc=assoc)
    res = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    assert res.count == GOLDEN[assoc]
