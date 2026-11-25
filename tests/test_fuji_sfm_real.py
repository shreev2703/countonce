"""Fuji-SfM loader against the real download (skips when ``data/fuji_sfm`` is absent).

CC BY-NC-SA 4.0 data: nothing here copies images or GT into the repo; only counts and tree ids
are asserted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.datasets.fuji_sfm import load_fuji_sfm
from bench.datasets.validate import validate_scene
from bench.splits import FUJI_SPLITS_PATH, load_splits, verify_fuji_splits

ROOT = Path(__file__).resolve().parent.parent / "data" / "fuji_sfm"
pytestmark = pytest.mark.skipif(
    not (ROOT / "unpacked" / "Fuji-SfM_dataset" / "2-SfM-set").is_dir(),
    reason="Fuji-SfM data absent (data/fuji_sfm)",
)

TEST_TREES = [f"Tree{i:02d}" for i in range(1, 9)]
TRAIN_TREES = ["Tree09", "Tree10", "Tree11"]


@pytest.fixture(scope="module")
def scene():
    return load_fuji_sfm(ROOT, validate=False)


def test_counts(scene):
    assert len(scene.gt_ids) == len(set(scene.gt_ids)) == 1455
    assert scene.gt_xyz.shape == (1455, 3) and scene.gt_extent.shape == (1455, 3)
    assert len(scene.image_paths) == 582
    assert [len(scene.side_indices[s]) for s in ("east", "west")] == [291, 291]
    assert len(scene.mask_images) == 288 and set(scene.mask_annotations) == {"train", "val"}
    assert not scene.intrinsics and not scene.extrinsics  # no poses ship


def test_trees_and_splits(scene):
    assert scene.trees == {"training": TRAIN_TREES, "test": TEST_TREES}
    n = {s: sum(1 for x in scene.gt_split if x == s) for s in ("training", "test")}
    assert n == {"training": 433, "test": 1022}
    assert int(scene.tree_mask("Tree01").sum()) == 147


def test_gt_is_metric(scene):
    # Apples are ~10 cm boxes (points on a 5 cm grid); coordinates are shifted global metres.
    assert 0.09 < float(scene.gt_extent.mean()) < 0.12
    assert np.isfinite(scene.gt_xyz).all()
    assert scene.gt_shift == (-336300.0, -4623400.0, 0.0)


def test_validate_scene_clean(scene):
    assert validate_scene(scene) == []


def test_frozen_real_split_matches_disk():
    splits = load_splits(FUJI_SPLITS_PATH)
    assert sorted(splits["test"]) == [f"fuji_sfm:{t}" for t in TEST_TREES]
    assert sorted(splits["val"]) == [f"fuji_sfm:{t}" for t in TRAIN_TREES]
    assert verify_fuji_splits(ROOT) == []
