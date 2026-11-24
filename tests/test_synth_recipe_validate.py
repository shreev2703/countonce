import json

import numpy as np
import pytest

from synth import validate as V
from synth.recipe import SceneRecipe, make_recipe


def test_recipe_deterministic_and_roundtrip():
    a, b = make_recipe(7), make_recipe(7)
    assert a == b and a.to_json() == b.to_json()
    assert make_recipe(8) != a
    assert SceneRecipe.from_json(a.to_json()) == a
    json.loads(a.to_json())


@pytest.mark.parametrize("seed", range(25))
def test_recipe_ranges(seed):
    r = make_recipe(seed)
    assert 5 <= r.n_objects <= 150 and 3 <= len(r.rig.cameras) <= 40


def test_recipe_pinned_fields_and_bounds():
    r = make_recipe(1, n_objects=5, n_views=40, arrangement="canopy", rig_kind="arc")
    assert (r.n_objects, len(r.rig.cameras), r.arrangement, r.rig.kind) == (5, 40, "canopy", "arc")
    with pytest.raises(ValueError):
        make_recipe(1, n_objects=151)
    with pytest.raises(ValueError):
        make_recipe(1, n_views=2)


def clean(v=3, h=8, w=8):
    rng = np.random.default_rng(0)
    masks = np.zeros((v, h, w), dtype=int)
    masks[:, 2:4, 2:4] = 1
    meta = {
        "scene_id": "s0",
        "recipe_version": "1.0",
        "seed": 0,
        "renderer": "test",
        "prompt": "x",
        "n_objects": 1,
        "n_views": v,
        "target_ids": [1],
        "object_positions": [(0.0, 0.0, 0.0)],
        "resolution": (h, w),
    }
    ang = np.linspace(0, 1.5, v)
    centers = np.stack([3 * np.cos(ang), 3 * np.sin(ang), np.ones(v)], axis=1)
    return V.RenderedScene(
        metadata=meta,
        images=rng.integers(40, 200, (v, h, w, 3)).astype(np.uint8),
        depth=rng.uniform(1, 5, (v, h, w)),
        instance_masks=masks,
        camera_centers=centers,
    )


def test_clean_scene_passes():
    assert V.validate_scene(clean()) == []


def broken(kind):
    s = clean()
    if kind == V.SCHEMA_ERROR:
        s.metadata["n_objects"] = 0
    elif kind == V.NAN_DEPTH:
        s.depth[0, 0, 0] = np.nan
    elif kind == V.BLACK_FRAME:
        s.images[1] = 0
    elif kind == V.OVEREXPOSED:
        s.images[2] = 255
    elif kind == V.ZERO_TARGET:
        s.instance_masks[:] = 0
    elif kind == V.FEW_VIEWS:
        s.instance_masks[1:] = 0
    elif kind == V.DEGENERATE_RIG:
        s.camera_centers[:] = s.camera_centers[0]
    return s


@pytest.mark.parametrize("kind", V.REASONS)
def test_each_rule_triggers(kind):
    assert kind in V.validate_scene(broken(kind))


def test_report_counts_and_files(tmp_path):
    report = V.validate_scenes([clean(), broken(V.NAN_DEPTH), broken(V.FEW_VIEWS)])
    assert report.n_scenes == 3 and report.n_quarantined == 2
    counts = report.counts_by_reason
    assert counts[V.NAN_DEPTH] == 1 and counts[V.FEW_VIEWS] == 1 and counts[V.BLACK_FRAME] == 0
    jp, mp = report.write(tmp_path)
    assert json.loads(jp.read_text())["n_quarantined"] == 2
    assert V.NAN_DEPTH in mp.read_text()
