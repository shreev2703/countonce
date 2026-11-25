"""The cached confidence sweep gives the same ranges as the original per-cell sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from countonce import pipeline
from countonce.confidence import k_grid, threshold_grid
from countonce.config import Config
from countonce.detect.base import FakeDetector
from tests.test_pipeline import PlaneRecon, make_scene

TINY = Path(__file__).resolve().parents[1] / "examples" / "tiny_scene"


def _reference_sweep(monkeypatch):
    def ref(config, raw, lifted, default_count, recon, frame_map, lift_kw):
        return pipeline.make_count_at_reference(config, raw, lifted, recon, frame_map, lift_kw)

    monkeypatch.setattr(pipeline, "make_count_at", ref)


def _seeded_script(script, seed=3):
    """Jitter boxes and spread scores so thresholds and k both change the counts."""
    rng = np.random.default_rng(seed)
    out = {}
    for fid, boxes in script.items():
        rows = []
        for x0, y0, x1, y1, _ in boxes:
            dx, dy = rng.normal(0, 1.5, 2)
            rows.append((x0 + dx, y0 + dy, x1 + dx, y1 + dy, float(rng.uniform(0.15, 0.95))))
        out[fid] = rows
    return out


def _ranges(res):
    return res.count, res.interval.low, res.interval.high


@pytest.mark.parametrize("assoc", ["v1", "v2"])
def test_seeded_synthetic_scene_ranges_match(tmp_path, monkeypatch, assoc):
    img_dir, script = make_scene(tmp_path)
    script = _seeded_script(script)

    def go(tag):
        cfg = Config(out_dir=tmp_path / tag, max_frames=0, assoc=assoc, score_threshold=0.5)
        return pipeline.run(
            img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon(),
            export=False,
        )  # fmt: skip

    new = go("new")
    _reference_sweep(monkeypatch)
    old = go("old")
    assert _ranges(new) == _ranges(old)
    assert new.interval.high > new.interval.low  # the sweep really varied, so the test has teeth


def test_every_grid_cell_matches(tmp_path):
    img_dir, script = make_scene(tmp_path)
    script = _seeded_script(script, seed=5)
    cfg = Config(out_dir=tmp_path / "o", max_frames=0, score_threshold=0.5)
    seen = {}
    orig = pipeline.make_count_at

    def spy(*a, **kw):
        seen["args"] = a
        return orig(*a, **kw)

    pipeline.make_count_at = spy
    try:
        pipeline.run(
            img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon(),
            export=False,
        )  # fmt: skip
    finally:
        pipeline.make_count_at = orig
    config, raw, lifted, default_count, recon, frame_map, lift_kw = seen["args"]
    fast = pipeline.make_count_at(config, raw, lifted, default_count, recon, frame_map, lift_kw)
    ref = pipeline.make_count_at_reference(config, raw, lifted, recon, frame_map, lift_kw)
    cells = [(k, t) for t in threshold_grid(0.5) for k in k_grid(config.merge_k)]
    assert [fast(k, t) for k, t in cells] == [ref(k, t) for k, t in cells]
    assert len({ref(k, t) for k, t in cells}) > 1


@pytest.mark.parametrize("assoc", ["v1", "v2"])
def test_tiny_scene_ranges_match(tmp_path, monkeypatch, assoc):
    def go(tag):
        cfg = Config(
            out_dir=tmp_path / tag, detector="color_blob", backbone="fixture", assoc=assoc,
            max_frames=0, backbone_options={"scene": str(TINY / "scene.json")},
        )  # fmt: skip
        return pipeline.run(TINY, "red sphere", cfg, export=False)

    new = go("new")
    _reference_sweep(monkeypatch)
    assert _ranges(new) == _ranges(go("old"))
