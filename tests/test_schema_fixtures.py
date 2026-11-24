"""Schema compatibility: every stored document version loads and renders (PLAN 8.10, 18).

``tests/fixtures/schemas/v<version>/`` holds a ``result.json`` and ``markers.json`` produced by
the CPU e2e pipeline. They are append-only: a schema bump adds a new directory (and an
upgrader in ``countonce.schemas.UPGRADERS``); old directories keep loading forever.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from countonce import schemas
from countonce.export import render_overlays
from countonce.types import (
    CountResult,
    Detection,
    Frame,
    Instance,
    Interval,
    ModelVersions,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "schemas"
VERSION_DIRS = sorted(p for p in FIXTURE_ROOT.glob("v*") if p.is_dir())


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_current_version_has_a_fixture():
    assert FIXTURE_ROOT / f"v{schemas.CURRENT_VERSION}" in VERSION_DIRS


@pytest.mark.parametrize("vdir", VERSION_DIRS, ids=lambda p: p.name)
def test_fixture_is_machine_independent(vdir):
    for name in ("result.json", "markers.json"):
        text = (vdir / name).read_text()
        assert "/home/" not in text and "/tmp/" not in text and "\\Users\\" not in text


@pytest.mark.parametrize("vdir", VERSION_DIRS, ids=lambda p: p.name)
def test_fixture_loads_and_markers_validate(vdir):
    raw_result = _load(vdir / "result.json")
    raw_markers = _load(vdir / "markers.json")
    assert raw_result["schema_version"] == vdir.name[1:] == raw_markers["schema_version"]
    result = schemas.load_result(raw_result)
    markers = schemas.load_markers(raw_markers)
    assert markers.count == len(markers.markers) == result.count
    assert [m.id for m in markers.markers] == [i.id for i in result.instances]
    assert markers.interval_low <= markers.count <= markers.interval_high


@pytest.mark.parametrize("vdir", VERSION_DIRS, ids=lambda p: p.name)
def test_fixture_renders_overlays(vdir, tmp_path):
    """Draw overlays from the stored result onto tiny generated frames."""
    stored = schemas.load_result(_load(vdir / "result.json"))
    frame_ids = sorted({int(d.split(":")[0]) for d in stored.detection_status})
    frames = []
    for fid in frame_ids:
        p = tmp_path / f"f{fid}.png"
        Image.fromarray(np.full((48, 64, 3), 40 + fid, dtype=np.uint8)).save(p)
        frames.append(Frame(fid, p, 64, 48, f"sha{fid}"))
    # Result documents store no pixel boxes; give every stored detection id a box on a grid.
    detections = []
    for det_id in stored.detection_status:
        fid, k = (int(x) for x in det_id.split(":"))
        x0, y0 = 4 + 10 * k, 4 + 8 * (fid % 4)
        detections.append(Detection(det_id, fid, 0.9, np.array([x0, y0, x0 + 6, y0 + 6.0])))
    result = CountResult(
        prompt=stored.prompt,
        count=stored.count,
        interval=Interval(
            stored.interval.low, stored.interval.high, stored.interval.level, stored.interval.method
        ),
        model_versions=ModelVersions(
            stored.model_versions.detector,
            stored.model_versions.backbone,
            stored.model_versions.associator,
        ),
        input_hash=stored.input_hash,
        confidence_method=stored.confidence_method,
        instances=[
            Instance(i.id, np.array(i.xyz), i.size, i.members, i.n_views, i.conf)
            for i in stored.instances
        ],
        detection_status=stored.detection_status,
    )
    out = render_overlays(frames, detections, result, tmp_path / "overlays")
    assert len(out) == len(frames)
    for path, fr in zip(out, frames, strict=True):
        img = np.asarray(Image.open(path).convert("RGB"))
        blank = np.full_like(img, 40 + fr.id)
        assert (img != blank).any(), "overlay drew nothing"
