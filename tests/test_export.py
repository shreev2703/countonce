"""Export: schema round-trip, versioning, CSV totals, overlays."""

import csv
import json

import numpy as np
import pytest
from pydantic import ValidationError

from countonce import schemas
from countonce.export import export_all, write_counts_csv, write_markers_json, write_result_json
from countonce.types import (
    CountResult,
    Detection,
    Frame,
    Instance,
    Interval,
    MatchEvidence,
    ModelVersions,
)


def make_result() -> CountResult:
    insts = [
        Instance(0, np.array([0.0, 0.1, 1.0]), 0.2, ["0:0", "1:0"], 2, 0.8),
        Instance(1, np.array([1.0, 0.0, 2.0]), 0.3, ["0:1"], 1, 0.5, possibly_moving=True),
    ]
    return CountResult(
        prompt="apples",
        count=2,
        interval=Interval(1, 3, 0.9, "sensitivity"),
        model_versions=ModelVersions("stub", "vggt", "v1"),
        input_hash="abc",
        instances=insts,
        per_frame_counts={0: 2, 1: 1},
        detection_status={"0:0": "new", "0:1": "new", "1:0": "duplicate", "1:1": "rejected"},
        reject_reasons={"1:1": "low_conf"},
        match_evidence=[MatchEvidence("0:0", "1:0", 0.1, 0.0, 0.0, 0.8, True, "merged")],
        warnings=[{"code": "W1", "message": "m", "fix": "f"}],
        timings={"detect": 1.5},
    )


def test_result_json_roundtrip(tmp_path):
    p = write_result_json(make_result(), tmp_path / "result.json")
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == "1.0"
    r = schemas.load_result(raw)
    assert r.count == 2 and r.interval.high == 3
    assert r.per_frame_counts == {"0": 2, "1": 1}
    assert r.instances[1].possibly_moving


def test_markers_json(tmp_path):
    p = write_markers_json(make_result(), tmp_path / "markers.json", scene_scale=1.5)
    m = schemas.load_markers(json.loads(p.read_text()))
    assert m.schema_version == "1.0" and len(m.markers) == 2
    assert m.markers[0].xyz == (0.0, 0.1, 1.0) and m.scene_scale == 1.5


def test_counts_csv_new_sums_to_count(tmp_path):
    p = write_counts_csv(make_result(), tmp_path / "counts.csv")
    rows = list(csv.DictReader(p.open()))
    assert rows[-1]["frame_id"] == "total"
    assert int(rows[-1]["new"]) == 2
    assert int(rows[-1]["duplicate"]) == 1 and int(rows[-1]["rejected"]) == 1
    assert [r["frame_id"] for r in rows[:-1]] == ["0", "1"]


def test_strict_schema_rejects_unknown_fields():
    raw = write_dict()
    raw["surprise"] = 1
    with pytest.raises(ValidationError):
        schemas.load_result(raw)


def write_dict():
    from countonce.export import to_result

    return to_result(make_result()).model_dump(mode="json")


def test_unknown_version_rejected():
    raw = write_dict()
    raw["schema_version"] = "0.1"
    with pytest.raises(ValueError, match="unsupported"):
        schemas.load_result(raw)
    del raw["schema_version"]
    with pytest.raises(ValueError, match="missing"):
        schemas.load_result(raw)


def test_older_version_upgrades(monkeypatch):
    """An old-version document loads through a registered upgrader."""
    raw = write_dict()
    raw["schema_version"] = "0.9"
    raw["n"] = raw.pop("count")  # old field name

    def up(d):
        d["count"] = d.pop("n")
        d["schema_version"] = "1.0"
        return d

    monkeypatch.setitem(schemas.UPGRADERS, "0.9", up)
    assert schemas.load_result(raw).count == 2


def test_overlays_and_export_all(tmp_path):
    from PIL import Image

    paths = []
    for i in range(2):
        p = tmp_path / f"im{i}.png"
        Image.new("RGB", (64, 48), (200, 200, 200)).save(p)
        paths.append(p)
    frames = [Frame(i, paths[i], 64, 48, "h") for i in range(2)]
    dets = [
        Detection("0:0", 0, 0.9, box_xyxy=np.array([5, 5, 20, 20.0])),
        Detection("0:1", 0, 0.9, point_xy=np.array([40.0, 30.0])),
        Detection("1:0", 1, 0.9, box_xyxy=np.array([6, 6, 22, 22.0])),
        Detection("1:1", 1, 0.2, mask=np.pad(np.ones((4, 4), bool), ((30, 14), (30, 30)))),
    ]
    out = export_all(make_result(), tmp_path / "out", frames, dets, scene_scale=1.0)
    assert (tmp_path / "out" / "result.json").exists()
    assert len(out["overlays"]) == 2
    im = Image.open(out["overlays"][0]).convert("RGB")
    assert im.getpixel((5, 10)) == (46, 160, 67)  # green box edge for a "new" detection
    im1 = Image.open(out["overlays"][1]).convert("RGB")
    assert im1.getpixel((6, 10)) == (47, 111, 224)  # blue for duplicate
