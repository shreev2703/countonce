"""Export: schema round-trip, versioning, CSV totals, overlays."""

import csv
import json

import numpy as np
import pytest
from pydantic import ValidationError

from countonce import schemas
from countonce.export import (
    export_all,
    render_overlays,
    write_counts_csv,
    write_markers_json,
    write_result_json,
)
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


def test_overlay_draws_in_original_coordinates_on_non_square_image(tmp_path):
    from PIL import Image

    from countonce.ingest.frames import resize_to_grid_mapped, vggt_grid_size

    orig = tmp_path / "wide.png"
    Image.new("RGB", (1000, 400), (220, 220, 220)).save(orig)
    (grid_path,), (pm,) = resize_to_grid_mapped([orig], tmp_path / "grid")
    gw, gh = vggt_grid_size(1000, 400)
    assert (gw, gh) == (518, 210) and Image.open(grid_path).size == (gw, gh)
    assert (pm.orig_width, pm.orig_height) == (1000, 400)
    assert pm.sx == pytest.approx(1000 / 518) and pm.sy == pytest.approx(pm.sx) and pm.oy == 0
    frame = Frame(0, grid_path, gw, gh, "h", pixel_map=pm, original_path=orig)
    det = Detection("0:0", 0, 0.9, box_xyxy=np.array([100.0, 50.0, 200.0, 150.0]))
    res = make_result()
    out = render_overlays([frame], [det], res, tmp_path / "ov")
    im = Image.open(out[0]).convert("RGB")
    assert im.size == (1000, 400)  # original resolution, not the grid's
    x0, y0, x1, y1 = (v for v in (*pm.to_original(100, 50), *pm.to_original(200, 150)))
    assert im.getpixel((round(x0) + 1, round((y0 + y1) / 2))) == (46, 160, 67)  # left edge
    assert im.getpixel((round(x1) - 1, round((y0 + y1) / 2))) == (46, 160, 67)  # right edge
    assert im.getpixel((round(x0) + 20, round((y0 + y1) / 2))) == (220, 220, 220)  # interior
    # a box drawn at un-mapped grid coordinates would have missed it entirely
    assert im.getpixel((101, 100)) == (220, 220, 220)


def test_grid_pixel_map_accounts_for_centre_crop_of_tall_frames(tmp_path):
    from PIL import Image

    from countonce.ingest.frames import grid_pixel_map, resize_to_grid_mapped

    orig = tmp_path / "tall.png"
    Image.new("RGB", (300, 900)).save(orig)
    (grid_path,), (pm,) = resize_to_grid_mapped([orig], tmp_path / "grid")
    assert Image.open(grid_path).size == (518, 518)
    assert pm == grid_pixel_map(300, 900)
    cx, cy = pm.to_original(259, 259)  # grid centre is the original centre
    assert (cx, cy) == (pytest.approx(150, abs=1), pytest.approx(450, abs=1))
    assert pm.to_original(0, 0)[1] == pytest.approx(300)  # 300 original px cropped off the top
