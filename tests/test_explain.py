"""'Explain this match' backend (PLAN 14.8): merged, cannot-link, far pairs and a 1.0 upgrade."""

import copy
import json
from pathlib import Path

import pytest

from countonce import schemas
from countonce.cli import main
from countonce.explain import explain_pair

FIX = Path(__file__).parent / "fixtures" / "schemas"


def _raw(version: str) -> dict:
    return json.loads((FIX / f"v{version}" / "result.json").read_text())


@pytest.fixture
def doc() -> dict:
    return _raw("1.1")


def test_merged_pair(doc):
    e = explain_pair(doc, "0:0", "1:0")
    assert e.merged and e.instance_a == e.instance_b
    assert e.cannot_link == "not_applicable" and not e.same_frame
    assert e.dist_3d_norm is not None and e.dist_3d_norm < e.k
    assert e.reproj_a_to_b_px is not None and e.reproj_a_to_b_px < 1.0
    assert e.reproj_b_to_a_px is not None and e.reproj_b_to_a_px < 1.0
    assert e.appearance_cosine is None and "appearance_cosine" in e.missing
    assert e.match_prob == 1.0
    assert e.reason.startswith("merged")


def test_cannot_link_pair(doc):
    e = explain_pair(doc, "0:0", "0:1")
    assert not e.merged and e.same_frame and e.cannot_link == "blocked"
    assert e.reason == "same frame: cannot-link"


def test_cannot_link_disabled_is_reported(doc):
    doc["config"]["config"]["cannot_link"] = False
    e = explain_pair(doc, "0:0", "0:1")
    assert e.cannot_link == "disabled" and not e.reason.startswith("same frame")


def test_far_pair(doc):
    e = explain_pair(doc, "0:0", "1:1")
    assert not e.merged and e.cannot_link == "not_applicable"
    assert e.dist_3d_norm is not None and e.dist_3d_norm > e.k
    assert e.reason == f"3D distance {e.dist_3d_norm:.1f}x object size > k={e.k:g}"
    assert "x object size > k=0.5" in e.reason


def test_appearance_cosine_when_features_stored(doc):
    for rec in doc["detections"]:
        if rec["id"] == "0:0":
            rec["feature"] = [1.0, 0.0]
        if rec["id"] == "1:0":
            rec["feature"] = [1.0, 1.0]
    e = explain_pair(doc, "0:0", "1:0")
    assert e.appearance_cosine == pytest.approx(2**-0.5)
    assert "appearance_cosine" not in e.missing


def test_unknown_and_identical_ids(doc):
    with pytest.raises(ValueError, match="unknown detection"):
        explain_pair(doc, "0:0", "9:9")
    with pytest.raises(ValueError, match="differ"):
        explain_pair(doc, "0:0", "0:0")


def test_v1_0_document_is_upgraded_and_explained():
    raw = _raw("1.0")
    assert raw["schema_version"] == "1.0" and "detections" not in raw
    upgraded = schemas.load_result(copy.deepcopy(raw))
    assert upgraded.schema_version == "1.1" and upgraded.detections == []

    e = explain_pair(raw, "0:0", "1:0")  # merged, but no geometry stored in 1.0
    assert e.merged and e.dist_3d_norm == 0.0  # falls back to the logged match evidence
    assert e.reproj_a_to_b_px is None and "reprojection_error" in e.missing
    assert explain_pair(raw, "0:0", "0:1").reason == "same frame: cannot-link"
    far = explain_pair(raw, "0:0", "1:1")
    assert not far.merged and "3d_distance" in far.missing
    assert far.reason.startswith("kept separate")


def test_markers_1_0_upgrade():
    m = schemas.load_markers(json.loads((FIX / "v1.0" / "markers.json").read_text()))
    assert m.schema_version == "1.1"


def test_cli_text_and_json(tmp_path, capsys):
    (tmp_path / "result.json").write_text((FIX / "v1.1" / "result.json").read_text())
    assert main(["explain", str(tmp_path), "0:0", "0:1"]) == 0
    assert "same frame: cannot-link" in capsys.readouterr().out
    assert main(["explain", str(tmp_path), "0:0", "1:0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["merged"] is True
    assert main(["explain", str(tmp_path), "0:0", "nope"]) == 2
    assert main(["explain", str(tmp_path / "missing"), "0:0", "1:0"]) == 2
