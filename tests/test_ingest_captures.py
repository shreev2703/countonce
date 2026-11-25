"""scripts/ingest_captures.py: phone folder -> valid scene, EXIF stripped, hand-count rules."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ingest_captures", ROOT / "scripts/ingest_captures.py"
)
ic = importlib.util.module_from_spec(spec)
sys.modules["ingest_captures"] = ic
spec.loader.exec_module(ic)


def make_folder(d: Path, n: int = 3, exif: bool = True) -> Path:
    d.mkdir(parents=True)
    for i in range(n):
        im = Image.new("RGB", (160, 120), (i * 40, 90, 200))
        kw = {}
        if exif:
            e = Image.Exif()
            e[0x010F] = "PhoneCo"
            e[0x0112] = 6  # rotate: stored landscape, displays portrait
            e[0x8825] = {1: "N", 2: (51.0, 30.0, 0.0)}  # GPS IFD
            kw["exif"] = e
        im.save(d / f"IMG_20261101_{i}.jpg", **kw)
    return d


def test_ingest_makes_valid_scene(tmp_path):
    src = make_folder(tmp_path / "phone")
    with Image.open(src / "IMG_20261101_0.jpg") as im0:
        assert im0.getexif()  # fixture really has EXIF
    rc = ic.main(
        [
            "--src",
            str(src),
            "--scene-id",
            "bikes_01",
            "--prompt",
            "bicycle",
            "--counts",
            "4",
            "4",
            "--tier",
            "easy",
            "--out",
            str(tmp_path / "scenes"),
        ]
    )
    assert rc == 0
    scene = tmp_path / "scenes" / "bikes_01"
    gt = json.loads((scene / "gt.json").read_text())
    assert gt["count"] == 4 and gt["n_images"] == 3 and gt["exif_stripped"]
    assert gt["prompt"] == "bicycle" and "visible in at least one" in gt["count_definition"]
    names = sorted(p.name for p in (scene / "images").iterdir())
    assert names == ["img_000.jpg", "img_001.jpg", "img_002.jpg"]
    for p in (scene / "images").iterdir():
        with Image.open(p) as im:
            assert not im.getexif() and im.size == (120, 160)  # orientation applied
    assert ic.validate_scene(scene) == []
    assert "IMG_2026" not in (scene / "gt.json").read_text()


def test_scene_list_zero_scene(tmp_path):
    src = make_folder(tmp_path / "phone", exif=False)
    lst = tmp_path / "list.yaml"
    lst.write_text(
        yaml.safe_dump(
            {
                "scenes": [
                    {
                        "id": "z1",
                        "prompt": "chair",
                        "tier": "medium",
                        "zero_target": True,
                        "physical_scene": "stools_cafe",
                    }
                ]
            }
        )
    )
    rc = ic.main(
        [
            "--src",
            str(src),
            "--scene-id",
            "z1",
            "--scene-list",
            str(lst),
            "--counts",
            "0",
            "0",
            "--out",
            str(tmp_path / "s"),
        ]
    )
    assert rc == 0
    gt = json.loads((tmp_path / "s/z1/gt.json").read_text())
    assert gt["count"] == 0 and gt["zero_target"] and gt["physical_scene"] == "stools_cafe"


def test_template_has_three_zero_scenes():
    doc = yaml.safe_load((ROOT / "configs/capture_scenes.template.yaml").read_text())
    scenes = doc["scenes"]
    assert sum(s["zero_target"] for s in scenes) >= 3
    assert len({s["id"] for s in scenes}) == len(scenes) >= 20
    assert {s["tier"] for s in scenes} == {"easy", "medium", "hard"}


@pytest.mark.parametrize(
    ("counts", "expected"),
    [([5, 5], 5), ([5, 6, 6], 6), ([5, 6, 5], 5), ([5, 5, 5], 5)],
)
def test_resolve_count(counts, expected):
    assert ic.resolve_count(counts) == expected


@pytest.mark.parametrize("counts", [[5], [5, 6], [1, 2, 3], [5, 5, 6], [-1, -1]])
def test_resolve_count_rejects(counts):
    with pytest.raises(ic.IngestError):
        ic.resolve_count(counts)


def test_rejects_bad_input(tmp_path):
    src = make_folder(tmp_path / "phone", exif=False)
    kw = {"prompt": "x", "tier": "easy", "physical_scene": "p", "out_root": tmp_path / "o"}
    with pytest.raises(ic.IngestError, match="--zero"):
        ic.ingest(src, scene_id="a", counts=[0, 0], zero=False, **kw)
    with pytest.raises(ic.IngestError, match="zero-target"):
        ic.ingest(src, scene_id="a", counts=[2, 2], zero=True, **kw)
    (src / "x.heic").write_bytes(b"x")
    with pytest.raises(ic.IngestError, match="HEIC"):
        ic.ingest(src, scene_id="a", counts=[2, 2], zero=False, **kw)
    (src / "x.heic").unlink()
    (src / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(ic.IngestError, match="not a readable"):
        ic.ingest(src, scene_id="a", counts=[2, 2], zero=False, **kw)
    assert not (tmp_path / "o" / "a").exists()  # nothing written on failure
    (src / "bad.jpg").unlink()
    ic.ingest(src, scene_id="a", counts=[2, 2], zero=False, **kw)
    with pytest.raises(ic.IngestError, match="--force"):
        ic.ingest(src, scene_id="a", counts=[2, 2], zero=False, **kw)
    ic.ingest(src, scene_id="a", counts=[2, 2], zero=False, force=True, **kw)


def test_scene_loads_in_bench_sweep(tmp_path):
    from bench.sweep import discover_scenes  # noqa: PLC0415

    ic.ingest(
        make_folder(tmp_path / "p"),
        tmp_path / "scenes",
        "s1",
        prompt="cup",
        counts=[3, 3],
        tier="hard",
        zero=False,
        physical_scene="s1",
    )
    scenes = discover_scenes(tmp_path / "scenes")
    assert [s["id"] for s in scenes] == ["s1"] and scenes[0]["tier"] == "hard"
