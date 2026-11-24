"""Pre-flight check on generated sharp, blurred and duplicated images."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from countonce.cli import main
from countonce.config import Config
from countonce.ingest.preflight import PreflightThresholds, load_thresholds, preflight


def _sharp(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    return Image.fromarray(base).resize((128, 128), Image.Resampling.NEAREST).convert("RGB")


def _save(tmp: Path, name: str, im: Image.Image) -> Path:
    p = tmp / name
    im.save(p)
    return p


@pytest.fixture
def photos(tmp_path):
    a = _save(tmp_path, "a.png", _sharp(1))
    b = _save(tmp_path, "b.png", _sharp(2))
    c = _save(tmp_path, "c.png", _sharp(3))
    blur = _save(tmp_path, "d.png", _sharp(4).filter(ImageFilter.GaussianBlur(6)))
    dup = _save(tmp_path, "e.png", Image.open(b))
    return [a, b, c, blur, dup]


def test_flags_blur_and_duplicates(photos):
    th = PreflightThresholds(max_neighbor_distance=1.0)
    r = preflight(photos, th)
    assert r.blurry == [3]
    assert r.duplicate_groups == [[1, 4]]
    assert "Photo 4 is blurry, retake it." in r.tips
    assert any("Photos 2 and 5 are nearly identical" in t for t in r.tips)
    assert not r.ok and r.thresholds_provisional


def test_clean_set_ok(tmp_path):
    ps = [_save(tmp_path, f"{i}.png", _sharp(i)) for i in range(4)]
    r = preflight(ps, PreflightThresholds(max_neighbor_distance=1.0))
    assert r.ok and r.tips == [] and r.diversity > 0.1


def test_three_identical_listed(tmp_path):
    im = _sharp(9)
    ps = [_save(tmp_path, f"{i}.png", im) for i in range(3)] + [_save(tmp_path, "x.png", _sharp(5))]
    r = preflight(ps, PreflightThresholds(max_neighbor_distance=1.0))
    assert r.duplicate_groups == [[0, 1, 2]]
    assert any("Photos 1, 2 and 3 are nearly identical" in t for t in r.tips)


def test_too_few_and_low_overlap(tmp_path):
    one = preflight([_save(tmp_path, "a.png", _sharp(1))])
    assert one.too_few_views and any("Only 1 photo" in t for t in one.tips)
    ps = [_save(tmp_path, f"{i}.png", _sharp(i)) for i in range(4)]
    r = preflight(ps, PreflightThresholds(max_neighbor_distance=0.0))
    assert r.low_overlap and len(r.isolated) == 4


def test_all_alike_low_diversity(tmp_path):
    base = np.full((64, 64, 3), 100, np.uint8)
    ps = []
    for i in range(4):
        arr = base.copy()
        arr[:, i * 8] = 160  # distinct sharp line, tiny thumbnail change
        ps.append(_save(tmp_path, f"{i}.png", Image.fromarray(arr)))
    r = preflight(ps, PreflightThresholds(dup_distance=0.0, max_neighbor_distance=1.0))
    assert r.diversity < 0.03 and any("look alike" in t for t in r.tips)


def test_overrides_and_tuned_file(tmp_path):
    with pytest.raises(ValueError):
        PreflightThresholds.from_overrides({"nope": 1})
    f = tmp_path / "w.yaml"
    f.write_text("thresholds:\n  min_frames: 5\n")
    assert load_thresholds(None, f).min_photos == 5
    assert load_thresholds({"min_photos": 7}, f).min_photos == 7
    assert load_thresholds(None, tmp_path / "missing.yaml").min_photos == 3
    assert Config(preflight={"min_photos": 4}).preflight == {"min_photos": 4}
    with pytest.raises(ValueError):
        Config(preflight={"bad": 1})


def test_cli(photos, capsys):
    rc = main(["preflight", *map(str, photos), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["blurry"] == [3] and out["thresholds_provisional"]
    assert main(["preflight", str(photos[0].parent / "nope.png")]) == 2
