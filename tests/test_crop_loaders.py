"""WGISD / MinneApple / AppleGrowthVision loaders on synthetic fixtures in the assumed layouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bench.datasets import validate as V
from bench.datasets.applegrowthvision import load_applegrowthvision, load_applegrowthvision_images
from bench.datasets.common import to_gt_doc
from bench.datasets.minneapple import load_minneapple
from bench.datasets.wgisd import load_wgisd
from bench.detector_sanity import run
from countonce.cli import main

W, H = 96, 64


def _img(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(path)


def _blob_img(path: Path, n: int) -> list[list[float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    yolo = []
    for k in range(n):
        x, y = 8 + 16 * k, 20
        arr[y - 4 : y + 4, x - 4 : x + 4] = (255, 0, 0)
        yolo.append([0, x / W, y / H, 8 / W, 8 / H])
    Image.fromarray(arr).save(path)
    return yolo


def _write_yolo(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(str(v) for v in r) + "\n" for r in rows))


def _wgisd(root: Path, counts: dict[str, int] | None = None) -> Path:
    counts = counts or {"a": 3, "b": 2, "c": 4}
    for stem, n in counts.items():
        _write_yolo(root / "data" / f"{stem}.txt", _blob_img(root / "data" / f"{stem}.jpg", n))
    (root / "data" / "a.npz").write_bytes(b"x")
    (root / "train.txt").write_text("a.jpg\nb.jpg\n")
    (root / "test.txt").write_text("c\n")
    return root


def _minneapple(root: Path) -> Path:
    for subset, ext in (("detection", "png"), ("counting", "jpg")):
        for split, n in (("train", 3), ("test", 2)):
            d = root / subset / split
            _blob_img(d / "images" / f"i{n}.{ext}", n)
            m = np.zeros((H, W), dtype=np.uint8)
            for k in range(n):
                m[16:24, 4 + 16 * k : 12 + 16 * k] = k + 1
            (d / "masks").mkdir(parents=True, exist_ok=True)
            Image.fromarray(m).save(d / "masks" / f"i{n}.png")
    _img(root / "detection" / "test" / "images" / "hidden.png", 9)  # no mask: skipped
    return root


def _agv(root: Path) -> Path:
    for i, n in enumerate([3, 2]):
        _write_yolo(root / "labels" / f"s{i}.txt", _blob_img(root / "images" / f"s{i}.jpg", n))
    _img(root / "images" / "unlabelled.jpg", 5)
    (root / "splits.json").write_text(json.dumps({"train": ["s0"], "val": ["s1.jpg"]}))
    counts = {"1": 40, "2": 55}
    (root / "trees").mkdir()
    (root / "trees" / "counts.json").write_text(json.dumps(counts))
    for t in counts:
        for k in range(3):
            _img(root / "trees" / t / "images" / f"{k}.jpg", int(t) * 10 + k)
    return root


def test_wgisd(tmp_path: Path) -> None:
    items = load_wgisd(_wgisd(tmp_path / "w"))
    assert [(i.name, i.split, i.count) for i in items] == [
        ("a.jpg", "train", 3),
        ("b.jpg", "train", 2),
        ("c.jpg", "test", 4),
    ]
    a = items[0]
    assert a.mask_path is not None and items[1].mask_path is None
    assert a.class_name == "grape cluster" and a.boxes_xyxy.shape == (3, 4)
    assert a.boxes_xyxy[0].tolist() == pytest.approx([4, 16, 12, 24])
    assert [i.name for i in load_wgisd(tmp_path / "w", split="test")] == ["c.jpg"]
    assert len(load_wgisd(tmp_path / "w", limit=1)) == 1
    with pytest.raises(ValueError, match="split"):
        load_wgisd(tmp_path / "w", split="val")
    with pytest.raises(FileNotFoundError, match="download_wgisd"):
        load_wgisd(tmp_path / "nope")


def test_minneapple(tmp_path: Path) -> None:
    root = _minneapple(tmp_path / "m")
    det = load_minneapple(root)
    assert [(i.name, i.count) for i in det] == [("detection/i2.png", 2), ("detection/i3.png", 3)]
    assert det[1].points_xy[0].tolist() == pytest.approx([7.5, 19.5])  # centroid of instance 1
    cnt = load_minneapple(root, "train", subset="counting")
    assert [(i.name, i.count) for i in cnt] == [("counting/i3.jpg", 3)]
    with pytest.raises(ValueError, match="subset"):
        load_minneapple(root, subset="nope")
    with pytest.raises(FileNotFoundError, match="download_minneapple"):
        load_minneapple(tmp_path / "nope")


def test_applegrowthvision_images_and_trees(tmp_path: Path) -> None:
    root = _agv(tmp_path / "g")
    imgs = load_applegrowthvision_images(root)
    assert [(i.name, i.split, i.count) for i in imgs] == [
        ("s0.jpg", "train", 3),
        ("s1.jpg", "val", 2),
    ]  # the unlabelled image is not a sanity item
    trees = load_applegrowthvision(root)
    assert [(s.scene_id, s.count, len(s.frames)) for s in trees] == [
        ("agv_tree_1", 40, 3),
        ("agv_tree_2", 55, 3),
    ]
    assert len(load_applegrowthvision(root, max_frames=2)[0].frames) == 2
    assert to_gt_doc(trees, "test")["scenes"] == {
        "agv_tree_1": {"count": 40},
        "agv_tree_2": {"count": 55},
    }
    with pytest.raises(FileNotFoundError, match="download_applegrowthvision"):
        load_applegrowthvision(tmp_path / "nope")


def test_load_dataset_registry(tmp_path: Path) -> None:
    from bench.datasets import load_dataset

    assert len(load_dataset("applegrowthvision", _agv(tmp_path / "g"))) == 2


def test_shared_validation_quarantines(tmp_path: Path) -> None:
    root = _wgisd(tmp_path / "w", {"a": 3, "b": 0, "c": 4})  # b: empty GT
    qp = tmp_path / "q.jsonl"
    assert [i.name for i in load_wgisd(root, quarantine_path=qp)] == ["a.jpg", "c.jpg"]
    assert V.load_quarantine(qp) == {"wgisd:b.jpg"}
    assert len(load_wgisd(root, validate=False)) == 3
    # out-of-bounds box
    _write_yolo(root / "data" / "a.txt", [[0, 0.99, 0.5, 0.5, 0.2]])
    q = V.validate_dataset("wgisd", root, tmp_path / "q2.jsonl")
    assert q.quarantined["wgisd:a.jpg"] == [V.OUT_OF_BOUNDS]


@pytest.mark.parametrize(
    ("name", "build", "n"),
    [
        ("wgisd", _wgisd, 3),
        ("minneapple", _minneapple, 2),
        ("applegrowthvision", _agv, 2),
        ("applegrowthvision_images", _agv, 2),
    ],
)
def test_validate_dataset_clean(tmp_path: Path, name: str, build, n: int) -> None:  # type: ignore[no-untyped-def]
    q = V.validate_dataset(name, build(tmp_path / "d"), tmp_path / "q.jsonl")
    assert q.n_scenes == n and not q.quarantined


def test_sanity_harness_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from countonce.detect.base import FakeDetector

    monkeypatch.setattr("bench.detector_sanity.build_detector", lambda name: FakeDetector())
    root = _wgisd(tmp_path / "w")
    out = tmp_path / "r" / "s.csv"
    s = run(root, "train", None, "color_blob", out, dataset="wgisd")
    assert s.n == 2 and s.mae == 2.5  # FakeDetector predicts 0; GT 3 and 2
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["dataset"] == "wgisd" and rows[0]["prompt"] == "grape cluster"
    out2 = tmp_path / "s2.csv"
    mroot = _minneapple(tmp_path / "m")
    argv = [
        "bench",
        "detector-sanity",
        "--dataset",
        "minneapple",
        "--detector",
        "fake",
        "--root",
        str(mroot),
    ]
    rc = main([*argv, "--split", "train", "--sanity-out", str(out2)])
    assert rc == 0 and out2.exists()
    assert main(["bench", "detector-sanity", "--dataset", "nope"]) == 2


def test_evaluate_opt_in_dataset_via_sweep(tmp_path: Path) -> None:
    from bench.datasets.common import to_sweep_scenes

    trees = load_applegrowthvision(_agv(tmp_path / "g"))
    sw = to_sweep_scenes(trees, tmp_path / "out")
    assert [s["id"] for s in sw] == ["agv_tree_1", "agv_tree_2"]
    assert sw[0]["gt"]["count"] == 40 and sw[0]["gt"]["dataset"] == "applegrowthvision"
