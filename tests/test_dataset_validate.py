"""Data-quality gates for real-data loaders: each rule fires on a deliberately broken scene."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bench.datasets import validate as V
from bench.datasets.common import BenchScene
from bench.datasets.fsc147 import FscImage, load_fsc147
from bench.datasets.fuji_sfm import FujiScene, load_fuji_sfm
from bench.datasets.onescene import load_onescene
from bench.evaluate import Predictions, ScenePrediction, evaluate


def _img(path: Path, seed: int, size: tuple[int, int] = (32, 24), fmt: str = "JPEG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format=fmt)
    return path


def _scene(tmp_path: Path, n: int = 3, **kw) -> BenchScene:
    frames = [_img(tmp_path / f"f{i}.jpg", i) for i in range(n)]
    return BenchScene(
        scene_id=kw.pop("scene_id", "s"), frames=frames, count=kw.pop("count", 5), **kw
    )


def _reasons(issues: list[V.Issue]) -> set[str]:
    return {i.reason for i in issues}


def test_clean_scene_passes(tmp_path):
    assert V.validate_scene(_scene(tmp_path, extra={"n_gt_items": 5, "n_cameras": 3})) == []


def test_empty_and_missing_gt(tmp_path):
    assert V.EMPTY_GT in _reasons(V.validate_scene(_scene(tmp_path, count=0)))
    s = _scene(tmp_path)
    s.count = None  # type: ignore[assignment]
    assert V.MISSING_GT in _reasons(V.validate_scene(s))


def test_gt_count_mismatch(tmp_path):
    s = _scene(tmp_path, extra={"n_gt_items": 6})
    assert V.GT_COUNT_MISMATCH in _reasons(V.validate_scene(s))


def test_image_count_mismatch(tmp_path):
    s = _scene(tmp_path, extra={"n_cameras": 4})
    assert V.IMAGE_COUNT_MISMATCH in _reasons(V.validate_scene(s))
    assert V.IMAGE_COUNT_MISMATCH in _reasons(V.validate_scene(BenchScene("e", [], 3)))


def test_unreadable_and_empty_images(tmp_path):
    s = _scene(tmp_path)
    s.frames[0].write_bytes(b"not an image")
    s.frames[1].write_bytes(b"")
    issues = V.validate_scene(s)
    assert [i.reason for i in issues].count(V.UNREADABLE_IMAGE) == 2


def test_truncated_images(tmp_path):
    s = _scene(tmp_path)
    data = s.frames[0].read_bytes()
    s.frames[0].write_bytes(data[:-10])
    png = _img(tmp_path / "p.png", 9, fmt="PNG")
    png.write_bytes(png.read_bytes()[:-20])
    s.frames.append(png)
    issues = V.validate_scene(s)
    assert [i.reason for i in issues].count(V.TRUNCATED_IMAGE) == 2


def test_duplicate_images(tmp_path):
    s = _scene(tmp_path)
    dup = tmp_path / "copy.jpg"
    dup.write_bytes(s.frames[0].read_bytes())
    s.frames.append(dup)
    assert V.DUPLICATE_IMAGE in _reasons(V.validate_scene(s))


def test_out_of_bounds_points_and_boxes(tmp_path):
    ok = _scene(tmp_path, extra={"points_xy": [[1, 1], [31, 23]], "boxes_xyxy": [[0, 0, 32, 24]]})
    assert V.validate_scene(ok) == []
    bad_pt = _scene(tmp_path, extra={"points_xy": [[40, 1]]})
    bad_box = _scene(tmp_path, extra={"boxes_xyxy": [[5, 5, 2, 9]]})
    nan_pt = _scene(tmp_path, extra={"points_xy": [[float("nan"), 1]]})
    for s in (bad_pt, bad_box, nan_pt):
        assert V.OUT_OF_BOUNDS in _reasons(V.validate_scene(s))


def _fsc(tmp_path: Path, pts, boxes=((0, 0, 10, 10),), name="1.jpg") -> FscImage:
    p = _img(tmp_path / name, 3)
    return FscImage(name, p, "apple", np.array(pts, float), np.array(boxes, float), "val")


def test_fsc_image_rules(tmp_path):
    assert V.validate_scene(_fsc(tmp_path, [[3, 3]])) == []
    assert V.EMPTY_GT in _reasons(V.validate_scene(_fsc(tmp_path, np.zeros((0, 2)))))
    assert V.OUT_OF_BOUNDS in _reasons(V.validate_scene(_fsc(tmp_path, [[99, 3]])))
    gone = _fsc(tmp_path, [[3, 3]])
    gone.path.unlink()
    assert V.UNREADABLE_IMAGE in _reasons(V.validate_scene(gone))


def _fuji(tmp_path: Path, n_gt=4, n_ids=4, n_img=3, n_k=0) -> FujiScene:
    imgs = [_img(tmp_path / f"i{i}.jpg", i) for i in range(n_img)]
    return FujiScene(
        root=tmp_path,
        image_paths=imgs,
        gt_ids=[f"a{i}" for i in range(n_ids)],
        gt_xyz=np.zeros((n_gt, 3)),
        intrinsics={i: np.eye(3) for i in range(n_k)},
    )


def test_fuji_rules(tmp_path):
    assert V.validate_scene(_fuji(tmp_path, n_k=3)) == []
    assert V.EMPTY_GT in _reasons(V.validate_scene(_fuji(tmp_path, 0, 0)))
    assert V.GT_COUNT_MISMATCH in _reasons(V.validate_scene(_fuji(tmp_path, 4, 3)))
    assert V.IMAGE_COUNT_MISMATCH in _reasons(V.validate_scene(_fuji(tmp_path, n_k=2)))
    f = _fuji(tmp_path)
    f.gt_xyz[0, 0] = np.inf
    assert V.OUT_OF_BOUNDS in _reasons(V.validate_scene(f))


def test_quarantine_jsonl_matches_synth_record(tmp_path):
    good = _scene(tmp_path / "g", scene_id="good")
    bad = _scene(tmp_path / "b", scene_id="bad", count=0)
    out = tmp_path / "q" / "quarantine.jsonl"
    clean, q = V.filter_scenes("demo", [good, bad], out)
    assert [s.scene_id for s in clean] == ["good"]
    assert q.n_scenes == 2 and q.quarantined == {"bad": [V.EMPTY_GT]}  # synth: scene_id -> reasons
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["scene_id"] == "bad" and rec["reasons"] == [V.EMPTY_GT]
    assert V.load_quarantine(out) == {"bad"}


def _onescene_root(root: Path, counts: dict[str, int]) -> Path:
    for sid in counts:
        for i in range(3):
            _img(root / "scenes" / sid / "images" / f"{i}.jpg", hash((sid, i)) % 1000)
    entries = [
        {"scene_id": sid, "category": "chair", "count": c, "split": "test"}
        for sid, c in counts.items()
    ]
    (root / "counting.json").write_text(json.dumps({"scenes": entries}))
    return root


def test_loader_validate_default_quarantines(tmp_path):
    root = _onescene_root(tmp_path / "d", {"ok": 4, "zero": 0})
    qp = tmp_path / "q.jsonl"
    scenes = load_onescene(root, quarantine_path=qp)
    assert [s.scene_id for s in scenes] == ["onescene_ok"]
    assert V.load_quarantine(qp) == {"onescene_zero"}
    assert len(load_onescene(root, validate=False)) == 2


def test_fuji_loader_raises_and_fsc_loader_drops(tmp_path):
    root = tmp_path / "fuji"
    _img(root / "2-SfM-set" / "raw_images_east_side" / "a.JPG", 1)
    (root / "2-SfM-set" / "raw_images_west_side").mkdir()
    gt = root / "3-3D_data" / "GroundTruth-3D_apples_locations" / "test_set"
    gt.mkdir(parents=True)
    np.savetxt(gt / "Tree01-annotations_000000.txt", np.full((2, 3), np.nan))
    with pytest.raises(V.DataQualityError):
        load_fuji_sfm(root, quarantine_path=tmp_path / "fq.jsonl")
    assert (tmp_path / "fq.jsonl").exists()
    assert load_fuji_sfm(root, validate=False).image_paths


def test_validate_dataset_unknown(tmp_path):
    with pytest.raises(ValueError):
        V.validate_dataset("nope", tmp_path)


def test_validate_dataset_end_to_end(tmp_path):
    root = _onescene_root(tmp_path / "d", {"ok": 4, "zero": 0})
    q = V.validate_dataset("onescene", root, tmp_path / "q.jsonl")
    assert q.n_scenes == 2 and list(q.quarantined) == ["onescene_zero"]


def test_fsc_loader_default_drops(tmp_path):
    root = tmp_path / "fsc"
    _img(root / "images_384_VarV2" / "1.jpg", 1)
    _img(root / "images_384_VarV2" / "2.jpg", 2)
    box = [[0, 0], [0, 5], [5, 5], [5, 0]]
    ann = {
        "1.jpg": {"points": [[1, 1]], "box_examples_coordinates": [box]},
        "2.jpg": {"points": [], "box_examples_coordinates": [box]},
    }
    (root / "annotation_FSC147_384.json").write_text(json.dumps(ann))
    (root / "Train_Test_Val_FSC_147.json").write_text(json.dumps({"val": ["1.jpg", "2.jpg"]}))
    (root / "ImageClasses_FSC147.txt").write_text("1.jpg\tapple\n2.jpg\tapple\n")
    assert [i.name for i in load_fsc147(root, "val")] == ["1.jpg"]
    assert len(load_fsc147(root, "val", validate=False)) == 2


# --- evaluator ---------------------------------------------------------------------------------


def _preds(ids: list[str]) -> Predictions:
    return Predictions(
        method="m", split="test", scenes=[ScenePrediction(scene_id=i, count=3) for i in ids]
    )


def _gt(ids: list[str], **extra) -> dict:
    return {"split": "test", "scenes": {i: {"count": 3} for i in ids}, **extra}


def test_evaluator_skips_and_counts_quarantined(tmp_path):
    ids = ["a", "b", "c", "d"]
    res = evaluate(
        _preds(ids), _gt(ids), n_boot=50, quarantine={"b"}, splits_path=tmp_path / "none"
    )
    assert res["n_scenes"] == 3
    assert res["n_quarantined_skipped"] == 1 and res["quarantined_skipped"] == ["b"]
    # predictions need not cover a quarantined scene; GT doc's own list is honoured too
    res2 = evaluate(
        _preds(["a", "c"]),
        _gt(ids, quarantined=["b", "d"]),
        n_boot=50,
        splits_path=tmp_path / "none",
    )
    assert res2["n_scenes"] == 2 and res2["n_quarantined_skipped"] == 2


def test_evaluator_without_quarantine_still_strict(tmp_path):
    from bench.evaluate import PredictionFormatError

    with pytest.raises(PredictionFormatError):
        evaluate(_preds(["a"]), _gt(["a", "b"]), n_boot=50, splits_path=tmp_path / "none")
    res = evaluate(_preds(["a"]), _gt(["a"]), n_boot=50, splits_path=tmp_path / "none")
    assert res["n_quarantined_skipped"] == 0


def test_evaluator_all_quarantined_errors(tmp_path):
    from bench.evaluate import PredictionFormatError

    with pytest.raises(PredictionFormatError):
        evaluate(_preds(["a"]), _gt(["a"]), quarantine={"a"}, splits_path=tmp_path / "none")


def test_cli_validate(tmp_path, capsys):
    from countonce.cli import main

    root = _onescene_root(tmp_path / "d", {"ok": 4, "zero": 0})
    out = tmp_path / "q.jsonl"
    rc = main(
        [
            "bench",
            "validate",
            "--dataset",
            "onescene",
            "--root",
            str(root),
            "--quarantine-out",
            str(out),
        ]
    )
    assert rc == 1 and "1 quarantined" in capsys.readouterr().out
    assert V.load_quarantine(out) == {"onescene_zero"}
