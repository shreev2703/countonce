"""FSC-147 loader and detector sanity harness on a tiny synthetic FSC-147-layout fixture."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bench.datasets.fsc147 import load_fsc147
from bench.detector_sanity import (
    SCHEMA,
    TODO_TEXT,
    render_markdown,
    run,
    run_detector_sanity,
)
from countonce.cli import main
from countonce.detect.base import FakeDetector
from countonce.detect.color_blob import ColorBlobDetector

GT = {"1.jpg": 3, "2.jpg": 5, "3.jpg": 2, "4.jpg": 4}


def _fsc(root: Path) -> Path:
    img_dir = root / "images_384_VarV2"
    img_dir.mkdir(parents=True)
    ann: dict[str, object] = {}
    for name, n in GT.items():
        arr = np.zeros((64, 96, 3), dtype=np.uint8)
        pts = []
        for k in range(n):
            x, y = 8 + 16 * k, 20
            arr[y - 4 : y + 4, x - 4 : x + 4] = (255, 0, 0)
            pts.append([x, y])
        Image.fromarray(arr).save(img_dir / name)
        ann[name] = {
            "points": pts,
            "box_examples_coordinates": [[[4, 16], [4, 24], [12, 24], [12, 16]]],
        }
    (root / "annotation_FSC147_384.json").write_text(json.dumps(ann))
    (root / "Train_Test_Val_FSC_147.json").write_text(
        json.dumps({"train": ["1.jpg"], "val": ["2.jpg", "3.jpg", "4.jpg"], "test": []})
    )
    (root / "ImageClasses_FSC147.txt").write_text("".join(f"{n}\tred ball\n" for n in GT))
    return root


def test_loader_yields_prompt_points_boxes(tmp_path: Path) -> None:
    items = load_fsc147(_fsc(tmp_path / "fsc"), split="val")
    assert [i.name for i in items] == ["2.jpg", "3.jpg", "4.jpg"]
    i = items[0]
    assert i.class_name == "red ball" and i.count == 5
    assert i.points_xy.shape == (5, 2)
    assert i.exemplar_boxes_xyxy.tolist() == [[4, 16, 12, 24]]
    assert len(load_fsc147(tmp_path / "fsc", limit=2)) == 2


def test_color_blob_end_to_end(tmp_path: Path) -> None:
    root = _fsc(tmp_path / "fsc")
    out = tmp_path / "r" / "detector_sanity.csv"
    s = run(root, "val", None, "color_blob", out)
    assert s.n == 3 and s.mae == 0.0 and s.rmse == 0.0
    rows = list(csv.DictReader(io.StringIO(out.read_text())))
    assert [r["kind"] for r in rows] == ["image"] * 3 + ["summary"]
    assert rows[0]["pred_count"] == rows[0]["gt_count"] == "5"


def test_errors_and_summary_nonzero(tmp_path: Path) -> None:
    items = load_fsc147(_fsc(tmp_path / "fsc"), split="val")
    out = tmp_path / "o.csv"
    s = run_detector_sanity(FakeDetector(), items, out_csv=out)  # predicts 0 everywhere
    assert s.mae == pytest.approx((5 + 2 + 4) / 3)
    assert s.mae_lo <= s.mae <= s.mae_hi
    assert s.rmse == pytest.approx(np.sqrt((25 + 4 + 16) / 3))
    with pytest.raises(ValueError, match="no images"):
        run_detector_sanity(ColorBlobDetector(), [], out_csv=out)


def test_csv_schema_stable_and_append_only(tmp_path: Path) -> None:
    items = load_fsc147(_fsc(tmp_path / "fsc"), split="val", limit=2)
    out = tmp_path / "o.csv"
    run_detector_sanity(ColorBlobDetector(), items, out_csv=out)
    run_detector_sanity(ColorBlobDetector(), items, out_csv=out)
    lines = list(csv.reader(io.StringIO(out.read_text())))
    assert tuple(lines[0]) == SCHEMA
    assert sum(1 for r in lines if tuple(r) == SCHEMA) == 1  # header written once
    assert len(lines) == 1 + 2 * 3
    assert all(len(r) == len(SCHEMA) for r in lines)
    assert SCHEMA[:7] == (
        "run_id", "timestamp", "dataset", "split", "detector", "score_threshold", "kind"
    )  # fmt: skip


def test_missing_root_is_clear(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(FileNotFoundError, match="download_fsc147.sh"):
        load_fsc147(tmp_path / "nope")
    rc = main(
        ["bench", "detector-sanity", "--root", str(tmp_path / "nope"),
         "--sanity-out", str(tmp_path / "o.csv")]
    )  # fmt: skip
    assert rc == 2 and "download_fsc147.sh" in capsys.readouterr().err
    assert not (tmp_path / "o.csv").exists()


def test_cli_end_to_end_and_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fsc(tmp_path / "fsc")
    out = tmp_path / "o.csv"
    assert render_markdown(out) == TODO_TEXT + "\n"
    rc = main(
        ["bench", "detector-sanity", "--dataset", "fsc147", "--split", "val", "--limit", "2",
         "--detector", "color_blob", "--root", str(root), "--sanity-out", str(out)]
    )  # fmt: skip
    assert rc == 0 and "MAE 0.00" in capsys.readouterr().out
    assert "color_blob" in render_markdown(out)
