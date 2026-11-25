"""Per-frame NMS and the near-duplicate-frame collapse warning (task M2)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from countonce.detect.base import nms_per_frame
from countonce.types import Detection


def _d(frame: int, k: int, score: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(
        id=Detection.make_id(frame, k), frame_id=frame, score=score, box_xyxy=np.array(box)
    )


def test_nms_off_by_default_value() -> None:
    dets = [_d(0, 0, 0.9, (0, 0, 10, 10)), _d(0, 1, 0.8, (0, 0, 10, 10))]
    assert nms_per_frame(dets, 0.0) == dets


def test_nms_drops_lower_scored_duplicate_per_frame_only() -> None:
    dets = [
        _d(0, 0, 0.7, (0, 0, 10, 10)),
        _d(0, 1, 0.9, (0.5, 0, 10, 10)),  # same object, higher score
        _d(0, 2, 0.8, (50, 50, 60, 60)),  # a different object
        _d(1, 0, 0.6, (0, 0, 10, 10)),  # same pixels in another frame: kept
    ]
    out = nms_per_frame(dets, 0.5)
    assert [d.id for d in out] == ["0:1", "0:2", "1:0"]


def test_nms_keeps_touching_neighbours() -> None:
    dets = [_d(0, 0, 0.9, (0, 0, 10, 10)), _d(0, 1, 0.8, (9, 0, 19, 10))]
    assert len(nms_per_frame(dets, 0.5)) == 2


def test_dedup_collapse_to_one_frame_warns(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    from PIL import Image

    from countonce import pipeline
    from countonce.config import Config

    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.png"
        Image.new("RGB", (32, 24), (190, 190, 190)).save(p)
        paths.append(p)
    cfg = Config(out_dir=str(tmp_path / "out"), detector="color_blob", backbone="fixture")
    with caplog.at_level(logging.WARNING, logger="countonce.pipeline"):
        try:
            pipeline.run(paths, "red sphere", cfg, export=False)
        except Exception:  # noqa: BLE001 - only the ingest warning is under test
            pass
    assert any("kept 1 of 3 frames" in r.getMessage() for r in caplog.records)
