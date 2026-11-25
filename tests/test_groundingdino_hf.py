"""HF GroundingDINO backend with a fake processor and model (no download, no transformers)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from countonce.detect import groundingdino as gd
from countonce.detect.cache import detection_cache_key
from countonce.types import Frame

torch = pytest.importorskip("torch")


class _Inputs(dict):  # type: ignore[type-arg]
    def to(self, _device: str) -> _Inputs:
        return self


class FakeProcessor:
    def __init__(self, boxes: np.ndarray, scores: np.ndarray, labels: list[str]) -> None:
        self.boxes, self.scores, self.labels = boxes, scores, labels
        self.calls: list[dict[str, object]] = []

    def __call__(self, images: object, text: str, return_tensors: str) -> _Inputs:
        self.calls.append({"text": text})
        return _Inputs(input_ids=torch.zeros(1, 3, dtype=torch.long))

    def post_process_grounded_object_detection(
        self,
        outputs: object,
        input_ids: object,
        threshold: float,
        text_threshold: float,
        target_sizes: list[tuple[int, int]],
    ) -> list[dict[str, object]]:
        self.calls[-1].update(threshold=threshold, target_sizes=target_sizes)
        keep = self.scores > threshold
        return [
            {
                "scores": torch.tensor(self.scores[keep]),
                "boxes": torch.tensor(self.boxes[keep]),
                "text_labels": [lb for lb, k in zip(self.labels, keep, strict=True) if k],
            }
        ]


def _frame(tmp_path: Path) -> Frame:
    from PIL import Image

    p = tmp_path / "f.png"
    Image.new("RGB", (200, 100)).save(p)
    return Frame(id=3, path=p, width=200, height=100, sha256="abc")


def _det(proc: FakeProcessor) -> gd.GroundingDinoDetector:
    d = gd.GroundingDinoDetector(device="cpu", floor=0.05, revision="deadbeef")
    d._processor = proc
    d._model = lambda **kw: object()
    return d


def test_default_backend_is_hf_and_package_path_kept() -> None:
    assert gd.GroundingDinoDetector().backend == "hf"
    assert gd.GroundingDinoDetector(config_path="a.py", checkpoint_path="b.pth").backend == (
        "groundingdino"
    )
    with pytest.raises(ValueError):
        gd.GroundingDinoDetector(backend="nope")


def test_hf_outputs_map_to_detections(tmp_path: Path) -> None:
    boxes = np.array([[10, 20, 50, 60], [100, 10, 180, 90], [0, 0, 5, 5]], dtype=np.float32)
    proc = FakeProcessor(
        boxes, np.array([0.9, 0.3, 0.01], dtype=np.float32), ["apple", "apple", ""]
    )
    dets = _det(proc).detect([_frame(tmp_path)], "Apple")
    assert proc.calls[0]["text"] == "apple ."
    assert proc.calls[0]["threshold"] == 0.05  # low raw floor, scores stored
    assert proc.calls[0]["target_sizes"] == [(100, 200)]
    assert [round(d.score, 2) for d in dets] == [0.9, 0.3]  # 0.01 is below the floor
    assert dets[0].box_xyxy is not None
    np.testing.assert_allclose(dets[0].box_xyxy, [10, 20, 50, 60])
    assert dets[0].id == "3:0" and dets[0].label == "apple"


def test_cxcywh_round_trip() -> None:
    from countonce.detect.base import cxcywh_norm_to_xyxy

    b = np.array([[10.0, 20.0, 50.0, 60.0]])
    n = gd.xyxy_px_to_cxcywh_norm(b, 200, 100)
    np.testing.assert_allclose(cxcywh_norm_to_xyxy(n, 200, 100), b)


def test_version_pins_model_and_revision_in_cache_key(tmp_path: Path) -> None:
    proc = FakeProcessor(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [])
    a = _det(proc)
    b = gd.GroundingDinoDetector(device="cpu", revision="cafe")
    assert a.version == "hf:IDEA-Research/grounding-dino-tiny@deadbeef"
    f = _frame(tmp_path)
    ka = detection_cache_key(f, "apple", a.name, a.version, a.config())
    kb = detection_cache_key(f, "apple", b.name, b.version, b.config())
    assert ka != kb
    c = gd.GroundingDinoDetector(device="cpu", revision="deadbeef", model_id="other/model")
    assert detection_cache_key(f, "apple", c.name, c.version, c.config()) != ka


def test_version_reads_local_cache_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refs = tmp_path / "models--a--b" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123\n")
    const = types.ModuleType("huggingface_hub.constants")
    const.HF_HUB_CACHE = str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", const)
    assert gd.cached_revision("a/b") == "abc123"
    assert gd.cached_revision("a/missing") is None


def test_negatives_suppress_with_hf_backend(tmp_path: Path) -> None:
    from countonce.detect.prompts import detect_with_negatives, parse_prompt

    boxes = np.array([[10, 20, 50, 60]], dtype=np.float32)

    class P(FakeProcessor):
        def __call__(self, images: object, text: str, return_tensors: str) -> _Inputs:
            r = super().__call__(images, text, return_tensors)
            self.scores = np.array([0.8 if "leaf" in text else 0.4], dtype=np.float32)
            return r

    proc = P(boxes, np.array([0.4], dtype=np.float32), ["x"])
    det = _det(proc)
    assert det.supports_negatives
    out = detect_with_negatives(det, [_frame(tmp_path)], parse_prompt("apple, not leaf"), 0.1)
    assert out == []  # the negative phrase scores higher on the same box
    assert len(proc.calls) == 2


def test_boxes_are_clipped_to_the_image(tmp_path: Path) -> None:
    boxes = np.array([[-0.1, 5, 50, 100.4], [300, 10, 320, 20]], dtype=np.float32)
    proc = FakeProcessor(boxes, np.array([0.9, 0.8], dtype=np.float32), ["a", "b"])
    dets = _det(proc).detect([_frame(tmp_path)], "apple")
    assert len(dets) == 1  # the second box is entirely outside, so it degenerates and is dropped
    assert dets[0].box_xyxy is not None
    np.testing.assert_allclose(dets[0].box_xyxy, [0, 5, 50, 100])
