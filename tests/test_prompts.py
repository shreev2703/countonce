"""Natural-language / negative prompts (PLAN 15, 9.8b)."""

import hashlib

import pytest

from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.detect.prompts import detect_with_negatives, parse_prompt
from countonce.pipeline import run
from countonce.types import Frame
from countonce.warnings import NEGATIVES_IGNORED
from tests.test_golden import GOLDEN, seeded_scene
from tests.test_pipeline import PlaneRecon


@pytest.mark.parametrize(
    ("text", "pos", "neg"),
    [
        ("red apples, not leaves", ["red apples"], ["leaves"]),
        ("apples but not green apples", ["apples"], ["green apples"]),
        ("apples excluding leaves", ["apples"], ["leaves"]),
        ("apples, not leaves, not stems", ["apples"], ["leaves", "stems"]),
        ("apples excluding leaves, stems", ["apples"], ["leaves", "stems"]),
        ("apples", ["apples"], []),
        ("apples, oranges", ["apples, oranges"], []),
        ("knotted rope", ["knotted rope"], []),
    ],
)
def test_parse_table(text, pos, neg):
    p = parse_prompt(text)
    assert (p.positives, p.negatives) == (pos, neg)


def test_plain_prompt_unchanged():
    assert parse_prompt("  red apples ").positive_prompt == "red apples"


def test_only_negative_rejected():
    with pytest.raises(ValueError):
        parse_prompt("not leaves")


class PhraseFake(FakeDetector):
    """Scores depend on the caption: ``by_phrase`` maps phrase -> {frame: boxes}."""

    supports_negatives = True

    def __init__(self, by_phrase):
        super().__init__()
        self.by_phrase = by_phrase
        self.prompts = []

    def detect(self, frames, prompt, exemplars=None):
        self.prompts.append(prompt)
        script = next((v for k, v in self.by_phrase.items() if k in prompt), {})
        self.script = script
        return super().detect(frames, prompt, exemplars)


def _frame():
    return Frame(
        id=0, path="/tmp/0.jpg", width=200, height=100, sha256=hashlib.sha256(b"0").hexdigest()
    )


def test_negative_box_dropped_positive_kept():
    apple, leaf = (10, 10, 30, 30), (100, 10, 130, 40)
    det = PhraseFake(
        {
            "apples": {0: [(*apple, 0.8), (*leaf, 0.4)]},  # the leaf also fires for "apples"
            "leaves": {0: [(*leaf, 0.7)]},
        }
    )
    out = detect_with_negatives(det, [_frame()], parse_prompt("apples, not leaves"), 0.0)
    assert [tuple(d.box_xyxy) for d in out] == [tuple(map(float, apple))]
    assert det.prompts[0] == "apples"


def test_margin_keeps_close_calls():
    box = (10, 10, 30, 30)
    det = PhraseFake({"apples": {0: [(*box, 0.5)]}, "leaves": {0: [(*box, 0.6)]}})
    parsed = parse_prompt("apples, not leaves")
    assert detect_with_negatives(det, [_frame()], parsed, 0.0) == []
    assert len(detect_with_negatives(det, [_frame()], parsed, 0.2)) == 1


def test_no_negatives_single_call_unchanged():
    det = PhraseFake({"apples": {0: [(10, 10, 30, 30, 0.8)]}})
    out = detect_with_negatives(det, [_frame()], parse_prompt("apples"), 0.0)
    assert len(out) == 1 and det.prompts == ["apples"]


def test_pipeline_unchanged_without_negatives(tmp_path):
    img_dir, script = seeded_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0, assoc="v1")
    res = run(img_dir, "widgets", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    assert res.count == GOLDEN["v1"]
    assert NEGATIVES_IGNORED not in [w["code"] for w in res.warnings]


def test_pipeline_ignored_negatives_warn(tmp_path):
    img_dir, script = seeded_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0, assoc="v1")
    res = run(
        img_dir,
        "widgets, not leaves",
        cfg,
        detector=FakeDetector(script),
        reconstructor=PlaneRecon(),
    )
    assert res.count == GOLDEN["v1"]
    assert NEGATIVES_IGNORED in [w["code"] for w in res.warnings]
