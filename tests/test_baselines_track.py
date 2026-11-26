"""ByteTrack re-implementation and E1 scoring helpers on a synthetic moving-box sequence."""

from __future__ import annotations

import numpy as np
import pytest

from bench import track_e1 as e1
from bench.baselines_track import ByteTrackParams, bytetrack, count_tracks, iou_matrix


def _boxes(t: int, n: int = 3) -> np.ndarray:
    """``n`` 40x40 boxes, far apart, each drifting 4 px per frame to the right."""
    out = [[100 + 200 * i + 4 * t, 100 + 150 * i, 140 + 200 * i + 4 * t, 140 + 150 * i]
           for i in range(n)]  # fmt: skip
    return np.array(out, float)


def _seq(n_frames: int, n: int = 3, score: float = 0.9) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(_boxes(t, n), np.full(n, score)) for t in range(n_frames)]


def test_iou_matrix_basic() -> None:
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[0, 0, 10, 10.0], [5, 0, 15, 10.0], [20, 20, 30, 30.0]])
    got = iou_matrix(a, b)[0]
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(50 / 150)
    assert got[2] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_moving_boxes_keep_their_ids() -> None:
    ids = bytetrack(_seq(8))
    assert all(len(i) == 3 for i in ids)
    for j in range(3):
        assert len({int(i[j]) for i in ids}) == 1
    assert len({int(x) for i in ids for x in i}) == 3


def test_new_object_appearing_mid_sequence_gets_one_new_id() -> None:
    frames = _seq(6)
    for t in range(3, 6):
        frames[t] = (
            np.vstack([_boxes(t), [[900 + 4 * t, 600, 940 + 4 * t, 640.0]]]),
            np.full(4, 0.9),
        )
    ids = bytetrack(frames)
    assert len({int(x) for i in ids for x in i}) == 4


def test_jump_larger_than_the_box_splits_the_track() -> None:
    """A wide viewpoint gap (no overlap between frames) yields duplicate ids."""
    frames = [(_boxes(0), np.full(3, 0.9)), (_boxes(0) + [60, 0, 60, 0], np.full(3, 0.9))]
    assert len({int(x) for i in bytetrack(frames) for x in i}) == 6


def test_low_score_detection_is_rescued_but_never_starts_a_track() -> None:
    p = ByteTrackParams(high_thr=0.5, low_thr=0.2)
    frames = [(_boxes(0, 1), np.array([0.9])), (_boxes(1, 1), np.array([0.3]))]
    ids = bytetrack(frames, p)
    assert ids[0][0] == ids[1][0] >= 0
    lone = bytetrack([(_boxes(0, 1), np.array([0.3]))], p)
    assert lone[0][0] == -1


def test_empty_frames_are_fine() -> None:
    frames = [(np.zeros((0, 4)), np.zeros(0)), *_seq(2, 1)]
    ids = bytetrack(frames)
    assert len(ids[0]) == 0 and ids[1][0] == ids[2][0]


def test_count_tracks_only_counts_kept_detections() -> None:
    assert count_tracks([0, 0, 1, 2, -1], [True, True, True, False, True]) == 2


def test_select_views_stride_and_shuffle() -> None:
    assert e1.select_views(10, 4, "capture", 0) == [0, 4, 8]
    a = e1.select_views(10, 2, "shuffled", 5)
    assert sorted(a) == [0, 2, 4, 6, 8] and a == e1.select_views(10, 2, "shuffled", 5)
    with pytest.raises(ValueError):
        e1.select_views(10, 0, "capture", 0)


def test_score_tracks_uses_roi_and_visible_gt() -> None:
    v0 = e1.PreparedView("a.jpg", [[0, 0, 5, 5], [9, 9, 20, 20]], [0.9, 0.9], [True, False], [1, 2])
    v1 = e1.PreparedView("b.jpg", [[0, 0, 5, 5]], [0.9], [True], [2, 3])
    win = e1.PreparedWindow("T", "east", 2, [v0, v1])
    got = e1.score_tracks(win, [0, 1], [np.array([0, 1]), np.array([0])])
    assert got == {"pred": 1, "gt": 3, "n_dets": 3, "n_dets_roi": 2}
    assert e1.score_tracks(win, [1], [np.array([5])])["gt"] == 2


def test_prepared_window_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    v = e1.PreparedView("a.jpg", [[0, 0, 5, 5]], [0.5], [True], [1])
    w = e1.PreparedWindow("T", "west", 1, [v], countonce={"v2": 3.0})
    assert e1.PreparedWindow.from_json(w.to_json()) == w


def test_csv_resume_and_bootstrap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "r.csv"
    row = {"split": "test", "tree": "T", "side": "east", "k": 5, "stride": 1, "order": "capture",
           "tracker": "bytetrack", "status": "ok", "pred": 3, "gt": 4}  # fmt: skip
    e1.append_row(p, row)
    e1.append_row(p, {**row, "tracker": "sam2", "status": "error"})
    assert e1.read_done(p) == {e1.row_key(row)}
    mean, lo, hi = e1.paired_bootstrap_ci([1.0, 3.0, 2.0, 2.0], ["a", "a", "b", "b"], n_boot=200)
    assert lo <= mean <= hi and mean == pytest.approx(2.0)


def test_shuffle_seed_differs_between_windows() -> None:
    seeds = {e1.window_seed(t, s, 10, 1) for t in ("Tree01", "Tree02") for s in ("east", "west")}
    assert len(seeds) == 4


def test_mask_box_iou() -> None:
    torch = pytest.importorskip("torch")
    from bench.baselines_track import mask_box_iou

    masks = torch.zeros((1, 20, 20), dtype=torch.bool)
    masks[0, 0:10, 0:10] = True
    got = mask_box_iou(masks, np.array([[0, 0, 10, 10], [5, 0, 15, 10], [12, 12, 18, 18.0]]))
    assert got[:, 0] == pytest.approx([1.0, 50 / 150, 0.0])


@pytest.mark.gpu
@pytest.mark.slow
def test_sam2_keeps_ids_on_a_moving_box_sequence() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from PIL import Image, ImageDraw

    from bench.baselines_track import Sam2Params, Sam2Tracker

    frames, imgs = [], []
    for t in range(5):
        # the third square only appears from frame 2 (a mid-stream prompt)
        boxes = np.array(
            [[60 + 8 * t, 60, 140 + 8 * t, 140], [300 + 8 * t, 200, 380 + 8 * t, 280]]
            + ([[500 + 8 * t, 340, 580 + 8 * t, 420]] if t >= 2 else []),
            float,
        )
        im = Image.new("RGB", (640, 480), (30, 30, 30))
        draw = ImageDraw.Draw(im)
        for b, colour in zip(boxes, ("red", "lime", "blue"), strict=False):
            draw.rectangle(b.tolist(), fill=colour)
        imgs.append(im)
        frames.append((boxes, np.full(len(boxes), 0.9)))
    ids = Sam2Tracker(Sam2Params()).track(imgs, frames)
    assert [i.tolist() for i in ids] == [[0, 1], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
