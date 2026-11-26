"""Experiment E1 on Fuji-SfM: trackers fed the detections CountOnce associates (RELATED_WORK §4).

A *prepared window* freezes everything a tracker needs, once per (tree, side, k): per view the
score-filtered detections (box, score, whether the lifted point lies in the tree's region of
interest) and the GT apples visible in that view. Trackers then see the views in capture order
(or shuffled), after a frame stride, and are scored as CountOnce is: count = distinct tracks among
in-ROI detections, GT = distinct visible apples of the views they were shown.

Window preparation needs the cached reconstruction and detections of the Core benchmark; the
tracker runs themselves need only the prepared JSON (ByteTrack) plus the images (SAM 2.1).
"""

from __future__ import annotations

import csv
import json
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.baselines_track import FrameDets, count_tracks

STRIDES = (1, 2, 4, 8)
ORDERS = ("capture", "shuffled")
FIELDS = (
    "split", "tree", "side", "k", "stride", "order", "tracker", "params", "status", "n_views",
    "pred", "gt", "n_dets", "n_dets_roi", "runtime_s", "peak_vram_mb", "error",
)  # fmt: skip


@dataclass
class PreparedView:
    image: str  # path of the downscaled frame the detections live on
    boxes: list[list[float]]
    scores: list[float]
    in_roi: list[bool]  # lifted point inside the region of interest (False if the lift failed)
    gt_visible: list[int]  # indices of GT apples visible in this view


@dataclass
class PreparedWindow:
    tree: str
    side: str
    k: int
    views: list[PreparedView]
    error: str = ""  # set when the window's reconstruction could not be aligned to the GT frame
    countonce: dict[str, float] | None = None  # v1/v2/sync counts (stride 1) + their GT

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> PreparedWindow:
        d = json.loads(s)
        return cls(
            d["tree"],
            d["side"],
            d["k"],
            [PreparedView(**v) for v in d["views"]],
            d.get("error", ""),
            d.get("countonce"),
        )


def select_views(n: int, stride: int, order: str, seed: int) -> list[int]:
    """View indices a tracker is shown: every ``stride``-th view, in capture order or shuffled.

    The shuffle is seeded (``seed``) so every tracker sees the same permutation."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    idx = list(range(0, n, stride))
    if order == "shuffled":
        idx = [int(i) for i in np.random.default_rng(seed).permutation(idx)]
    return idx


def window_seed(tree: str, side: str, k: int, stride: int) -> int:
    return zlib.crc32(f"{tree}/{side}/{k}/{stride}".encode())


def window_frames(win: PreparedWindow, views: Sequence[int]) -> list[FrameDets]:
    return [
        (
            np.array(win.views[v].boxes, float).reshape(-1, 4),
            np.array(win.views[v].scores, float),
        )
        for v in views
    ]


def score_tracks(
    win: PreparedWindow, views: Sequence[int], track_ids: Sequence[np.ndarray]
) -> dict[str, int]:
    """Count and GT of one tracker run (``track_ids[i]`` belongs to ``views[i]``)."""
    flat_ids = np.concatenate([np.asarray(t, int) for t in track_ids]) if track_ids else np.zeros(0)
    keep = [bool(f) for v in views for f in win.views[v].in_roi]
    gt = {g for v in views for g in win.views[v].gt_visible}
    return {
        "pred": count_tracks(flat_ids, keep),
        "gt": len(gt),
        "n_dets": int(len(keep)),
        "n_dets_roi": int(sum(keep)),
    }


def run_bytetrack(win: PreparedWindow, views: Sequence[int], params: Any) -> dict[str, int]:
    from bench.baselines_track import bytetrack

    return score_tracks(win, views, bytetrack(window_frames(win, views), params))


def run_sam2(win: PreparedWindow, views: Sequence[int], tracker: Any) -> dict[str, int]:
    from PIL import Image

    imgs = [Image.open(win.views[v].image).convert("RGB") for v in views]
    return score_tracks(win, views, tracker.track(imgs, window_frames(win, views)))


# ----------------------------------------------------------------------------- CSV (resumable)


def row_key(r: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(r[f]) for f in ("split", "tree", "side", "k", "stride", "order", "tracker"))


def read_done(path: Path) -> set[tuple[str, ...]]:
    """Keys of rows that finished (``ok``/``failed``/``skipped``); ``error`` rows are retried."""
    if not path.exists():
        return set()
    with path.open() as f:
        return {row_key(r) for r in csv.DictReader(f) if r["status"] in ("ok", "failed", "skipped")}


def append_row(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def mean_abs_error(rows: Sequence[dict[str, Any]]) -> float:
    """Mean |pred - gt| over ``ok`` rows (nan if none)."""
    e = [abs(float(r["pred"]) - float(r["gt"])) for r in rows if r["status"] == "ok"]
    return float(np.mean(e)) if e else float("nan")


def paired_bootstrap_ci(
    diffs: Sequence[float], groups: Sequence[str], n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``diffs`` with a 95% bootstrap CI that resamples whole groups (trees)."""
    d = np.asarray(diffs, float)
    g = np.asarray(groups)
    names = sorted(set(g))
    per = [d[g == n] for n in names]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(per), len(per))
        means.append(np.concatenate([per[i] for i in pick]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
