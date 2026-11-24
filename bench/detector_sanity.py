"""Single-image detector sanity check on FSC-147 (PLAN §9.1): count error per image.

Runs any ``Detector`` on each image with the class name as prompt, counts detections above a
score threshold, and appends per-image rows plus one summary row (MAE/RMSE with an
image-bootstrap CI, the image being the resampling unit) to ``results/detector_sanity.csv``.
Sanity check of the wrappers only; not a counting benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from bench.datasets.fsc147 import FscImage, load_fsc147
from bench.stats import bootstrap_ci
from countonce.detect.base import Detector, Exemplar, filter_by_score
from countonce.types import Frame

CSV_PATH = Path("results/detector_sanity.csv")
# Append-only: never reorder or remove a column; add new ones at the end only.
SCHEMA = (
    "run_id",
    "timestamp",
    "dataset",
    "split",
    "detector",
    "score_threshold",
    "kind",  # "image" or "summary"
    "image",
    "prompt",
    "gt_count",
    "pred_count",
    "abs_err",
    "sq_err",
    "n",
    "mae",
    "mae_lo",
    "mae_hi",
    "rmse",
    "rmse_lo",
    "rmse_hi",
)
TODO_TEXT = "TODO: fill from results"


@dataclass(frozen=True)
class SanitySummary:
    n: int
    mae: float
    mae_lo: float
    mae_hi: float
    rmse: float
    rmse_lo: float
    rmse_hi: float


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def summarize(
    gt: Sequence[int], pred: Sequence[int], n_boot: int = 2000, seed: int = 0
) -> SanitySummary:
    err = np.abs(np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float))
    m = bootstrap_ci(err, np.mean, n_boot=n_boot, seed=seed)
    r = bootstrap_ci(err, _rmse, n_boot=n_boot, seed=seed)
    return SanitySummary(m.n, m.estimate, m.low, m.high, r.estimate, r.low, r.high)


def _frame(img: FscImage, idx: int) -> Frame:
    with Image.open(img.path) as im:
        w, h = im.size
    return Frame(
        id=idx,
        path=img.path,
        width=w,
        height=h,
        sha256=hashlib.sha256(img.path.read_bytes()).hexdigest(),
    )


def run_detector_sanity(
    detector: Detector,
    images: Sequence[FscImage],
    *,
    dataset: str = "fsc147",
    split: str = "val",
    score_threshold: float = 0.3,
    use_exemplars: bool = False,
    out_csv: str | Path = CSV_PATH,
    n_boot: int = 2000,
) -> SanitySummary:
    """Run ``detector`` per image and append rows to ``out_csv``. Raises on an empty image list."""
    if not images:
        raise ValueError("no images to evaluate (empty split or limit=0)")
    run_id = uuid.uuid4().hex[:12]
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows: list[list[object]] = []
    gts: list[int] = []
    preds: list[int] = []
    for i, img in enumerate(images):
        frame = _frame(img, i)
        ex = (
            [Exemplar(i, tuple(map(float, b))) for b in img.exemplar_boxes_xyxy]  # type: ignore[arg-type]
            if use_exemplars
            else None
        )
        dets = filter_by_score(detector.detect([frame], img.class_name, ex), score_threshold)
        gts.append(img.count)
        preds.append(len(dets))
        e = abs(len(dets) - img.count)
        rows.append(
            [run_id, ts, dataset, split, detector.name, score_threshold, "image", img.name]
            + [img.class_name, img.count, len(dets), e, e * e, 1, "", "", "", "", "", ""]
        )
    s = summarize(gts, preds, n_boot=n_boot)
    rows.append(
        [run_id, ts, dataset, split, detector.name, score_threshold, "summary", "", "", "", "", ""]
        + ["", s.n, s.mae, s.mae_lo, s.mae_hi, s.rmse, s.rmse_lo, s.rmse_hi]
    )
    _append(rows, out_csv)
    return s


def _append(rows: Sequence[Sequence[object]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SCHEMA)
        w.writerows(rows)
    return p


def summary_rows(path: str | Path = CSV_PATH) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("kind") == "summary"]


def render_markdown(path: str | Path = CSV_PATH) -> str:
    """Summary table for docs; ``TODO: fill from results`` when there are no rows."""
    rows = summary_rows(path)
    if not rows:
        return TODO_TEXT + "\n"
    out = ["| detector | split | n | MAE [95% CI] | RMSE [95% CI] |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['detector']} | {r['split']} | {r['n']} "
            f"| {float(r['mae']):.2f} [{float(r['mae_lo']):.2f}, {float(r['mae_hi']):.2f}] "
            f"| {float(r['rmse']):.2f} [{float(r['rmse_lo']):.2f}, {float(r['rmse_hi']):.2f}] |"
        )
    return "\n".join(out) + "\n"


def build_detector(name: str) -> Detector:
    from countonce.config import Config
    from countonce.pipeline import build_detector as _build

    return _build(Config(detector=name))


def run(
    root: str | Path,
    split: str,
    limit: int | None,
    detector_name: str,
    out_csv: str | Path = CSV_PATH,
    score_threshold: float = 0.3,
) -> SanitySummary:
    images = load_fsc147(root, split=split, limit=limit)
    return run_detector_sanity(
        build_detector(detector_name),
        images,
        split=split,
        score_threshold=score_threshold,
        out_csv=out_csv,
    )
