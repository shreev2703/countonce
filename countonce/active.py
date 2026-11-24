"""Active-learning selection: which finished jobs should a pilot user review next (PLAN §16B.3).

A job is worth reviewing when the matcher was unsure about it. Two signals, both read from the
persisted ``result.json``:

* **match uncertainty**: how close the calibrated match probabilities are to 0.5
  (``1 - 2|p - 0.5|`` per evidence row, averaged), and the share of rows inside ``band``;
* **interval width**: the conformal interval width relative to the count, clipped at ``width_cap``.

The score is a weighted sum of the two, in [0, 1]. Jobs that already carry a ``corrected/``
directory are skipped by default, since a reviewer has looked at them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from countonce.explain import load_result_dir
from countonce.schemas import ResultV11


@dataclass
class ReviewItem:
    job_id: str
    path: str
    score: float
    prob_uncertainty: float  # mean closeness of match probabilities to 0.5, in [0, 1]
    frac_in_band: float  # share of match evidence with p within `band` of 0.5
    rel_interval_width: float  # (high - low) / max(count, 1)
    count: int
    n_evidence: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prob_uncertainty(probs: Iterable[float]) -> float:
    p = list(probs)
    return sum(1.0 - 2.0 * abs(x - 0.5) for x in p) / len(p) if p else 0.0


def score_result(
    result: ResultV11,
    job_id: str = "",
    path: str = "",
    *,
    band: float = 0.15,
    width_cap: float = 1.0,
    w_prob: float = 0.5,
    w_width: float = 0.5,
) -> ReviewItem:
    probs = [e.match_prob for e in result.match_evidence]
    unc = prob_uncertainty(probs)
    in_band = sum(abs(p - 0.5) <= band for p in probs) / len(probs) if probs else 0.0
    width = (result.interval.high - result.interval.low) / max(result.count, 1)
    total = w_prob + w_width
    score = (w_prob * unc + w_width * min(width / width_cap, 1.0)) / total if total else 0.0
    return ReviewItem(job_id, path, score, unc, in_band, width, result.count, len(probs))


def rank_jobs(
    job_dirs: Iterable[Path | str],
    *,
    skip_reviewed: bool = True,
    **weights: float,
) -> list[ReviewItem]:
    """Score every directory holding a ``result.json``; most uncertain first (ties by job id)."""
    items = []
    for d in map(Path, job_dirs):
        if skip_reviewed and (d / "corrected").is_dir():
            continue
        items.append(score_result(load_result_dir(d), d.name, str(d), **weights))
    return sorted(items, key=lambda i: (-i.score, i.job_id))


def find_jobs(root: Path | str) -> list[Path]:
    """Job directories under ``root`` (``root`` itself counts when it holds a result.json)."""
    r = Path(root)
    if (r / "result.json").is_file():
        return [r]
    return sorted(p.parent for p in r.glob("*/result.json"))


def export_review_queue(
    items: Iterable[ReviewItem], path: Path | str, top: int | None = None
) -> Path:
    """Write the ranked queue as JSONL (one item per line, with a 1-based ``rank``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    chosen = list(items)[:top]
    with p.open("w", encoding="utf-8") as f:
        for rank, it in enumerate(chosen, 1):
            f.write(json.dumps({"rank": rank, **it.to_dict()}, sort_keys=True) + "\n")
    return p
