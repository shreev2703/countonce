"""Data-backed warning thresholds (PLAN 8.8 / 9.9): tune ``WarningThresholds`` from bench rows.

Reads ``results/bench.csv`` rows of the calibration (``val``) split only; the frozen ``test``
split is refused (``bench.splits``). For each warning, candidate cuts are scored with the
§9.9 slice flagging (``bench.slices.slice_report``): the chosen cut is the best one whose
slice error is significantly worse than overall. With too few scenes, or no flagged cut, the
current default is kept and recorded with ``source: default``.

Only covariates present in the CSV (``n_views``, ``low``/``high``/``pred``) or in the run's
``result.json`` (instance view counts) can be tuned. Depth confidence, failed-lift share and
the small-count rule are not stored per run, so they keep their defaults.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from countonce.warnings import WarningThresholds

from .slices import SceneMeta, SceneRecord, slice_report
from .splits import DEFAULT_PATH, load_splits

MIN_SCENES = 15  # fewer scenes than this: keep defaults (the §8.8 real-scene rule)
MIN_SLICE = 5


class SplitViolation(ValueError):
    """Rows from the frozen test split (or an unassigned real scene) were offered for tuning."""


@dataclass(frozen=True)
class TuneRow:
    scene_id: str
    n_views: int
    pred: float
    gt: float
    low: float
    high: float
    multiview_fraction: float = float("nan")

    @property
    def rel_width(self) -> float:
        return (self.high - self.low) / max(self.pred, 1.0)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _f(x: str | None) -> float:
    try:
        return float(x) if x not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def _multiview_fraction(run_dir: str) -> float:
    p = Path(run_dir) / "result.json" if run_dir else None
    if p is None or not p.is_file():
        return float("nan")
    try:
        inst = json.loads(p.read_text()).get("instances") or []
        if not inst:
            return float("nan")
        return sum(len(i.get("views", [])) >= 2 or i.get("n_views", 0) >= 2 for i in inst) / len(
            inst
        )
    except (ValueError, AttributeError, TypeError):
        return float("nan")


def load_rows(
    csv_path: str | Path,
    *,
    method: str,
    splits_path: str | Path = DEFAULT_PATH,
    split: str = "val",
) -> list[TuneRow]:
    """Finished rows of ``method`` in the ``val`` split. Test-split scenes are never returned."""
    if split == "test":
        raise SplitViolation("the test split is never used for tuning")
    splits = load_splits(splits_path)
    allowed, tests = set(splits.get(split, {})), set(splits.get("test", {}))
    out: list[TuneRow] = []
    with Path(csv_path).open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("method") != method or r.get("status") != "ok":
                continue
            sid = r["scene_id"]
            if sid in tests or r.get("split") == "test":
                continue
            if r.get("domain", "real") != "synthetic" and sid not in allowed:
                continue
            pred, gt = _f(r.get("pred")), _f(r.get("gt"))
            if math.isnan(pred) or math.isnan(gt):
                continue
            low, high = _f(r.get("low")), _f(r.get("high"))
            out.append(
                TuneRow(
                    sid,
                    int(_f(r.get("n_views")) or 0),
                    pred,
                    gt,
                    pred if math.isnan(low) else low,
                    pred if math.isnan(high) else high,
                    _multiview_fraction(r.get("run_dir", "")),
                )
            )
    return out


def _check_no_test(rows: Sequence[TuneRow], splits_path: str | Path) -> None:
    tests = set(load_splits(splits_path).get("test", {}))
    bad = sorted({r.scene_id for r in rows} & tests)
    if bad:
        raise SplitViolation(f"test-split scenes in tuning rows: {bad[:5]}")


def _pick(
    rows: Sequence[TuneRow],
    inside: Callable[[TuneRow, float], bool],
    cuts: Sequence[float],
) -> tuple[float | None, int]:
    """Cut whose slice is significantly worse than overall, with the highest CI lower bound.

    Reuses ``slice_report`` flagging (bootstrap CI of slice MAE minus overall MAE above 0);
    among flagged cuts the one with the largest lower bound is the most confidently bad, and
    ties go to the earlier (listed first) cut. Returns ``(cut, n_in_slice)`` or ``(None, 0)``.
    """
    metas = [SceneMeta(gt_count=r.gt, n_views=r.n_views) for r in rows]
    recs = [
        SceneRecord(f"{r.scene_id}#{i}", m, r.pred)
        for i, (r, m) in enumerate(zip(rows, metas, strict=True))
    ]
    pos = {id(m): i for i, m in enumerate(metas)}
    best: tuple[float, float, int] | None = None
    for c in cuts:
        chosen = frozenset(i for i, r in enumerate(rows) if inside(r, c))
        if len(chosen) < MIN_SLICE:
            continue
        rep = slice_report(
            recs, {"cut": lambda m, ch=chosen: pos[id(m)] in ch}, min_scenes=MIN_SLICE, n_boot=1000
        )
        if rep and rep[0].worse_error and (best is None or rep[0].err_diff_ci[0] > best[1]):
            best = (c, rep[0].err_diff_ci[0], rep[0].n)
    return (best[0], best[2]) if best else (None, 0)


def _entry(value: float, n: int, source: str, default: float) -> dict[str, Any]:
    return {"value": value, "n": n, "source": source, "default": default}


def tune(
    rows: Sequence[TuneRow],
    *,
    csv_sha256: str = "",
    method: str = "",
    splits_path: str | Path = DEFAULT_PATH,
    min_scenes: int = MIN_SCENES,
) -> dict[str, Any]:
    """Tune thresholds from ``rows``; returns the ``configs/warnings.yaml`` document."""
    _check_no_test(rows, splits_path)
    d = WarningThresholds()
    n = len(rows)
    ents: dict[str, dict[str, Any]] = {
        "min_frames": _entry(d.min_frames, n, "default", d.min_frames),
        "min_multiview_fraction": _entry(
            d.min_multiview_fraction, 0, "default", d.min_multiview_fraction
        ),
        "max_range_width_fraction": _entry(
            d.max_range_width_fraction, n, "default", d.max_range_width_fraction
        ),
    }
    if n >= min_scenes:
        # too few frames: slice n_views < c; best (largest) flagged cut wins.
        cuts = sorted({float(r.n_views) for r in rows})[1:]
        c, k = _pick(rows, lambda r, c: r.n_views < c, cuts)
        if c is not None:
            ents["min_frames"] = _entry(int(c), k, "tuned", d.min_frames)
        # wide range: slice rel_width > c; best (smallest) flagged cut wins.
        w = sorted({round(r.rel_width, 4) for r in rows if r.rel_width > 0})
        c, k = _pick(rows, lambda r, c: r.rel_width > c, w)
        if c is not None:
            ents["max_range_width_fraction"] = _entry(
                round(c, 4), k, "tuned", d.max_range_width_fraction
            )
    mv = [r for r in rows if not math.isnan(r.multiview_fraction)]
    if len(mv) >= min_scenes:
        cuts = sorted({round(r.multiview_fraction, 4) for r in mv})[1:]
        c, k = _pick(mv, lambda r, c: r.multiview_fraction < c, cuts)
        ents["min_multiview_fraction"] = (
            _entry(c, k, "tuned", d.min_multiview_fraction)
            if c is not None
            else _entry(d.min_multiview_fraction, len(mv), "default", d.min_multiview_fraction)
        )
    return {
        "source_csv_sha256": csv_sha256,
        "method": method,
        "split": "val",
        "n_scenes": n,
        "thresholds": ents,
        # Flat block that Config(warnings=...) accepts.
        "warnings": {k: v["value"] for k, v in ents.items() if v["source"] == "tuned"},
    }


def write_warnings_yaml(doc: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def tune_from_csv(
    csv_path: str | Path = "results/bench.csv",
    out: str | Path = "configs/warnings.yaml",
    *,
    method: str = "co_v2_vggt",
    splits_path: str | Path = DEFAULT_PATH,
) -> dict[str, Any]:
    rows = load_rows(csv_path, method=method, splits_path=splits_path)
    doc = tune(rows, csv_sha256=file_sha256(csv_path), method=method, splits_path=splits_path)
    write_warnings_yaml(doc, out)
    return doc
