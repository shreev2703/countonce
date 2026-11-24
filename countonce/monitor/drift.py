"""Input/output drift monitoring (PLAN 16B.2): population-stability index per feature.

Rows come from ``result.json`` files (features derived from what the result records) or from a
bench CSV / ``bench.slices`` metadata columns. Numeric features are binned on the reference
set's quantiles; categorical features (the prompt) use PSI over the category union plus a
chi-square statistic. Missing values (NaN) are dropped per feature; a feature with no usable
values on either side is reported under ``skipped`` rather than silently scored.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NUMERIC_FEATURES = (
    "n_frames",
    "resolution_mpix",
    "mean_depth_conf",
    "lift_failure_rate",
    "detection_density",
)
CATEGORICAL_FEATURES = ("prompt",)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Bench CSV / SceneMeta column names that stand for a drift feature.
_ALIASES = {
    "n_views": "n_frames",
    "n_frames": "n_frames",
    "frame_count": "n_frames",
    "resolution_mpix": "resolution_mpix",
    "mean_depth_conf": "mean_depth_conf",
    "lift_failure_rate": "lift_failure_rate",
    "density": "detection_density",
    "detection_density": "detection_density",
    "prompt": "prompt",
}

EPS = 1e-4
N_BINS = 10
MIN_ROWS = 2


@dataclass(frozen=True)
class DriftThresholds:
    warn: float = 0.1
    alert: float = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.warn <= self.alert:
            raise ValueError("drift thresholds need 0 <= warn <= alert")

    def level(self, psi: float) -> str:
        return "alert" if psi >= self.alert else "warn" if psi >= self.warn else "ok"


def _psi_from_counts(ref: np.ndarray, cur: np.ndarray, eps: float) -> float:
    p = np.maximum(ref / ref.sum(), eps)
    q = np.maximum(cur / cur.sum(), eps)
    return float(np.sum((q - p) * np.log(q / p)))


def numeric_psi(
    reference: Sequence[float], current: Sequence[float], *, bins: int = N_BINS, eps: float = EPS
) -> float:
    """PSI with quantile bin edges from ``reference`` (duplicates collapsed), epsilon-smoothed."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    k = edges.size + 1
    rc = np.bincount(np.searchsorted(edges, ref, side="left"), minlength=k)
    cc = np.bincount(np.searchsorted(edges, cur, side="left"), minlength=k)
    return _psi_from_counts(rc.astype(float), cc.astype(float), eps)


def categorical_drift(
    reference: Sequence[str], current: Sequence[str], *, eps: float = EPS
) -> dict[str, float]:
    """PSI over the union of categories, and the chi-square statistic of ``current`` against
    reference proportions (smoothed so unseen categories stay finite)."""
    if not reference or not current:
        raise ValueError("categorical drift needs non-empty reference and current samples")
    cats = sorted(set(reference) | set(current))
    rc = Counter(reference)
    cc = Counter(current)
    r = np.array([rc[c] for c in cats], float)
    c = np.array([cc[c] for c in cats], float)
    psi = _psi_from_counts(r, c, eps)
    p = np.maximum(r / r.sum(), eps)
    expected = p / p.sum() * c.sum()
    chi2 = float(np.sum((c - expected) ** 2 / expected))
    return {"psi": psi, "chi2": chi2, "dof": float(len(cats) - 1)}


# ---------------------------------------------------------------- row loading


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


_DEPTH_RE = re.compile(r"depth confidence\s+([0-9]*\.?[0-9]+)")


def row_from_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Drift features recorded by (or derivable from) one ``result.json``.

    Optional top-level / config keys ``mean_depth_conf``, ``image_width``/``image_height`` are
    used when present; otherwise the depth confidence is recovered from the low-depth-confidence
    warning text when that warning fired, and resolution is left blank.
    """
    cfg = raw.get("config") or {}
    per_frame = raw.get("per_frame_counts") or {}
    status = raw.get("detection_status") or {}
    n_frames = float(len(per_frame)) if per_frame else float("nan")
    n_det = len(status)
    fallbacks = _num(cfg.get("lift_fallbacks"))
    depth = _num(raw.get("mean_depth_conf", cfg.get("mean_depth_conf")))
    if math.isnan(depth):
        for w in raw.get("warnings") or []:
            m = _DEPTH_RE.search(str(w.get("message", "")))
            if m:
                depth = float(m.group(1))
                break
    w_px, h_px = _num(cfg.get("image_width")), _num(cfg.get("image_height"))
    return {
        "n_frames": n_frames,
        "resolution_mpix": w_px * h_px / 1e6,
        "mean_depth_conf": depth,
        "lift_failure_rate": fallbacks / n_det if n_det and not math.isnan(fallbacks) else math.nan,
        "detection_density": n_det / n_frames if n_det and n_frames == n_frames else math.nan,
        "prompt": raw.get("prompt"),
    }


def row_from_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Map a ``SceneMeta.as_dict()`` / bench CSV row onto drift feature names."""
    out: dict[str, Any] = {f: math.nan for f in NUMERIC_FEATURES}
    out["prompt"] = None
    for key, val in meta.items():
        feat = _ALIASES.get(key)
        if feat is None:
            continue
        if feat == "prompt":
            out[feat] = (val or None) if isinstance(val, str) else None
        else:
            out[feat] = _num(val)
    return out


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Rows from a bench CSV, or a directory / file tree of ``result.json`` files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"drift input not found: {p}")
    if p.is_dir():
        from countonce.monitor.etl import is_export, load_drift_rows

        if is_export(p):  # DuckDB over the OLAP Parquet export instead of the live store
            return load_drift_rows(p)
    if p.is_file():
        if p.suffix.lower() == ".csv":
            with p.open(newline="") as f:
                return [row_from_meta(r) for r in csv.DictReader(f)]
        return [row_from_result(json.loads(p.read_text()))]
    rows = []
    for fp in sorted(p.rglob("result.json")):
        rows.append(row_from_result(json.loads(fp.read_text())))
    if not rows:
        csvs = sorted(p.glob("*.csv"))
        if csvs:
            return load_rows(csvs[0])
    return rows


# ---------------------------------------------------------------- report


def compute_drift(
    reference: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    thresholds: DriftThresholds | None = None,
) -> dict[str, Any]:
    """PSI per feature plus flags. Raises ValueError if either side has < 2 rows."""
    th = thresholds or DriftThresholds()
    for name, rows in (("reference", reference), ("current", current)):
        if len(rows) < MIN_ROWS:
            raise ValueError(
                f"drift needs at least {MIN_ROWS} {name} rows to compare distributions "
                f"(got {len(rows)})"
            )
    feats: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for f in NUMERIC_FEATURES:
        r = [x for x in (_num(row.get(f)) for row in reference) if x == x]
        c = [x for x in (_num(row.get(f)) for row in current) if x == x]
        if not r or not c:
            skipped[f] = "no values in " + ("reference" if not r else "current")
            continue
        psi = numeric_psi(r, c)
        feats[f] = {
            "kind": "numeric",
            "psi": psi,
            "level": th.level(psi),
            "n_reference": len(r),
            "n_current": len(c),
            "reference_mean": float(np.mean(r)),
            "current_mean": float(np.mean(c)),
        }
    for f in CATEGORICAL_FEATURES:
        r_c = [str(row[f]) for row in reference if row.get(f)]
        c_c = [str(row[f]) for row in current if row.get(f)]
        if not r_c or not c_c:
            skipped[f] = "no values in " + ("reference" if not r_c else "current")
            continue
        d = categorical_drift(r_c, c_c)
        feats[f] = {
            "kind": "categorical",
            **d,
            "level": th.level(d["psi"]),
            "n_reference": len(r_c),
            "n_current": len(c_c),
        }
    if not feats:
        raise ValueError("no drift feature is present in both reference and current rows")
    flagged = {f: v["level"] for f, v in feats.items() if v["level"] != "ok"}
    return {
        "thresholds": {"warn": th.warn, "alert": th.alert},
        "n_reference": len(reference),
        "n_current": len(current),
        "features": feats,
        "skipped": skipped,
        "flagged": flagged,
        "status": "alert" if "alert" in flagged.values() else "warn" if flagged else "ok",
    }


def run_drift(
    reference: str | Path,
    current: str | Path,
    out: str | Path | None = None,
    thresholds: DriftThresholds | None = None,
) -> dict[str, Any]:
    report = compute_drift(load_rows(reference), load_rows(current), thresholds)
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
