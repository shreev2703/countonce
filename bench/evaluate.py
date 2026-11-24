"""CountOnce-Bench evaluator (PLAN §16): score any method's per-scene predictions against GT.

Usage: ``python -m bench.evaluate --predictions preds.json --gt test [--reference ref.json]``.

Prediction file (validated by :class:`Predictions`, schema in ``bench/schemas/``)::

    {"method": "my-method", "split": "test",
     "scenes": [{"scene_id": "s1", "count": 12, "interval": [10, 14],
                 "assignments": {"det0": "a", "det1": "a"}}]}

GT file (``bench/gt/<split>.json`` or an explicit path)::

    {"split": "test", "scenes": {"s1": {"count": 12, "instances": {"det0": "x", "det1": "x"}}}}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bench import metrics, splits, stats

SCHEMA_PATH = Path(__file__).with_name("schemas") / "predictions.schema.json"
GT_DIR = Path(__file__).with_name("gt")


class PredictionFormatError(ValueError):
    """The prediction (or GT) file is malformed; the message says where and why."""


class ScenePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1)
    count: float = Field(ge=0, allow_inf_nan=False)
    interval: tuple[float, float] | None = None
    assignments: dict[str, str] | None = Field(
        default=None, description="detection id -> predicted instance label"
    )

    @model_validator(mode="after")
    def _interval_ok(self) -> ScenePrediction:
        if self.interval is not None:
            lo, hi = self.interval
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
                raise ValueError(f"interval {self.interval} must be finite with low <= high")
        return self


class Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1)
    split: str = Field(min_length=1, description="split the scenes belong to, e.g. 'test'")
    kind: str = Field(default="countonce", description="countonce | baseline | vlm | other")
    scenes: list[ScenePrediction] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_scenes(self) -> Predictions:
        ids = [s.scene_id for s in self.scenes]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate scene_id(s): {dup}")
        return self


def write_schema(path: Path = SCHEMA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(Predictions.model_json_schema(), indent=2, sort_keys=True) + "\n")
    return path


def load_predictions(path: str | Path) -> Predictions:
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except OSError as e:
        raise PredictionFormatError(f"{p}: cannot read file ({e})") from e
    except json.JSONDecodeError as e:
        raise PredictionFormatError(f"{p}: not valid JSON ({e})") from e
    try:
        return Predictions.model_validate(raw)
    except ValidationError as e:
        lines = [
            f"  {'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}"
            for err in e.errors()
        ]
        raise PredictionFormatError(f"{p}: invalid predictions\n" + "\n".join(lines)) from e


def load_gt(gt: str | Path) -> dict[str, Any]:
    """Resolve ``gt`` as a file path or a split name under ``bench/gt/``."""
    p = Path(gt)
    if not p.is_file():
        p = GT_DIR / f"{gt}.json"
    if not p.is_file():
        raise PredictionFormatError(f"GT not found: {gt!r} (looked for a file and {p})")
    try:
        doc = json.loads(p.read_text())
        scenes = doc["scenes"]
        split = doc["split"]
        for sid, s in scenes.items():
            if not isinstance(s["count"], int | float):
                raise TypeError(f"{sid}: count must be a number")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        raise PredictionFormatError(f"{p}: malformed GT file ({e!r})") from e
    return {"split": split, "scenes": scenes}


def frozen_problems(
    split: str, scene_ids: set[str], splits_path: str | Path = splits.DEFAULT_PATH
) -> list[str]:
    """Why ``scene_ids`` do not constitute the full frozen ``split`` (empty list = OK)."""
    if not Path(splits_path).is_file():
        return [f"no frozen split file at {splits_path}; nothing is frozen yet"]
    frozen = splits.load_splits(splits_path)
    if split not in frozen:
        return [f"split {split!r} is not a frozen split (frozen: {sorted(frozen)})"]
    want = set(frozen[split])
    problems = []
    if scene_ids - want:
        problems.append(f"scenes not in frozen '{split}': {sorted(scene_ids - want)}")
    if want - scene_ids:
        problems.append(f"frozen '{split}' scenes not scored: {sorted(want - scene_ids)}")
    return problems


def evaluate(
    preds: Predictions,
    gt: dict[str, Any],
    reference: Predictions | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    splits_path: str | Path = splits.DEFAULT_PATH,
) -> dict[str, Any]:
    """Score ``preds`` against ``gt``. Every GT scene must be predicted (no silent skips)."""
    if preds.split != gt["split"]:
        raise PredictionFormatError(
            f"predictions are for split {preds.split!r} but GT is split {gt['split']!r}"
        )
    by_id = {s.scene_id: s for s in preds.scenes}
    unknown = sorted(set(by_id) - set(gt["scenes"]))
    missing = sorted(set(gt["scenes"]) - set(by_id))
    if unknown:
        raise PredictionFormatError(f"predictions for scenes not in GT: {unknown}")
    if missing:
        raise PredictionFormatError(f"no prediction for GT scenes: {missing}")

    ids = sorted(by_id)
    p = [by_id[i].count for i in ids]
    g = [float(gt["scenes"][i]["count"]) for i in ids]
    abs_err = metrics.abs_errors(p, g)
    out: dict[str, Any] = {
        "method": preds.method,
        "kind": preds.kind,
        "split": preds.split,
        "n_scenes": len(ids),
        "mae": stats.bootstrap_ci(list(abs_err), n_boot=n_boot, seed=seed).__dict__,
        "mape": metrics.mape(p, g),
        "within_10pct": metrics.within_rate(p, g, 0.10),
        "frozen_problems": frozen_problems(preds.split, set(ids), splits_path),
    }
    iv = [i for i in ids if by_id[i].interval is not None]
    if iv:
        lo = [by_id[i].interval[0] for i in iv]  # type: ignore[index]
        hi = [by_id[i].interval[1] for i in iv]  # type: ignore[index]
        out["interval_coverage"] = metrics.interval_coverage(
            lo, hi, [float(gt["scenes"][i]["count"]) for i in iv]
        )
        out["n_interval_scenes"] = len(iv)
    f1s = []
    for i in ids:
        a, gi = by_id[i].assignments, gt["scenes"][i].get("instances")
        if a is None or not gi:
            continue
        dets = sorted(set(a) & set(gi))
        if len(dets) >= 2:
            f1s.append(metrics.pairwise_prf([a[d] for d in dets], [gi[d] for d in dets])[2])
    if f1s:
        out["assoc_f1"] = stats.bootstrap_ci(f1s, n_boot=n_boot, seed=seed).__dict__
        out["n_assoc_scenes"] = len(f1s)
    if reference is not None:
        ref = {s.scene_id: s.count for s in reference.scenes}
        if set(ref) != set(ids):
            raise PredictionFormatError("reference method must cover exactly the same scenes")
        ref_err = metrics.abs_errors([ref[i] for i in ids], g)
        res = stats.paired_compare(list(abs_err), list(ref_err), n_boot=n_boot, seed=seed)
        out["vs_reference"] = {"reference": reference.method, **res.__dict__}
    return out


def _json_safe(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.generic):
        return _json_safe(o.item())
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.evaluate", description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt", required=True, help="split name (bench/gt/<split>.json) or GT file")
    ap.add_argument("--reference", help="reference method's predictions for a paired test")
    ap.add_argument("--out", help="write the result JSON here")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--write-schema", action="store_true", help="refresh the published schema")
    a = ap.parse_args(argv)
    try:
        if a.write_schema:
            write_schema()
        res = evaluate(
            load_predictions(a.predictions),
            load_gt(a.gt),
            load_predictions(a.reference) if a.reference else None,
            n_boot=a.n_boot,
        )
    except PredictionFormatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    text = json.dumps(_json_safe(res), indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
