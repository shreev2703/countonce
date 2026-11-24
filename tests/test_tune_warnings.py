import csv
import json

import numpy as np
import pytest
import yaml

from bench.runner import COLUMNS
from bench.splits import write_splits
from bench.tune_warnings import SplitViolation, load_rows, tune, tune_from_csv
from countonce.config import Config
from countonce.warnings import WarningThresholds, assess_quality


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def _synthetic(n=60, bad_views=3, seed=0):
    """Planted bad slice: scenes with n_views <= bad_views have large error."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        nv = int(rng.integers(2, 12))
        gt = 30
        err = int(rng.integers(8, 14)) if nv <= bad_views else int(rng.integers(0, 2))
        out.append({"scene_id": f"s{i}", "n_views": nv, "pred": gt + err, "gt": gt,
                    "low": gt + err - 1, "high": gt + err + 1, "status": "ok",
                    "method": "m", "split": "val", "domain": "synthetic"})  # fmt: skip
    return out


@pytest.fixture
def splits(tmp_path):
    p = tmp_path / "splits.json"
    write_splits({"t0": "a", "t1": "b", "v0": "c", "v1": "d"}, p)
    return p


def test_planted_bad_slice_found(tmp_path, splits):
    csv_path = tmp_path / "b.csv"
    _write_csv(csv_path, _synthetic())
    rows = load_rows(csv_path, method="m", splits_path=splits)
    doc = tune(rows, csv_sha256="x", splits_path=splits)
    t = doc["thresholds"]["min_frames"]
    assert t["source"] == "tuned"
    assert t["value"] == 4  # cut just above the planted boundary (n_views <= 3)
    assert t["n"] >= 5
    assert doc["warnings"]["min_frames"] == t["value"]


def test_small_n_keeps_defaults(tmp_path, splits):
    csv_path = tmp_path / "b.csv"
    _write_csv(csv_path, _synthetic(n=8))
    doc = tune_from_csv(csv_path, tmp_path / "w.yaml", splits_path=splits, method="m")
    assert doc["warnings"] == {}
    assert doc["thresholds"]["min_frames"]["source"] == "default"
    assert doc["thresholds"]["min_frames"]["value"] == WarningThresholds().min_frames


def test_wide_range_threshold(tmp_path, splits):
    rows = []
    for i in range(40):
        wide = i < 15
        gt = 20
        pred = gt + (10 if wide else 0)
        low, high = (pred - 8, pred + 8) if wide else (pred - 1, pred + 1)
        rows.append(
            dict(scene_id=f"s{i}", n_views=8, pred=pred, gt=gt, low=low, high=high,
                 status="ok", method="m", domain="synthetic")
        )  # fmt: skip
    p = tmp_path / "b.csv"
    _write_csv(p, rows)
    doc = tune(load_rows(p, method="m", splits_path=splits), splits_path=splits)
    e = doc["thresholds"]["max_range_width_fraction"]
    assert e["source"] == "tuned" and 0.04 < e["value"] < 0.5


def test_test_split_never_used(tmp_path, splits):
    from bench.splits import load_splits

    test_id = next(iter(load_splits(splits)["test"]))
    rows = _synthetic(n=30)
    rows += [dict(r, scene_id=test_id, domain="real", split="test") for r in rows[:5]]
    p = tmp_path / "b.csv"
    _write_csv(p, rows)
    loaded = load_rows(p, method="m", splits_path=splits)
    assert all(r.scene_id != test_id for r in loaded)
    with pytest.raises(SplitViolation):
        load_rows(p, method="m", splits_path=splits, split="test")
    from bench.tune_warnings import TuneRow

    with pytest.raises(SplitViolation):
        tune([TuneRow(test_id, 5, 1, 1, 1, 1)], splits_path=splits)


def test_yaml_has_hash_and_n(tmp_path, splits):
    p = tmp_path / "b.csv"
    _write_csv(p, _synthetic())
    out = tmp_path / "w.yaml"
    tune_from_csv(p, out, splits_path=splits, method="m")
    d = yaml.safe_load(out.read_text())
    assert len(d["source_csv_sha256"]) == 64
    assert all("n" in v for v in d["thresholds"].values())
    Config(warnings=d["warnings"])  # round-trips into Config


def test_config_overrides_reach_assess_quality():
    cfg = Config(warnings={"min_frames": 6})
    th = cfg.warning_thresholds()
    assert th.min_frames == 6 and isinstance(th.min_frames, int)
    codes = [w["code"] for w in assess_quality(n_frames=5, count=3, thresholds=th)]
    assert "too_few_frames" in codes
    assert assess_quality(n_frames=5, count=3) == []
    with pytest.raises(ValueError):
        Config(warnings={"nope": 1})
    assert json.loads(json.dumps(cfg.to_dict()))["warnings"] == {"min_frames": 6}
