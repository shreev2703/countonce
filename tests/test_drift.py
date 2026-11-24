import json

import numpy as np
import pytest

from countonce.cli import main
from countonce.monitor.drift import (
    DriftThresholds,
    categorical_drift,
    compute_drift,
    load_rows,
    numeric_psi,
    row_from_result,
)


def _rows(rng, n, shift=0.0, prompt="apples"):
    return [
        {
            "n_frames": float(rng.integers(8, 16)) + shift * 20,
            "mean_depth_conf": float(rng.normal(0.8 + shift, 0.05)),
            "detection_density": float(rng.normal(5 + shift * 20, 1)),
            "lift_failure_rate": float(rng.uniform(0, 0.2)),
            "resolution_mpix": 1.0,
            "prompt": prompt,
        }
        for _ in range(n)
    ]


def test_identical_distributions_no_flags():
    rows = _rows(np.random.default_rng(0), 300)
    rep = compute_drift(rows, rows)
    assert rep["status"] == "ok" and rep["flagged"] == {}
    assert all(v["psi"] < 1e-6 for v in rep["features"].values())
    assert numeric_psi([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0, abs=1e-9)


def test_same_population_resample_small_psi():
    a, b = _rows(np.random.default_rng(1), 500), _rows(np.random.default_rng(2), 500)
    assert compute_drift(a, b)["features"]["mean_depth_conf"]["psi"] < 0.1


def test_shifted_distribution_flagged():
    ref = _rows(np.random.default_rng(0), 300)
    cur = _rows(np.random.default_rng(1), 300, shift=0.3, prompt="pears")
    rep = compute_drift(ref, cur)
    assert rep["status"] == "alert"
    assert rep["flagged"]["mean_depth_conf"] == "alert"
    assert rep["flagged"]["prompt"] == "alert"
    assert rep["features"]["prompt"]["chi2"] > 0
    assert "lift_failure_rate" not in rep["flagged"]


def test_thresholds_configurable_and_validated():
    ref = list(np.linspace(0, 1, 200))
    cur = list(np.linspace(0, 1, 200) + 0.15)
    psi = numeric_psi(ref, cur)
    assert DriftThresholds(warn=psi + 1, alert=psi + 2).level(psi) == "ok"
    assert DriftThresholds(warn=0.0, alert=psi + 1).level(psi) == "warn"
    with pytest.raises(ValueError):
        DriftThresholds(warn=0.5, alert=0.1)


def test_constant_reference_and_unseen_category_finite():
    assert np.isfinite(numeric_psi([1.0] * 20, [2.0] * 20))
    assert np.isfinite(categorical_drift(["a"] * 5, ["b"] * 5)["psi"])


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_rows_clear_error(n):
    rows = _rows(np.random.default_rng(0), 10)
    with pytest.raises(ValueError, match="at least 2 current rows"):
        compute_drift(rows, rows[:n])
    with pytest.raises(ValueError, match="at least 2 reference rows"):
        compute_drift(rows[:n], rows)


def test_no_shared_feature_error():
    with pytest.raises(ValueError, match="no drift feature"):
        compute_drift([{"prompt": None}] * 3, [{"prompt": None}] * 3)


def _result(n_frames, n_det, fallbacks, prompt, depth=None):
    return {
        "prompt": prompt,
        "per_frame_counts": {str(i): 1 for i in range(n_frames)},
        "detection_status": {f"0:{i}": "kept" for i in range(n_det)},
        "config": {"lift_fallbacks": fallbacks},
        "warnings": [
            {"code": "low_depth_confidence", "message": f"median depth confidence {depth:.2f})"}
        ]
        if depth
        else [],
    }


def test_row_from_result_and_cli(tmp_path, capsys):
    r = row_from_result(_result(4, 20, 5, "widgets", depth=0.4))
    assert r["n_frames"] == 4 and r["detection_density"] == 5
    assert r["lift_failure_rate"] == 0.25 and r["mean_depth_conf"] == 0.4

    ref, cur = tmp_path / "ref", tmp_path / "cur"
    for d, nf, p in ((ref, 4, "widgets"), (cur, 40, "gadgets")):
        for i in range(12):
            (d / str(i)).mkdir(parents=True)
            (d / str(i) / "result.json").write_text(json.dumps(_result(nf + i % 3, 20, 2, p)))
    out = tmp_path / "drift.json"
    assert main(["drift", "--reference", str(ref), "--current", str(cur), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["status"] == "alert" and "n_frames" in rep["flagged"]
    assert "status alert" in capsys.readouterr().out


def test_csv_reference_and_cli_error(tmp_path, capsys):
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text("scene_id,n_views,density,prompt\ns1,8,3.0,apples\ns2,12,4.0,apples\n")
    assert len(load_rows(csv_path)) == 2
    cur = tmp_path / "cur"
    cur.mkdir()
    assert main(["drift", "--reference", str(csv_path), "--current", str(cur)]) == 2
    assert "at least 2 current rows" in capsys.readouterr().err
