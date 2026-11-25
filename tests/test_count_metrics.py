"""Count-first metrics on toy inputs (PLAN §9.4, task M3)."""

import math

import pytest

from bench import evaluate, figures, metrics


def test_signed_bias_sign_and_cancellation():
    assert metrics.signed_bias([12, 8], [10, 10]) == 0.0  # errors cancel, MAE does not
    assert metrics.mae([12, 8], [10, 10]) == 2.0
    assert metrics.signed_bias([12, 14], [10, 10]) == 3.0  # over-count
    assert metrics.signed_bias([5], [10]) == -5.0  # under-count


def test_count_summary_leads_with_counts():
    s = metrics.count_summary([0, 3, 10], [0, 0, 10])
    assert next(iter(s)) == "mae"
    assert s["zero_scene_false_count"] == 1.5 and s["zero_scene_false_positive_rate"] == 0.5
    assert s["signed_bias"] == 1.0 and s["within_10pct"] == pytest.approx(2 / 3)
    assert "assoc_f1" not in s
    assert "interval_coverage" in metrics.count_summary([1], [1], [0], [2])


def test_zero_scene_nan_without_zero_scenes():
    assert math.isnan(metrics.count_summary([2], [2])["zero_scene_false_count"])


def _decomp(clusters, gt_ids, universe):
    d = metrics.count_decomposition(clusters, gt_ids, universe)
    assert metrics.decomposition_error(d) == len(set(clusters)) - len(set(universe))
    return d


def test_decomposition_perfect():
    d = _decomp(["a", "b"], [1, 2], [1, 2])
    assert d == dict.fromkeys(metrics.DECOMPOSITION_KEYS, 0)


def test_decomposition_duplicate():
    d = _decomp(["a", "b", "c"], [1, 1, 2], [1, 2])
    assert d["duplicates"] == 1 and metrics.decomposition_error(d) == 1


def test_decomposition_false_positive_and_missed():
    # cluster "z" is a non-target; object 3 never detected
    d = _decomp(["a", "z"], [1, None], [1, 3])
    assert (d["false_positives"], d["missed"]) == (1, 1)
    assert metrics.decomposition_error(d) == 0


def test_decomposition_wrong_merge():
    # objects 1 and 2 (one detection each... 2 has fewer) merged into one cluster
    d = _decomp(["a", "a", "a"], [1, 1, 2], [1, 2])
    assert d["wrong_merges"] == 1 and metrics.decomposition_error(d) == -1


def test_decomposition_mixed_identity():
    clusters = ["a", "a", "b", "c", "d", "e"]
    gt = [1, 2, 2, 2, None, 4]
    d = _decomp(clusters, gt, [1, 2, 3, 4])
    # a->1 (tie 1/2 -> repr smallest), b->2, c->2 (dup), d FP, e->4; 3 never seen
    assert d == {"missed": 1, "duplicates": 1, "false_positives": 1, "wrong_merges": 0}
    assert metrics.decomposition_error(d) == 1  # 5 clusters - 4 objects


def test_decomposition_rejects_unknown_ids():
    with pytest.raises(ValueError):
        metrics.count_decomposition(["a"], [9], [1])
    with pytest.raises(ValueError):
        metrics.count_decomposition(["a"], [1, 2], [1, 2])


def _evaluate():
    gt = {
        "split": "test",
        "scenes": {
            "a": {"count": 2, "instances": {"d0": 1, "d1": 1, "d2": 2}},
            "z": {"count": 0},
        },
    }
    preds = evaluate.Predictions.model_validate(
        {
            "method": "m",
            "split": "test",
            "scenes": [
                {"scene_id": "a", "count": 3, "assignments": {"d0": "p", "d1": "q", "d2": "r"}},
                {"scene_id": "z", "count": 2},
            ],
        }
    )
    return evaluate.evaluate(preds, gt, n_boot=50, splits_path="/nonexistent")


def test_evaluate_reports_count_metrics_first(tmp_path):
    r = _evaluate()
    keys = list(r)
    for count_key in ("mae", "signed_bias", "mape", "within_10pct", "zero_scene"):
        assert keys.index(count_key) < keys.index("assoc_f1")
    assert r["signed_bias"]["estimate"] == pytest.approx(1.5)
    assert r["zero_scene"]["false_count"] == 2.0
    assert r["decomposition"]["duplicates"] == 1.0  # labelled scenes only
    assert "assoc_f1_explanatory" in r


def test_decomposition_figure(tmp_path):
    out = figures.fig_count_decomposition([_evaluate()], tmp_path / "d.png")
    assert out is not None and out.stat().st_size > 0
    assert figures.fig_count_decomposition([{"method": "x"}], tmp_path / "e.png") is None
