import json

import pytest

from bench import evaluate, leaderboard, splits


def _gt(split="test"):
    return {
        "split": split,
        "scenes": {
            "a": {"count": 10, "instances": {"d0": "x", "d1": "x", "d2": "y", "d3": "y"}},
            "b": {"count": 20},
            "c": {"count": 30},
        },
    }


def _preds(method="m", split="test", counts=(12, 20, 27), **extra):
    scenes = [{"scene_id": s, "count": c} for s, c in zip("abc", counts, strict=True)]
    scenes[0].update(extra)
    return evaluate.Predictions.model_validate({"method": method, "split": split, "scenes": scenes})


def _freeze(tmp_path, test=("a", "b", "c")):
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"version": 1, "splits": {"test": {s: "h" for s in test}, "val": {}}}))
    return p


def test_known_mae_and_f1(tmp_path):
    preds = _preds(assignments={"d0": "p", "d1": "p", "d2": "q", "d3": "q"}, interval=[9, 13])
    r = evaluate.evaluate(preds, _gt(), n_boot=200, splits_path=_freeze(tmp_path))
    assert r["mae"]["estimate"] == pytest.approx((2 + 0 + 3) / 3)
    assert r["assoc_f1"]["estimate"] == pytest.approx(1.0)
    assert r["interval_coverage"] == 1.0
    assert r["frozen_problems"] == []


def test_f1_partial(tmp_path):
    preds = _preds(assignments={"d0": "p", "d1": "p", "d2": "p", "d3": "q"})
    r = evaluate.evaluate(preds, _gt(), n_boot=50, splits_path=_freeze(tmp_path))
    # pred pairs 3, true pairs 2, tp 1 -> P=1/3, R=1/2, F1=0.4
    assert r["assoc_f1"]["estimate"] == pytest.approx(0.4)


def test_paired_reference(tmp_path):
    ref = _preds("ref", counts=(15, 25, 20))
    r = evaluate.evaluate(_preds(), _gt(), ref, n_boot=200, splits_path=_freeze(tmp_path))
    assert r["vs_reference"]["diff"] < 0
    assert r["vs_reference"]["reference"] == "ref"


@pytest.mark.parametrize(
    "doc,msg",
    [
        ({"method": "m", "split": "test", "scenes": [{"scene_id": "a", "count": -1}]}, "count"),
        ({"method": "m", "split": "test", "scenes": []}, "scenes"),
        ({"method": "m", "scenes": [{"scene_id": "a", "count": 1}]}, "split"),
        (
            {
                "method": "m",
                "split": "t",
                "scenes": [{"scene_id": "a", "count": 1, "interval": [3, 1]}],
            },
            "low <= high",
        ),
        (
            {"method": "m", "split": "t", "scenes": [{"scene_id": "a", "count": 1}] * 2},
            "duplicate",
        ),
    ],
)
def test_malformed_rejected(tmp_path, doc, msg):
    f = tmp_path / "p.json"
    f.write_text(json.dumps(doc))
    with pytest.raises(evaluate.PredictionFormatError, match=msg):
        evaluate.load_predictions(f)


def test_not_json_and_cli_error(tmp_path, capsys):
    f = tmp_path / "p.json"
    f.write_text("{nope")
    assert evaluate.main(["--predictions", str(f), "--gt", "x"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_missing_scene_and_split_mismatch():
    with pytest.raises(evaluate.PredictionFormatError, match="split"):
        evaluate.evaluate(_preds(split="val"), _gt())
    short = evaluate.Predictions.model_validate(
        {"method": "m", "split": "test", "scenes": [{"scene_id": "a", "count": 1}]}
    )
    with pytest.raises(evaluate.PredictionFormatError, match="no prediction"):
        evaluate.evaluate(short, _gt())


def test_published_schema_current():
    assert json.loads(evaluate.SCHEMA_PATH.read_text()) == evaluate.Predictions.model_json_schema()


def _write_dir(tmp_path, preds_list):
    d = tmp_path / "preds"
    d.mkdir()
    for p in preds_list:
        (d / f"{p.method}.json").write_text(p.model_dump_json())
    g = tmp_path / "gt.json"
    g.write_text(json.dumps(_gt()))
    return d, g


def test_leaderboard_renders_frozen(tmp_path):
    d, g = _write_dir(tmp_path, [_preds("ref", counts=(15, 25, 20)), _preds("good")])
    rows = leaderboard.build_rows(d, g, "ref", _freeze(tmp_path), n_boot=100)
    md = leaderboard.render(rows)
    assert [r["method"] for r in rows] == ["good", "ref"]
    assert "| 1 | good |" in md


def test_leaderboard_refuses_non_frozen(tmp_path):
    d, g = _write_dir(tmp_path, [_preds("m")])
    with pytest.raises(leaderboard.LeaderboardError, match="not frozen|nothing is frozen"):
        leaderboard.build_rows(d, g, splits_path=tmp_path / "missing.json")
    with pytest.raises(leaderboard.LeaderboardError, match="not scored"):
        leaderboard.build_rows(d, g, splits_path=_freeze(tmp_path, ("a", "b", "c", "z")))
    d2, g2 = tmp_path / "d2", tmp_path / "gt2.json"
    d2.mkdir()
    (d2 / "m.json").write_text(_preds("m", split="val").model_dump_json())
    g2.write_text(json.dumps(_gt("val")))
    with pytest.raises(leaderboard.LeaderboardError, match="not a frozen split"):
        only_test = tmp_path / "only_test.json"
        only_test.write_text(json.dumps({"version": 1, "splits": {"test": {"a": "h"}}}))
        leaderboard.build_rows(d2, g2, splits_path=only_test)
    assert splits.DEFAULT_PATH  # real path untouched
