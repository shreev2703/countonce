import csv
import json

import pytest

from bench import human_baseline as hb
from bench.runner import COLUMNS

GT = {"s1": 10, "s2": 20, "s3": 30, "s4": 40}


def _manifest():
    return hb.Manifest.model_validate(
        {"scenes": {s: {"images": [f"{s}_{i}.jpg" for i in range(4)], "count": c}
                    for s, c in GT.items()}}
    )  # fmt: skip


def _write(tmp_path, rows, name="r.csv"):
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "scene_id", "count", "seconds", "notes"])
        w.writerows(rows)
    return p


def _fixture(tmp_path):
    rows = []
    for pid, off, sec in (("p1", 0, 60), ("p2", 2, 90), ("p3", -1, 120)):
        for s, c in GT.items():
            rows.append([pid, s, c + off, sec, ""])
    return _write(tmp_path, rows)


def test_sheets_static_and_hide_gt(tmp_path):
    paths = hb.write_sheets(_manifest(), tmp_path / "out")
    assert len(paths) == 5
    page = paths[1].read_text()
    assert "<script" not in page and page.count("<img") == 4
    assert "Seconds" in page and "Count:" in page
    assert "30" not in (tmp_path / "out" / "scene_s3.html").read_text().replace("s3_", "")
    assert "timer" in paths[0].read_text()


def test_sheet_escapes_html():
    sc = hb.Scene(images=['"><script>x</script>.jpg'])
    assert "<script>" not in hb.scene_sheet_html("<b>", sc)


def test_load_ok_and_errors(tmp_path):
    r = hb.load_responses(_fixture(tmp_path), list(GT))
    assert len(r) == 12
    bad = _write(tmp_path, [["p1", "nope", 3, 10, ""]], "u.csv")
    with pytest.raises(hb.StudyError, match="unknown scene 'nope'"):
        hb.load_responses(bad, list(GT))
    dup = _write(tmp_path, [["p1", "s1", 3, 10, ""], ["p1", "s1", 4, 11, ""]], "d.csv")
    with pytest.raises(hb.StudyError, match="duplicate.*line 2"):
        hb.load_responses(dup, list(GT))
    neg = _write(tmp_path, [["p1", "s1", -3, 10, ""]], "n.csv")
    with pytest.raises(hb.StudyError, match="line 2"):
        hb.load_responses(neg, list(GT))
    nocol = tmp_path / "c.csv"
    nocol.write_text("participant_id,scene_id\np,s1\n")
    with pytest.raises(hb.StudyError, match="missing column"):
        hb.load_responses(nocol, list(GT))


def test_scoring_known_values(tmp_path):
    resp = hb.load_responses(_fixture(tmp_path), list(GT))
    res = hb.score_study(_manifest(), resp, n_boot=200)
    assert res["participants"]["p1"]["mae"]["estimate"] == 0
    assert res["participants"]["p1"]["within_10pct"]["estimate"] == 1
    assert res["participants"]["p2"]["mae"]["estimate"] == 2
    # offsets 0, +2, -1 -> pooled MAE = (0 + 2 + 1) / 3
    assert res["pooled"]["mae"]["estimate"] == pytest.approx(1.0)
    ci = res["pooled"]["mae"]
    assert ci["low"] <= ci["estimate"] <= ci["high"]
    assert res["pooled"]["mean_seconds_per_scene"] == pytest.approx(90)


def test_paired_and_time_ratio(tmp_path):
    resp = hb.load_responses(_fixture(tmp_path), list(GT))
    mpred = {"s1": 10, "s2": 20, "s3": 31, "s4": 39}
    msec = dict.fromkeys(GT, 3.0)
    res = hb.score_study(_manifest(), resp, mpred, msec, n_boot=200)
    v = res["vs_method"]
    assert v["n"] == 4 and v["method_mae"]["estimate"] == pytest.approx(0.5)
    assert v["diff"] == pytest.approx(1.0 - 0.5)
    assert res["time_ratio"]["estimate"] == pytest.approx(30.0)
    assert res["time_ratio"]["low"] == pytest.approx(30.0)


def test_method_scenes_from_bench_csv(tmp_path):
    p = tmp_path / "bench.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for s, pr, t in (("s1", 10, 2), ("s2", 22, 4)):
            w.writerow({"run_key": s, "status": "ok", "scene_id": s, "method": "m", "n_views": 5,
                        "sample": "0", "ordered": 1, "pred": pr, "gt": GT[s], "runtime_s": t,
                        "config_json": "{}"})  # fmt: skip
    pred, secs = hb.method_scenes(p, "m")
    assert pred == {"s1": 10, "s2": 22} and secs == {"s1": 2, "s2": 4}
    with pytest.raises(hb.StudyError):
        hb.method_scenes(p, "absent")


def test_cli_and_figure(tmp_path):
    pytest.importorskip("matplotlib")
    man = tmp_path / "m.json"
    man.write_text(json.dumps(_manifest().model_dump(mode="json")))
    assert hb.main(["sheets", "--manifest", str(man), "--out", str(tmp_path / "s")]) == 0
    resp = _fixture(tmp_path)
    fig = tmp_path / "f.png"
    code = hb.main(["figure", "--manifest", str(man), "--responses", str(resp),
                    "--out-fig", str(fig), "--n-boot", "100"])  # fmt: skip
    assert code == 0 and fig.exists()
