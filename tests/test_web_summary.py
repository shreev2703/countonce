import csv

from bench import web_summary as ws
from bench.fuji_report import PROTOCOL, SIDES, TEST_TREES
from bench.runner import COLUMNS


def _rows(method: str, pred: int) -> list[dict[str, str]]:
    return [
        {
            "status": "ok", "method": method, "n_views": "3", "scene_id": t, "sample": s,
            "pred": str(pred), "gt": "10", "split": "test", "git_commit": "a" * 40,
            "git_dirty": "True", "timestamp_utc": "2026-11-25T10:00:00+00:00",
            "config_json": f'{{"protocol": "{PROTOCOL}"}}',
            "assoc_f1": "", "duplicate_rate": "", "miss_rate": "",
        }
        for t in TEST_TREES
        for s in SIDES
    ]  # fmt: skip


def test_summary_aggregates_and_counts_slice_wins():
    rows = _rows("countonce_v2", 12) + _rows("sum", 15)
    out = ws.summarise(rows)
    by = {(r["method"], r["metric"]): r for r in out}
    assert by[("countonce_v2", "mae")]["estimate"] == 2.0
    assert by[("sum", "signed_bias")]["estimate"] == 5.0
    wins = by[("countonce_v2", "slices_won_vs_sum")]
    assert (wins["estimate"], wins["n"]) == (16, 16)
    assert by[("sum", "mae")]["git_dirty"] == "true"
    assert ("countonce_v1", "mae") not in by  # no rows, no cell


def test_write_removes_stale_summary_when_there_are_no_rows(tmp_path):
    out = tmp_path / "benchmark_summary.csv"
    out.write_text("old")
    assert ws.write(tmp_path / "missing.csv", out) == 0
    assert not out.exists()


def test_write_round_trips(tmp_path):
    src = tmp_path / "bench.csv"
    rows = _rows("max", 8)
    with src.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "s.csv"
    assert ws.write(src, out) == 3  # mae, signed_bias, mape at k=3
    with out.open() as fh:
        assert [r["metric"] for r in csv.DictReader(fh)] == list(ws.WEB_METRICS)
