"""Sites and visits: ordering, interval-overlap flagging, typed errors, JSON round-trip."""

import json
import shutil
from pathlib import Path

import pytest

from countonce import sites
from countonce.cli import main

FIX = Path(__file__).parent / "fixtures" / "schemas" / "v1.1" / "result.json"


def _run(tmp_path: Path, name: str, count: int, low: int, high: int, ts: str | None, report=False):
    d = tmp_path / name
    d.mkdir()
    data = json.loads(FIX.read_text())
    data["count"] = count
    data["interval"].update(low=low, high=high)
    if ts is None:
        data["config"].pop("timestamp_utc", None)
    else:
        data["config"]["timestamp_utc"] = ts
    (d / "result.json").write_text(json.dumps(data))
    if report:
        (d / "report.html").write_text("<html></html>")
    return d


def _index(tmp_path):
    idx = sites.SiteIndex(tmp_path / "sites.json")
    idx.add("Row 4 apples", "north block")
    return idx


def test_ordering_by_captured_at_and_report_links(tmp_path):
    idx = _index(tmp_path)
    late = _run(tmp_path, "late", 30, 28, 32, "2026-03-01T00:00:00+00:00", report=True)
    early = _run(tmp_path, "early", 10, 9, 11, "2026-01-01T00:00:00+00:00")
    idx.attach("row-4-apples", late)
    idx.attach("row-4-apples", early)
    rows = sites.visit_rows(idx.get("row-4-apples"))
    assert [r.count for r in rows] == [10, 30]
    assert rows[0].report_html is None
    assert rows[1].report_html == str(late.resolve() / "report.html")


def test_interval_overlap_flagging(tmp_path):
    rows = [
        sites.VisitRow("t1", "a", 10, 8, 12, 0.9, None),
        sites.VisitRow("t2", "b", 13, 12, 15, 0.9, None),  # touches at 12: overlap
        sites.VisitRow("t3", "c", 20, 18, 22, 0.9, None),  # disjoint
    ]
    ds = sites.deltas(rows)
    assert [(d.change, d.within_noise) for d in ds] == [(3, True), (7, False)]
    assert "(within noise)" in sites.render_table(sites.Site("s", "S"), rows, ds)


def test_missing_run_dir_raises_typed_error(tmp_path):
    idx = _index(tmp_path)
    with pytest.raises(sites.RunDirError):
        idx.attach("row-4-apples", tmp_path / "nope", "2026-01-01T00:00:00")
    d = _run(tmp_path, "r", 1, 1, 1, "2026-01-01T00:00:00+00:00")
    idx.attach("row-4-apples", d)
    shutil.rmtree(d)
    with pytest.raises(sites.RunDirError):
        sites.visit_rows(idx.get("row-4-apples"))


def test_other_typed_errors(tmp_path):
    idx = _index(tmp_path)
    with pytest.raises(sites.DuplicateSiteError):
        idx.add("Row 4 apples")
    with pytest.raises(sites.SiteNotFoundError):
        idx.get("missing")
    d = _run(tmp_path, "nots", 1, 1, 1, None)
    with pytest.raises(sites.CapturedAtError):
        idx.attach("row-4-apples", d)
    idx.attach("row-4-apples", d, "2026-02-01T00:00:00")  # explicit time wins


def test_json_round_trip(tmp_path):
    idx = _index(tmp_path)
    idx.attach("row-4-apples", _run(tmp_path, "a", 1, 1, 2, "2026-01-01T00:00:00+00:00"))
    idx.save()
    again = sites.SiteIndex.load(tmp_path / "sites.json")
    assert again.to_dict() == idx.to_dict()
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(sites.SiteError):
        sites.SiteIndex.load(tmp_path / "bad.json")


def test_cli_flow_and_report(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    f = str(tmp_path / "sites.json")
    a = _run(tmp_path, "a", 10, 9, 11, "2026-01-01T00:00:00+00:00")
    b = _run(tmp_path, "b", 20, 18, 22, "2026-02-01T00:00:00+00:00")
    assert main(["sites", "--file", f, "add", "Plot A"]) == 0
    assert main(["sites", "--file", f, "attach", "plot-a", str(a)]) == 0
    assert main(["sites", "--file", f, "attach", "plot-a", str(b)]) == 0
    assert main(["sites", "--file", f, "list"]) == 0
    assert "2 visits" in capsys.readouterr().out
    out = tmp_path / "out"
    assert main(["sites", "--file", f, "report", "plot-a", "--out", str(out)]) == 0
    assert "+10" in (out / "plot-a_report.md").read_text()
    assert (out / "plot-a_counts.png").stat().st_size > 0
    assert main(["sites", "--file", f, "report", "nope"]) == 2
