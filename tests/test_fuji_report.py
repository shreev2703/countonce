"""Fuji report: cells need backing rows (PLAN §9.8)."""

import json
from pathlib import Path

import pytest

from bench import fuji_report as fr
from bench.runner import COLUMNS, read_rows  # noqa: F401


def _row(tree, side, k, method, pred, gt=10.0, status="ok", f1="0.5"):
    r = {c: "" for c in COLUMNS}
    r.update(
        run_key=f"{tree}{side}{k}{method}", status=status, scene_id=tree, split="test",
        method=method, n_views=str(k), sample=side, pred=str(pred), gt=str(gt),
        config_json=json.dumps({"protocol": fr.PROTOCOL}),
        assoc_f1=f1, duplicate_rate="0.1", miss_rate="0.2",
    )  # fmt: skip
    return r


def _write(path: Path, rows):
    import csv

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _full(k=3, methods=fr.METHODS, skip=()):
    return [
        _row(t, s, k, m, 9 + i % 3)
        for i, t in enumerate(fr.TEST_TREES)
        for s in fr.SIDES
        for m in methods
        if (t, s, m) not in skip
    ]


def test_write_cell_refuses_without_rows():
    rep = fr.compute([])
    with pytest.raises(fr.MissingBackingRow):
        fr.write_cell(rep, "countonce_v2|3|mae")
    assert fr.cell_or_todo(rep, "countonce_v2|3|mae") == fr.TODO


def test_incomplete_slice_stays_todo(tmp_path):
    rows = _full(skip={("Tree08", "west", "countonce_v2")})
    rep = fr.compute(rows)
    assert "countonce_v2|3|mae" not in rep["cells"]
    assert "countonce_v1|3|mae" in rep["cells"]
    assert "countonce_v2|countonce_v1|3|mae" not in rep["paired"]


def test_filled_cells_and_paired(tmp_path):
    rep = fr.compute(_full())
    assert fr.write_cell(rep, "countonce_v2|3|mae").startswith("")
    assert "countonce_v2|sum|3|mae" in rep["paired"]
    assert rep["paired"]["countonce_v2|sum|3|mae"]["n"] == 8
    assert "countonce_v1|5|mae" not in rep["cells"]  # k=5 never ran


def test_build_with_no_rows_writes_only_todo(tmp_path):
    (tmp_path / "README.md").write_text("# x\n")
    rep = fr.build(tmp_path / "missing.csv", tmp_path)
    assert rep["cells"] == {}
    md = (tmp_path / "docs/RESULTS_FUJI.md").read_text()
    assert fr.TODO in md and "No Fuji test-split rows" in md
    readme = (tmp_path / "README.md").read_text()
    assert fr.BEGIN in readme
    fr.build(tmp_path / "missing.csv", tmp_path)  # idempotent
    assert (tmp_path / "README.md").read_text().count(fr.BEGIN) == 1


def test_build_ignores_non_fuji_and_val_rows(tmp_path):
    other = _row("Tree01", "east", 3, "countonce_v2", 1)
    other["config_json"] = "{}"
    val = _row("Tree01", "east", 3, "countonce_v2", 1)
    val["split"] = "val"
    p = tmp_path / "b.csv"
    _write(p, [other, val])
    assert fr.fuji_rows(p) == []


def test_failure_analysis_lists_run_ids():
    rows = _full(methods=("countonce_v2",))
    rows.append(_row("Tree02", "east", 3, "countonce_v2", 0, status="error"))
    txt = fr.failure_analysis(rows)
    assert "Tree02east3countonce_v2" in txt and "failed rows" in txt
