"""Paper table and project-site generators; no LaTeX needed (PLAN §23, §16B.4)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.test_bench_sweep import _synthetic_csv

ROOT = Path(__file__).parent.parent


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


tables = _load("make_paper_tables")
site = _load("make_site")
TODO = "TODO: fill from results"


def test_tables_with_data(tmp_path):
    csv_path = tmp_path / "bench.csv"
    _synthetic_csv(csv_path)
    paths = tables.render_all(csv_path, tmp_path / "t")
    assert {p.stem for p in paths} == set(tables.TABLES)
    main = (tmp_path / "t" / "main_results.tex").read_text()
    assert "co\\_v2" in main and "[" in main and "\\toprule" in main
    assert main.count(TODO) == 4  # no assoc_f1 in the synthetic csv: one honest TODO per row
    assert (
        tables.render_all(csv_path, tmp_path / "t2")
        and main == (tmp_path / "t2" / "main_results.tex").read_text()
    )  # deterministic
    assert TODO not in (tmp_path / "t" / "coverage.tex").read_text() or "sum" in main
    # synthetic methods have no ablation names, so that table is a placeholder, not an error
    assert TODO in (tmp_path / "t" / "ablation.tex").read_text()


def test_tables_without_data(tmp_path):
    paths = tables.render_all(tmp_path / "missing.csv", tmp_path / "t")
    assert len(paths) == len(tables.TABLES)
    for p in paths:
        assert TODO in p.read_text()
    assert tables.main(["--csv", str(tmp_path / "x.csv"), "--out", str(tmp_path / "o")]) == 0


def test_site_with_and_without_data(tmp_path):
    pytest.importorskip("matplotlib")
    from bench.figures import make_all

    csv_path = tmp_path / "bench.csv"
    _synthetic_csv(csv_path)
    make_all(csv_path, tmp_path / "figs")
    out = tmp_path / "site" / "index.html"
    rc = site.main(
        ["--csv", str(csv_path), "--figures", str(tmp_path / "figs"),
         "--readme", str(ROOT / "README.md"), "--out", str(out)]
    )  # fmt: skip
    page = out.read_text()
    assert rc == 0 and "data:image/png;base64," in page
    assert "<table>" in page and "Static scenes only" in page
    assert "http://" not in page and "https://" not in page  # self-contained
    assert "Failure gallery" in page

    empty = site.build(tmp_path / "no.csv", tmp_path / "nofigs", tmp_path / "NOREADME.md")
    assert empty.count(TODO) >= 4 and "base64" not in empty
