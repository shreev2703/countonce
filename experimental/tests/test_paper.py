"""Paper draft checks (quarantined, PLAN §28.5): run with ``pytest experimental``."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
PAPER = ROOT / "experimental" / "paper"

spec = importlib.util.spec_from_file_location(
    "make_paper_tables", ROOT / "scripts" / "make_paper_tables.py"
)
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def test_main_tex_inputs_only_generated_files_and_cites_resolve():
    tex = (PAPER / "main.tex").read_text()
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    assert inputs
    produced = {f"tables/{n}" for n in tables.TABLES}
    assert set(inputs) <= produced
    for sec in ("Introduction", "Related Work", "Method", "Synthetic Data", "Experiments",
                "Ablations", "Limitations"):  # fmt: skip
        assert f"\\section{{{sec}}}" in tex
    bib = (PAPER / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for c in re.findall(r"\\cite\{([^}]+)\}", tex) for k in c.split(",")}
    assert cited and cited <= keys


def test_refs_cover_arxiv_ids_in_docs():
    bib = (PAPER / "refs.bib").read_text()
    text = (ROOT / "docs" / "RELATED_WORK.md").read_text() + (
        ROOT / "docs" / "MODEL_SURVEY.md"
    ).read_text()
    named = {  # arXiv ids of the works the paper actually cites
        "2407.04619", "2512.23351", "2506.15368", "2608.06236", "2603.00912", "2305.02398",
        "2507.14095", "2610.38926", "2605.15195", "2602.23361",
    }  # fmt: skip
    for i in named:
        assert i in text, f"{i} no longer in docs"
        assert i in bib
