"""The README front page (PLAN §28.6b): no placeholders, numbers and links that exist."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from countonce.config import Config

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
METHODS = {
    "CountOnce v1": "countonce_v1",
    "CountOnce v2": "countonce_v2",
    "Sum of per-photo counts": "sum",
    "Max per-photo count": "max",
    "Median per-photo count": "median",
}


def test_no_placeholders() -> None:
    assert "TODO" not in README


def test_relative_links_resolve() -> None:
    targets = re.findall(r"\]\((?!https?:|#)([^)#]+)", README)
    assert targets
    assert [t for t in targets if not (ROOT / t).exists()] == []


def test_fuji_table_matches_committed_summary() -> None:
    with (ROOT / "web/content/benchmark/benchmark_summary.csv").open() as fh:
        mae = {(r["method"], r["k"]): r for r in csv.DictReader(fh) if r["metric"] == "mae"}
    for label, method in METHODS.items():
        cells = [
            "{:.1f} [{:.1f}, {:.1f}]".format(
                *(float(mae[method, k][c]) for c in ("estimate", "low", "high"))
            )
            for k in ("3", "5", "10", "20")
        ]
        assert f"| {label} | {' | '.join(cells)} |" in README


def test_readme_configs_load() -> None:
    assert Config.from_yaml(ROOT / "examples/tiny_scene/all_views.yaml").dedup_threshold == 0
    assert Config.from_yaml(ROOT / "configs/readme_real_failure.yaml").backbone == "vggt"
