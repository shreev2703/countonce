"""End-to-end: drive the real ``countonce run`` command on the bundled tiny scene (PLAN 18).

No fakes are injected: the subprocess uses the CLI's own backend selection with the test-only
``color_blob`` detector and ``fixture`` backbone, so it runs on CPU with no weights.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from countonce.schemas import load_markers, load_result

SCENE = Path(__file__).resolve().parents[1] / "examples" / "tiny_scene"


def _cli(out: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "countonce.cli", "run", str(SCENE), "--out", str(out), *args],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize("assoc", ["v1", "v2"])
def test_cli_counts_tiny_scene(tmp_path: Path, assoc: str) -> None:
    gt = json.loads((SCENE / "scene.json").read_text())["gt_count"]
    out = tmp_path / "out"
    proc = _cli(
        out, "--prompt", "red sphere", "--detector", "color_blob", "--backbone", "fixture",
        "--assoc", assoc,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert f"red sphere: {gt}" in proc.stdout

    result = load_result(json.loads((out / "result.json").read_text()))  # schema validation
    assert result.count == gt
    assert result.model_versions.detector.startswith("color_blob")
    markers = load_markers(json.loads((out / "markers.json").read_text()))
    assert len(markers.markers) == gt
    with (out / "counts.csv").open() as fh:
        assert len(list(csv.reader(fh))) >= 2
    overlays = sorted((out / "overlays").iterdir())
    assert len(overlays) >= 1 and all(p.stat().st_size > 0 for p in overlays)


def test_cli_other_colour_counts_other_class(tmp_path: Path) -> None:
    proc = _cli(
        tmp_path / "o", "--prompt", "blue sphere", "--detector", "color_blob",
        "--backbone", "fixture", "--assoc", "v1",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert "blue sphere: 1" in proc.stdout


def test_cli_prompt_without_colour_is_a_clean_error(tmp_path: Path) -> None:
    proc = _cli(
        tmp_path / "o", "--prompt", "widget", "--detector", "color_blob", "--backbone", "fixture"
    )
    assert proc.returncode == 2
    assert "colour word" in proc.stderr
