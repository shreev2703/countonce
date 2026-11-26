"""Repo hygiene (PLAN §28.5): no agent files or personal paths are tracked."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_FILES = {
    "TASKS_OVERNIGHT.md",
    "scripts/hud.py",
    "scripts/dashboard.sh",
    "scripts/morning_review.sh",
    "scripts/overnight_orchestrator.sh",
}


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False)
    return out.stdout.split()


def test_license_is_apache() -> None:
    assert "Apache License" in (ROOT / "LICENSE").read_text()


def test_no_agent_files_tracked() -> None:
    assert not AGENT_FILES & set(_tracked())


@pytest.mark.xfail(reason="untracking CLAUDE.md needs a human commit (docs/GATES.md, M21)")
def test_agent_rules_file_untracked() -> None:
    assert "CLAUDE.md" not in _tracked()


def test_no_personal_paths_in_tracked_text() -> None:
    needles = ["/home/" + "shreev", "super" + "_nga"]
    bad = []
    for name in _tracked():
        if name == "CLAUDE.md":  # pending, see the xfail below
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if any(n in text for n in needles):
            bad.append(name)
    assert not bad, bad
