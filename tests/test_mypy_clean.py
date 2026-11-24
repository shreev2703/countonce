"""`mypy countonce` must report zero errors (PLAN section 4 rule 5)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(importlib.util.find_spec("mypy") is None, reason="mypy not installed")
def test_mypy_countonce_clean() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "countonce"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Success" in proc.stdout
