"""The CI ``full-cpu`` job must install every extra its tests importorskip."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_skips import ALLOWED  # noqa: E402

# importorskip module -> pyproject extra(s) that provide it (any one suffices)
MODULE_EXTRAS: dict[str, tuple[str, ...]] = {
    "dvc": ("dvc",),
    "boto3": ("storage", "test"),
    "moto": ("test",),
    "pyarrow": ("data",),
    "duckdb": ("data",),
    "matplotlib": ("test",),
    "PIL": ("test",),
    "safetensors": ("test",),
    "numpy": ("test", "dev"),
    "sqlalchemy": ("db",),
    "alembic": ("db",),
    "psycopg": ("db",),
    "yaml": ("test",),
    "fastapi": ("api",),
    "httpx": ("api",),
    "scipy": ("test",),
    "cv2": ("test",),
}
BASE = {"torch"}  # installed by its own step in full-cpu


def _full_cpu_extras() -> set[str]:
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    for step in wf["jobs"]["full-cpu"]["steps"]:
        m = re.search(r"pip install -e \.\[([^\]]+)\]", step.get("run", ""))
        if m:
            return {e.strip() for e in m.group(1).split(",")}
    raise AssertionError("full-cpu has no `pip install -e .[...]` step")


def _skip_modules() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in sorted((ROOT / "tests").glob("*.py")):
        for mod in re.findall(r'importorskip\(\s*"([\w.]+)"', path.read_text()):
            out.setdefault(mod.split(".")[0], set()).add(path.name)
    return out


def test_full_cpu_installs_every_extra_that_importorskip_needs() -> None:
    have = _full_cpu_extras()
    for mod, files in _skip_modules().items():
        if mod in BASE:
            continue
        if any(a in mod.lower() for a in ALLOWED):  # allow-listed skips may stay skipped
            continue
        assert mod in MODULE_EXTRAS, f"map {mod!r} (used in {sorted(files)}) to an extra"
        assert have & set(MODULE_EXTRAS[mod]), (
            f"{mod!r} (importorskip in {sorted(files)}) needs one of "
            f"{MODULE_EXTRAS[mod]} in full-cpu, which installs {sorted(have)}"
        )


def test_full_cpu_has_postgres_service_and_url() -> None:
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = wf["jobs"]["full-cpu"]
    assert job["services"]["postgres"]["image"] == "postgres:16"
    assert "--health-cmd" in job["services"]["postgres"]["options"]
    assert job["env"]["COUNTONCE_PG_URL"].startswith("postgresql://")
