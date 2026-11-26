"""Public Python API, config resolution outside the repo, and the built wheel's contents."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from countonce.resources import resolve_config

ROOT = Path(__file__).resolve().parents[1]


def test_import_is_light_and_exports_resolve():
    code = (
        "import sys, countonce\n"
        "assert 'torch' not in sys.modules and 'countonce.pipeline' not in sys.modules\n"
        "import countonce.pipeline, countonce.config, countonce.types\n"
        "assert countonce.run is countonce.pipeline.run\n"
        "assert countonce.Config is countonce.config.Config\n"
        "assert countonce.CountResult is countonce.types.CountResult\n"
        "assert callable(countonce.load_result)\n"
        "assert 'torch' not in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stderr


def test_unknown_attribute():
    import countonce

    with pytest.raises(AttributeError):
        countonce.nope  # noqa: B018


def test_resolve_config_falls_back_to_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = resolve_config("configs/alerts.yaml")
    assert p.is_file() and p.parent.name == "configs" and "countonce" in p.parts
    local = tmp_path / "configs"
    local.mkdir()
    (local / "alerts.yaml").write_text("rules: []\n")
    assert resolve_config("configs/alerts.yaml") == Path("configs/alerts.yaml")


def test_resolve_config_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="not shipped"):
        resolve_config("configs/bench.yaml")


def test_cli_missing_dev_config_message(tmp_path, monkeypatch, capsys):
    from countonce.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["bench"]) == 2
    assert "not shipped" in capsys.readouterr().err


@pytest.mark.slow
def test_wheel_contents(tmp_path):
    has_pip = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True)
    if has_pip.returncode == 0:
        cmd = [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
               "--no-build-isolation", "-w", str(tmp_path), "-q"]  # fmt: skip
    elif shutil.which("uv"):
        cmd = ["uv", "build", "--wheel", "--no-build-isolation", "--offline",
               "--python", sys.executable, "--out-dir", str(tmp_path), str(ROOT)]  # fmt: skip
    else:
        pytest.skip("neither pip nor uv available to build a wheel")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    if r.returncode != 0:
        pytest.skip(f"offline wheel build unavailable: {r.stderr[-300:]}")
    wheel = next(tmp_path.glob("countonce-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        entry = next(n for n in names if n.endswith("entry_points.txt"))
        eps = z.read(entry).decode()
    assert "countonce/py.typed" in names
    for cfg in ("alerts", "colmap", "cost"):
        assert f"countonce/configs/{cfg}.yaml" in names
    assert "countonce/resources.py" in names
    assert "countonce = countonce.cli:main" in eps
    assert "countonce-mcp" not in eps  # quarantined in experimental/
