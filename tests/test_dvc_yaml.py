from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from synth.dag import STAGE_GRAPH, STAGES, DagConfig, dvc_lock_hash

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "make_dvc_yaml", ROOT / "scripts" / "make_dvc_yaml.py"
)
mk = importlib.util.module_from_spec(_spec)
sys.modules["make_dvc_yaml"] = mk
_spec.loader.exec_module(mk)


def test_committed_files_are_byte_identical_to_regenerated():
    for name, text in mk.render().items():
        assert (ROOT / name).read_text() == text, f"{name} drifted; run make dvc-yaml"


def test_regeneration_is_deterministic():
    assert mk.render() == mk.render()


def test_graph_covers_every_stage_and_known_params():
    assert set(STAGE_GRAPH) == set(STAGES)
    known = {f for f in DagConfig.__dataclass_fields__}
    for spec in STAGE_GRAPH.values():
        assert {p.split(".")[0] for p in spec["params"]} <= known
        assert all(d in STAGES or "/" in d for d in spec["deps"])


def test_every_stage_dep_and_out_in_dvc_yaml():
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
    work = yaml.safe_load((ROOT / "params.yaml").read_text())["work_dir"]
    assert list(dvc) == list(STAGES)
    for st, spec in STAGE_GRAPH.items():
        entry = dvc[st]
        assert f"--only {st}" in entry["cmd"]
        for d in spec["deps"]:
            assert (f"{work}/{d}" if d in STAGES else d) in entry["deps"]
        assert entry.get("params", []) == list(spec["params"])
        outs = [o if isinstance(o, str) else next(iter(o)) for o in entry["outs"]]
        assert f"{work}/{st}" in outs
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    for spec in STAGE_GRAPH.values():
        assert all(p.split(".")[0] in params for p in spec["params"])


def test_no_remote_configured():
    assert (
        not (ROOT / ".dvc" / "config").exists()
        or "remote" not in (ROOT / ".dvc" / "config").read_text()
    )


def test_check_flag_detects_drift(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "dag.yaml", tmp_path / "configs" / "dag.yaml")
    assert mk.main(["--root", str(tmp_path), "--check"]) == 1
    assert mk.main(["--root", str(tmp_path)]) == 0
    assert mk.main(["--root", str(tmp_path), "--check"]) == 0
    (tmp_path / "dvc.yaml").write_text("stages: {}\n")
    assert mk.main(["--root", str(tmp_path), "--check"]) == 1


def test_dvc_lock_hash_fallback(tmp_path):
    assert dvc_lock_hash(tmp_path / "dvc.lock") is None
    lock = tmp_path / "dvc.lock"
    lock.write_text("schema: '2.0'\n")
    h = dvc_lock_hash(lock)
    assert h and h.startswith("dvc:") and h == dvc_lock_hash(lock)
    lock.write_text("schema: '2.0'\nstages: {}\n")
    assert dvc_lock_hash(lock) != h


def test_dvc_dag_runs(tmp_path):
    pytest.importorskip("dvc")
    for name in ("dvc.yaml", "params.yaml"):
        shutil.copy(ROOT / name, tmp_path / name)
    dvc = [sys.executable, "-m", "dvc"]
    subprocess.run([*dvc, "init", "--no-scm", "-q"], cwd=tmp_path, check=True)
    out = subprocess.run([*dvc, "dag"], cwd=tmp_path, check=True, capture_output=True, text=True)
    for st in STAGES:
        assert st in out.stdout
