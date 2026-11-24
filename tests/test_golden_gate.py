"""Golden regression gate: compare logic, pins, CPU run and workflow security."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from bench import golden

ROOT = Path(__file__).resolve().parent.parent
TOL = dict(golden.DEFAULT_TOLERANCES)


@pytest.fixture(scope="module")
def cpu_result():
    cfg = golden.load_config()
    return golden.build_result(golden.run_synthetic(cfg), "cpu-synthetic")


def test_identical_run_passes(cpu_result):
    table, problems = golden.compare(cpu_result, copy.deepcopy(cpu_result), TOL)
    assert problems == []
    assert "REGRESSION" not in table and "| **overall** |" in table


@pytest.mark.parametrize(
    ("scene_metric", "value"),
    [("mape", 0.3), ("assoc_f1", 0.5), ("peak_vram_mb", 5000.0), ("latency_s", 99.0)],
)
def test_seeded_regression_flagged(cpu_result, scene_metric, value):
    cur = copy.deepcopy(cpu_result)
    cur["scenes"]["syn0"][scene_metric] = value
    cur["overall"] = golden.summarize(cur["scenes"])
    table, problems = golden.compare(cpu_result, cur, TOL)
    assert any(p.startswith("syn0/" + scene_metric) for p in problems)
    assert "REGRESSION" in table


def test_tolerance_off_and_small_noise_pass(cpu_result):
    cur = copy.deepcopy(cpu_result)
    cur["scenes"]["syn1"]["latency_s"] *= 100
    cur["overall"] = golden.summarize(cur["scenes"])
    assert golden.compare(cpu_result, cur, {**TOL, "latency_s": None})[1] == []
    assert golden.compare(cpu_result, cur, TOL)[1] != []


def test_missing_scene_is_regression(cpu_result):
    cur = copy.deepcopy(cpu_result)
    del cur["scenes"]["syn2"]
    assert any("syn2" in p for p in golden.compare(cpu_result, cur, TOL)[1])


def test_cli_exit_codes(cpu_result, tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(cpu_result))
    bad = copy.deepcopy(cpu_result)
    bad["scenes"]["syn0"]["mape"] = 0.5
    bad["overall"] = golden.summarize(bad["scenes"])
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(bad))
    args = ["compare", "--baseline", str(base), "--current"]
    assert golden.main([*args, str(base)]) == 0
    assert golden.main([*args, str(cur)]) == 1


def test_committed_cpu_baseline_matches_run(cpu_result):
    base = json.loads((ROOT / "configs" / "golden_cpu_baseline.json").read_text())
    _, problems = golden.compare(base, cpu_result, {**TOL, "latency_s": None})
    assert problems == []


def test_pin_mismatch_refused():
    cfg = golden.load_config()
    cfg["synthetic_scenes"][0]["n_objects"] += 1
    with pytest.raises(ValueError, match="hash"):
        golden.verify_synthetic_pins(cfg)


def test_real_scenes_need_pins(tmp_path):
    cfg = {"scenes": [{"id": "a", "dir": "a", "prompt": "x", "gt_count": 1, "sha256": None}]}
    with pytest.raises(ValueError, match="sha256"):
        golden.verify_real_pins(cfg, tmp_path)


def test_gpu_workflow_is_locked_down():
    text = (ROOT / ".github" / "workflows" / "golden-gpu.yml").read_text()
    wf = yaml.safe_load(text)
    triggers = wf.get(True, wf.get("on"))  # YAML 1.1 parses the key `on` as True
    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    assert "pull_request" not in text
    assert "secrets." not in text
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["golden"]
    assert job["runs-on"] == ["self-hosted", "gpu"]
    assert job["timeout-minutes"] <= 120
    assert any("upload-artifact" in str(s.get("uses", "")) for s in job["steps"])


def test_hosted_ci_runs_cpu_gate_only():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "bench.golden compare --cpu-synthetic" in text
    assert "self-hosted" not in text
