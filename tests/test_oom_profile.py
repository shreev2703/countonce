"""OOM ladder, allocator env, and profiling harness. CPU only, no torch."""

import csv
import subprocess
import sys

import pytest

from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.oom import (
    ALLOC_ENV,
    Attempt,
    GpuOomError,
    RungUnavailable,
    is_oom,
    run_with_oom_ladder,
)
from countonce.profile import COLUMNS, ProfileConfig, StageProfiler, run_profile


class SimulatedOOM(RuntimeError):
    pass


def _fake_oom_class():
    return type("OutOfMemoryError", (RuntimeError,), {})


def _stage_failing_below(rung: int, log: list[Attempt]):
    def stage(att: Attempt):
        log.append(att)
        if att.rung < rung:
            raise _fake_oom_class()("CUDA out of memory")
        return f"ok@{att.name}"

    return stage


NAMES = ["initial", "free_cache", "smaller_chunks", "lower_res_0.75", "lower_res_0.5"]


def test_ladder_order_and_all_rungs_fail():
    log: list[Attempt] = []
    freed: list[int] = []

    def stage(att):
        log.append(att)
        raise _fake_oom_class()("out of memory")

    with pytest.raises(GpuOomError) as ei:
        run_with_oom_ladder(stage, n_frames=16, free=lambda: freed.append(len(log)))
    assert [a.name for a in log] == NAMES
    assert freed == [1, 2, 3, 4]  # cache freed before every retry, never before the first try
    assert ei.value.code == "ERR_GPU_OOM"
    assert "ERR_GPU_OOM" in str(ei.value)
    assert ei.value.tip and ei.value.retryable
    assert ei.value.attempts == NAMES


def test_chunks_and_resolution_settings():
    log: list[Attempt] = []
    with pytest.raises(GpuOomError):
        run_with_oom_ladder(_stage_failing_below(99, log), n_frames=16, free=lambda: None)
    assert [a.chunk_size for a in log] == [None, None, 8, 8, 8]
    assert [a.resolution_scale for a in log] == [1.0, 1.0, 1.0, 0.75, 0.5]


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_success_at_rung_n_stops_ladder(n):
    log: list[Attempt] = []
    res = run_with_oom_ladder(_stage_failing_below(n, log), n_frames=8, free=lambda: None)
    assert res.rung == n
    assert len(log) == len(res.attempts)
    assert log[-1].rung == n
    assert res.value == f"ok@{log[-1].name}"


def test_non_oom_error_propagates():
    def stage(att):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        run_with_oom_ladder(stage, free=lambda: None)


def test_unavailable_rung_is_skipped():
    def stage(att):
        if att.rung < 3:
            raise _fake_oom_class()("out of memory")
        if att.resolution_scale == 0.75:
            raise RungUnavailable
        return att.name

    res = run_with_oom_ladder(stage, n_frames=8, free=lambda: None)
    assert res.name == "lower_res_0.5"


def test_is_oom():
    assert is_oom(_fake_oom_class()("x"))
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_oom(RuntimeError("shape mismatch"))
    assert not is_oom(ValueError("out of memory"))


def test_ladder_logs_rung(caplog):
    caplog.set_level("INFO", logger="countonce.oom")
    run_with_oom_ladder(_stage_failing_below(2, []), stage_name="recon", free=lambda: None)
    assert any("fallback rung 2 (smaller_chunks)" in r.message for r in caplog.records)


def test_env_var_set_before_torch_import():
    code = (
        "import sys, os\n"
        "assert 'torch' not in sys.modules\n"
        "from countonce import cli\n"
        "assert 'torch' not in sys.modules\n"
        f"assert 'expandable_segments:True' in os.environ[{ALLOC_ENV!r}]\n"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != ALLOC_ENV}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_configure_allocator_keeps_user_options(monkeypatch):
    from countonce import oom

    monkeypatch.setenv(ALLOC_ENV, "max_split_size_mb:128")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert oom.configure_allocator()
    import os

    assert os.environ[ALLOC_ENV] == "max_split_size_mb:128,expandable_segments:True"
    monkeypatch.setenv(ALLOC_ENV, "expandable_segments:False")
    oom.configure_allocator()
    assert os.environ[ALLOC_ENV] == "expandable_segments:False"


def test_configure_allocator_warns_if_torch_loaded(monkeypatch):
    from countonce import oom

    monkeypatch.setitem(sys.modules, "torch", object())
    assert oom.configure_allocator() is False


def test_stage_profiler_no_cuda():
    p = StageProfiler()
    with p.stage("a"):
        pass
    assert p.stats[0].stage == "a" and p.stats[0].wall_s >= 0
    if "torch" not in sys.modules:
        assert p.stats[0].peak_vram_mb is None


def test_pipeline_records_fallback_rung(tmp_path):
    from tests.test_pipeline import PlaneRecon, make_scene

    class OomOnce(PlaneRecon):
        calls = 0

        def reconstruct(self, frames):
            OomOnce.calls += 1
            if OomOnce.calls == 1:
                raise _fake_oom_class()("CUDA out of memory")
            return super().reconstruct(frames)

    img_dir, script = make_scene(tmp_path)
    from countonce.pipeline import run

    res = run(
        img_dir,
        "thing",
        Config(out_dir=tmp_path / "out", dedup_threshold=0),
        detector=FakeDetector(script),
        reconstructor=OomOnce(),
        export=False,
    )
    assert res.oom_fallback_rung == 1


def test_run_profile_writes_csv(tmp_path):
    from tests.test_pipeline import PlaneRecon, make_scene

    (tmp_path / "scene").mkdir()
    img_dir, script = make_scene(tmp_path / "scene")
    cfg = ProfileConfig(
        inputs=[img_dir],
        prompt="thing",
        repeats=2,
        warmup=1,
        out_csv=tmp_path / "results" / "p.csv",
        run=Config(out_dir=tmp_path / "out", dedup_threshold=0),
    )
    import countonce.pipeline as pl

    orig = pl.build_detector, pl.build_reconstructor
    pl.build_detector = lambda c: FakeDetector(script)
    pl.build_reconstructor = lambda c: PlaneRecon()
    try:
        path, rows = run_profile(cfg)
    finally:
        pl.build_detector, pl.build_reconstructor = orig
    with path.open() as fh:
        recs = list(csv.DictReader(fh))
    assert list(recs[0]) == COLUMNS
    assert {r["stage"] for r in recs} >= {"ingest", "detect", "recon", "total"}
    assert {r["repeat"] for r in recs} == {"0", "1"}
    assert len(recs) == len(rows)


def test_profile_config_requires_inputs(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("prompt: x\n")
    with pytest.raises(ValueError):
        ProfileConfig.from_yaml(f)
