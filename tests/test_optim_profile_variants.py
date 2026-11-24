"""Optimization switches and variant profiling. CPU only, fake stages."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.optim import (
    BatchedDetector,
    check_sdpa,
    configure_inference,
    maybe_compile,
    pick_sdpa_kernel,
    preload,
    wrap_detector,
)
from countonce.profile import VARIANT_COLUMNS, ProfileConfig, run_profile_variants
from countonce.types import CountResult, Interval, ModelVersions

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "make_profile_report", ROOT / "scripts" / "make_profile_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_profile_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_config_defaults_are_eager():
    c = Config()
    assert (c.compile_reconstructor, c.compile_detector) == ("off", "off")
    assert c.detector_batch_size == 0 and c.preload_workers == 0 and not c.pin_memory
    with pytest.raises(ValueError):
        Config(compile_reconstructor="max-autotune")


def test_maybe_compile_noop_without_cuda():
    sentinel = object()
    assert maybe_compile(sentinel, "off") is sentinel
    if "torch" not in sys.modules or not sys.modules["torch"].cuda.is_available():
        assert maybe_compile(sentinel, "reduce-overhead") is sentinel


def test_pick_sdpa_kernel_prefers_flash_and_flags_math():
    ok = pick_sdpa_kernel(lambda k: k in ("efficient", "math"))
    assert ok.kernel == "efficient" and not ok.math_fallback
    bad = pick_sdpa_kernel(lambda k: k == "math")
    assert bad.kernel == "math" and bad.math_fallback and "math" in bad.note
    none = pick_sdpa_kernel(lambda k: False)
    assert none.kernel is None and none.math_fallback


def test_check_sdpa_cpu_is_unavailable():
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        pytest.skip("GPU present")
    assert check_sdpa().available is False


def test_preload_orders_results_and_threads():
    paths = [Path(f"{i}.png") for i in range(20)]
    assert preload(paths, lambda p: p.stem, workers=4) == [p.stem for p in paths]
    assert preload(paths, lambda p: p.stem, workers=0, pin=True) == [p.stem for p in paths]


class CountingDetector:
    name = "fake"

    def __init__(self):
        self.calls = []

    def config(self):
        return {}

    def detect(self, frames, prompt, exemplars=None):
        self.calls.append(len(frames))
        return [f.id for f in frames]


def test_batched_detector_chunks_in_order():
    inner = CountingDetector()
    frames = [type("F", (), {"id": i})() for i in range(10)]
    out = BatchedDetector(inner, 4).detect(frames, "x")
    assert inner.calls == [4, 4, 2] and out == list(range(10))
    with pytest.raises(ValueError):
        BatchedDetector(inner, 0)


def test_wrap_and_configure():
    det = FakeDetector({})
    assert wrap_detector(det, Config()) is det
    assert isinstance(wrap_detector(det, Config(detector_batch_size=3)), BatchedDetector)

    class Rec:
        compile_mode = "off"
        preload_workers = 0
        pin_memory = False

    r = configure_inference(
        Rec(), Config(compile_reconstructor="default", preload_workers=3, pin_memory=True), "recon"
    )
    assert (r.compile_mode, r.preload_workers, r.pin_memory) == ("default", 3, True)
    fake = object()
    assert configure_inference(fake, Config(compile_detector="default"), "detector") is fake


def _result(count: int, total: float, recon: float) -> CountResult:
    return CountResult(
        prompt="x",
        count=count,
        interval=Interval(low=count, high=count, level=0.9, method="sensitivity"),
        model_versions=ModelVersions(backbone="b", detector="d", associator="v1"),
        input_hash="h",
        timings={"recon": recon, "total": total},
    )


def _profile_cfg(tmp_path: Path, variants: str) -> ProfileConfig:
    f = tmp_path / "p.yaml"
    f.write_text(
        f"inputs: {tmp_path}\nprompt: x\nwarmup: 1\nrepeats: 3\n"
        f"variants_csv: {tmp_path / 'v.csv'}\nvariants:\n{variants}"
    )
    return ProfileConfig.from_yaml(f)


def test_run_profile_variants_csv(tmp_path, monkeypatch):
    monkeypatch.setattr("countonce.profile._cuda", lambda: None)  # force CPU path
    cfg = _profile_cfg(
        tmp_path,
        "  - name: baseline\n  - name: fast\n    run: {detector_batch_size: 4}\n"
        "  - name: drifty\n    run: {preload_workers: 2}\n",
    )
    seen: list[Config] = []

    def fake_run(c: Config) -> CountResult:
        seen.append(c)
        if c.detector_batch_size:
            return _result(10, 1.0, 0.5)
        if c.preload_workers:
            return _result(12, 2.0, 1.0)
        return _result(10, 2.0, 1.0)

    path, rows = run_profile_variants(cfg, run_fn=fake_run)
    assert len(seen) == 3 * (1 + 3)  # warmup + repeats per variant
    with path.open() as fh:
        recs = list(csv.DictReader(fh))
    assert list(recs[0]) == VARIANT_COLUMNS
    deltas = {r["variant"]: r["count_delta"] for r in recs if r["stage"] == "total"}
    assert deltas == {"baseline": "0", "fast": "0", "drifty": "2"}
    assert {r["variant"] for r in recs} == {"baseline", "fast", "drifty"}
    assert all(r["peak_vram_mb"] == "" for r in recs)  # CPU: no VRAM


def test_failed_variant_recorded_not_fatal(tmp_path):
    cfg = _profile_cfg(
        tmp_path, "  - name: baseline\n  - name: boom\n    run: {pin_memory: true}\n"
    )

    def fake_run(c: Config) -> CountResult:
        if c.pin_memory:
            raise RuntimeError("compile blew up")
        return _result(5, 1.0, 0.5)

    _, rows = run_profile_variants(cfg, run_fn=fake_run)
    bad = [r for r in rows if r.variant == "boom"]
    assert len(bad) == 1 and bad[0].status.startswith("error: compile blew up")


def test_variants_validation(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _profile_cfg(tmp_path, "  - name: a\n  - name: a\n")
    with pytest.raises(ValueError, match="name"):
        _profile_cfg(tmp_path, "  - run: {}\n")
    f = tmp_path / "q.yaml"
    f.write_text(f"inputs: {tmp_path}\nprompt: x\n")
    with pytest.raises(ValueError, match="variants"):
        run_profile_variants(ProfileConfig.from_yaml(f))


def test_report_missing_csv_is_todo(tmp_path):
    mod = _script()
    out = tmp_path / "PROFILE.md"
    out.write_text("# Profiling report\n\nkeep me\n")
    assert mod.main(["--csv", str(tmp_path / "none.csv"), "--out", str(out)]) == 0
    text = out.read_text()
    assert "keep me" in text and "TODO: fill from results" in text and mod.START in text


def test_report_table_and_idempotent(tmp_path):
    mod = _script()
    cfg = _profile_cfg(
        tmp_path, "  - name: baseline\n  - name: fast\n    run: {detector_batch_size: 4}\n"
    )
    run_profile_variants(
        cfg, run_fn=lambda c: _result(10, 1.0 if c.detector_batch_size else 2.0, 0.5)
    )
    out = tmp_path / "PROFILE.md"
    out.write_text("# T\n\nintro\n")
    args = ["--csv", str(tmp_path / "v.csv"), "--out", str(out)]
    mod.main(args)
    first = out.read_text()
    assert "2.00x" in first and "| fast | 3 |" in first and "(baseline)" in first
    assert "TODO" in first  # VRAM is empty on CPU: a TODO cell, never a number
    mod.main(args)
    assert out.read_text() == first  # markers replaced, not appended


def test_percentile():
    mod = _script()
    assert mod.percentile([1, 2, 3, 4, 5], 50) == 3
    assert mod.percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)
