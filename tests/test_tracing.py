"""Stage tracing: span tree for a fake job, error status, no-op path, OTLP skip, trace show."""

import builtins
import logging
from pathlib import Path

import pytest
from test_pipeline import PlaneRecon, make_scene

from countonce.cli import main
from countonce.config import Config
from countonce.detect.base import FakeDetector
from countonce.jobs.store import SqliteJobStore
from countonce.jobs.worker import Worker
from countonce.monitor import tracing
from countonce.pipeline import run
from countonce.types import CountResult, Interval, ModelVersions


def _tree(spans):
    ids = {s.span_id: s for s in spans}
    return {s.name: (ids[s.parent_id].name if s.parent_id else None) for s in spans}


def _traced_job(tmp_path: Path, runner_kind: str = "ok"):
    img_dir, script = make_scene(tmp_path)
    store = SqliteJobStore(tmp_path / "q.db")
    jid, _ = store.submit("u", "widgets", [str(p) for p in sorted(img_dir.glob("*.png"))])
    cfg = Config(out_dir=tmp_path / "out", max_frames=0)
    det = FakeDetector(script if runner_kind == "ok" else {})

    def runner(job, inputs, progress):
        if runner_kind == "fail":

            class Boom(PlaneRecon):
                def reconstruct(self, frames):
                    raise RuntimeError("gpu exploded")

            return run(inputs, "w", cfg, detector=det, reconstructor=Boom())
        return run(inputs, "w", cfg, detector=det, reconstructor=PlaneRecon())

    tcfg = tracing.TracingConfig("jsonl", tmp_path / "traces")
    w = Worker(store, runner, tracing_cfg=tcfg)
    assert w.step()
    return jid, tmp_path / "traces" / f"{jid}.jsonl", store


def test_span_tree_for_one_job(tmp_path: Path) -> None:
    jid, path, store = _traced_job(tmp_path)
    assert store.get(jid)["status"] == "done"
    spans = tracing.load_trace(path)
    assert len({s.trace_id for s in spans}) == 1  # one job, one trace
    tree = _tree(spans)
    assert tree["job"] is None
    for name in ("claim", "download", "run", "upload"):
        assert tree[name] == "job"
    for name in ("ingest", "detect", "reconstruct", "associate", "confidence", "export"):
        assert tree[name] == "run"
    assert all(s.status == "ok" and s.end_ns >= s.start_ns for s in spans)
    root = next(s for s in spans if s.name == "job")
    assert root.attrs["job_id"] == jid and root.start_ns <= min(s.start_ns for s in spans)
    out = tracing.waterfall(spans)
    assert "  ingest" in out and out.splitlines()[1].startswith("job")


def test_stage_spans_reuse_pipeline_timings(tmp_path: Path) -> None:
    img_dir, script = make_scene(tmp_path)
    cfg = Config(out_dir=tmp_path / "out", max_frames=0)
    tcfg = tracing.TracingConfig("jsonl", tmp_path / "t")
    with tracing.trace("j1", tcfg):
        res = run(img_dir, "w", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    by = {s.name: s for s in tracing.load_trace(tmp_path / "t" / "j1.jsonl")}
    for key, name in (("ingest", "ingest"), ("detect", "detect"), ("recon", "reconstruct")):
        assert by[name].duration_s == pytest.approx(res.timings[key], abs=1e-6)


def test_config_tracing_starts_its_own_trace(tmp_path: Path) -> None:
    img_dir, script = make_scene(tmp_path)
    cfg = Config(
        out_dir=tmp_path / "out",
        max_frames=0,
        tracing={"exporter": "jsonl", "dir": str(tmp_path / "t")},
    )
    run(img_dir, "w", cfg, detector=FakeDetector(script), reconstructor=PlaneRecon())
    spans = tracing.load_trace(tmp_path / "t" / "out.jsonl")
    assert _tree(spans)["pipeline.run"] is None and "detect" in _tree(spans)
    with pytest.raises(ValueError):
        Config(tracing={"exporter": "bogus"})


def test_failing_stage_has_error_status(tmp_path: Path) -> None:
    jid, path, store = _traced_job(tmp_path, "fail")
    assert store.get(jid)["status"] != "done"
    spans = {s.name: s for s in tracing.load_trace(path)}
    assert spans["reconstruct"].status == "error"
    assert "gpu exploded" in (spans["reconstruct"].error or "")
    assert spans["run"].status == "error" and spans["job"].status == "error"
    assert spans["ingest"].status == spans["detect"].status == "ok"
    assert "associate" not in spans and "upload" not in spans
    assert "ERROR" in tracing.waterfall(list(spans.values()))


def test_noop_path_adds_no_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with tracing.span("x", a=1) as sp:
        assert sp is None and not tracing.active()
    assert tracing.record_span("y", 0.0, 1.0) is None
    with tracing.trace("j", None) as root, tracing.trace("j", tracing.TracingConfig()) as r2:
        assert root is None and r2 is None
    store = SqliteJobStore(tmp_path / "q.db")
    store.submit("u", "p", ["a.jpg"])

    def runner(job, inputs, progress):
        return CountResult(
            prompt="p",
            count=1,
            interval=Interval(1, 1, 0.9, "m"),
            model_versions=ModelVersions(detector="d", backbone="b", associator="a"),
            input_hash="h",
        )

    before = {p for p in tmp_path.rglob("*")}
    assert Worker(store, runner).step()
    new = {p for p in tmp_path.rglob("*")} - before
    assert not [p for p in new if "trace" in str(p) or p.suffix == ".jsonl"]
    assert not (tmp_path / "runs").exists()


def test_otlp_skips_cleanly_without_package(tmp_path: Path, monkeypatch, caplog) -> None:
    real_import = builtins.__import__

    def no_otel(name, *a, **k):
        if name.startswith("opentelemetry"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_otel)
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING, "countonce.tracing"):
        with tracing.trace("j", tracing.TracingConfig("otlp", tmp_path / "t")):
            with tracing.span("stage"):
                pass
    assert "not installed" in caplog.text
    assert list(tmp_path.iterdir()) == []  # no files, no exception


def test_trace_show_cli(tmp_path: Path, capsys) -> None:
    jid, path, _ = _traced_job(tmp_path)
    capsys.readouterr()
    assert main(["trace", "show", jid, "--dir", str(path.parent)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("trace ") and "reconstruct" in out and "upload" in out
    assert main(["trace", "show", "nope", "--dir", str(path.parent)]) == 2
