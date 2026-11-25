"""One trace from ``POST /jobs/{id}/submit`` through the queue to the worker's stages."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_api import Env  # noqa: E402

from countonce.jobs.worker import Worker  # noqa: E402
from countonce.monitor import tracing  # noqa: E402
from countonce.types import CountResult, Interval, ModelVersions  # noqa: E402

TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
CFG = tracing.TracingConfig("memory")


@pytest.fixture(autouse=True)
def _clear() -> None:
    tracing.MEMORY.clear()


def _runner(job, inputs, progress):  # type: ignore[no-untyped-def]
    for stage in ("ingest", "detect", "reconstruct"):
        with tracing.span(stage):
            pass
    return CountResult(
        prompt=job["prompt"],
        count=1,
        interval=Interval(1, 1, 0.9, "sensitivity"),
        model_versions=ModelVersions(detector="d", backbone="b", associator="a"),
        input_hash="h",
    )


def _by_name(spans: list[tracing.Span]) -> dict[str, tracing.Span]:
    return {s.name: s for s in spans}


def test_one_trace_from_submit_to_upload(tmp_path: Path) -> None:
    env = Env(tmp_path)
    c = env.client(tracing_cfg=CFG)
    jid = env.queued_job(c)
    spans = _by_name(tracing.MEMORY.spans)
    submit = spans["POST /jobs/{job_id}/submit"]
    assert submit.parent_id is None and submit.attrs["http.status_code"] == 200
    assert submit.attrs["http.route"] == "/jobs/{job_id}/submit"
    assert env.store.get(jid)["trace_context"] == (
        tracing.format_traceparent(submit.trace_id, submit.span_id)
    )

    w = Worker(env.store, _runner, tracing_cfg=CFG, blobs=None)
    assert w.step() and env.store.get(jid)["status"] == "done"

    all_spans = tracing.MEMORY.spans
    names = {s.name for s in all_spans if s.trace_id == submit.trace_id}
    assert {"job", "claim", "download", "run", "ingest", "detect", "reconstruct", "upload"} <= names
    # the upload/create/health calls are request spans of their own traces; every span of the
    # job itself is in the submit's trace
    job_names = {"job", "claim", "download", "run", "ingest", "detect", "reconstruct", "upload"}
    assert {s.trace_id for s in all_spans if s.name in job_names} == {submit.trace_id}
    ids = {s.span_id: s for s in all_spans}
    parent = {
        s.name: ids[s.parent_id].name if s.parent_id in ids else None
        for s in all_spans
        if s.trace_id == submit.trace_id
    }
    assert parent["job"] == "POST /jobs/{job_id}/submit"
    for n in ("claim", "download", "run", "upload"):
        assert parent[n] == "job"
    for n in ("ingest", "detect", "reconstruct"):
        assert parent[n] == "run"
    assert next(s for s in all_spans if s.name == "job").parent_id == submit.span_id


def test_incoming_traceparent_is_continued_and_echoed(tmp_path: Path) -> None:
    env = Env(tmp_path)
    c = env.client(tracing_cfg=CFG)
    r = c.get("/health", headers={"traceparent": TP})
    span = _by_name(tracing.MEMORY.spans)["GET /health"]
    assert span.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert span.parent_id == "00f067aa0ba902b7"
    assert r.headers["traceparent"] == tracing.format_traceparent(span.trace_id, span.span_id)
    assert span.attrs["http.method"] == "GET"
    # a malformed header starts a fresh trace instead of failing the request
    r = c.get("/health", headers={"traceparent": "garbage"})
    assert r.status_code == 200
    fresh = tracing.parse_traceparent(r.headers["traceparent"])
    assert fresh is not None and fresh.trace_id != "4bf92f3577b34da6a3ce929d0e0e4736"


def test_error_status_is_tagged(tmp_path: Path) -> None:
    env = Env(tmp_path)
    c = env.client(tracing_cfg=CFG)
    assert c.get("/jobs/nope", headers=env.h).status_code == 404
    span = _by_name(tracing.MEMORY.spans)["GET /jobs/{job_id}"]
    assert span.attrs["http.status_code"] == 404


def test_tracing_off_adds_nothing(tmp_path: Path) -> None:
    env = Env(tmp_path)
    c = env.client()
    jid = env.queued_job(c)
    assert "traceparent" not in c.get("/health").headers
    assert env.store.get(jid)["trace_context"] is None
    assert tracing.MEMORY.spans == []


def test_worker_starts_a_fresh_trace_without_stored_context(tmp_path: Path) -> None:
    env = Env(tmp_path)
    c = env.client()  # API untraced: the job has no trace_context (an old job)
    jid = env.queued_job(c)
    Worker(env.store, _runner, tracing_cfg=CFG).step()
    spans = tracing.MEMORY.spans
    assert len({s.trace_id for s in spans}) == 1
    root = next(s for s in spans if s.name == "job")
    assert root.parent_id is None and root.attrs["job_id"] == jid


def test_worker_ignores_a_malformed_stored_context(tmp_path: Path) -> None:
    env = Env(tmp_path)
    jid, _ = env.store.submit("u", "apples", ["a.jpg"], trace_context="not-a-traceparent")
    Worker(env.store, _runner, tracing_cfg=CFG).step()
    root = next(s for s in tracing.MEMORY.spans if s.name == "job")
    assert root.parent_id is None and root.attrs["job_id"] == jid


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "00-" + "0" * 32 + "-00f067aa0ba902b7-01",  # all-zero trace id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01",  # all-zero span id
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # forbidden version
        "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",  # short trace id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",  # v00 is exactly 4 parts
        "00-4BF92F3577B34DA6A3CE929D0E0E473G-00f067aa0ba902b7-01",
    ],
)
def test_parse_traceparent_rejects_invalid(value: str | None) -> None:
    assert tracing.parse_traceparent(value) is None


def test_traceparent_roundtrip_and_flags() -> None:
    ctx = tracing.parse_traceparent(TP.upper())
    assert ctx == tracing.TraceContext("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", True)
    assert tracing.format_traceparent(ctx.trace_id, ctx.span_id, ctx.sampled) == TP
    off = tracing.parse_traceparent(TP[:-2] + "00")
    assert off is not None and not off.sampled
    assert tracing.parse_traceparent("01-" + TP[3:] + "-more") is not None  # future versions
