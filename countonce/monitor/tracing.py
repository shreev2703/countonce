"""Stage tracing: a tiny span API, a JSONL file exporter and an optional OTLP exporter.

Off by default. ``span(name, **attrs)`` is a no-op (yields ``None``, creates nothing) unless a
trace is active, which only ``trace(...)`` starts. A trace buffers its spans and hands them to an
exporter when the root span closes: ``jsonl`` writes ``<dir>/<job_id>.jsonl`` (one span per line),
``otlp`` forwards to OpenTelemetry if the ``opentelemetry`` packages are installed (lazy import;
absent packages log a warning and export nothing). Nothing here needs a network or an account
unless OTLP is configured.

Stage durations are not measured twice: the pipeline already times its stages for
``CountResult.timings`` (see ``countonce/profile.py``), and ``record_span`` turns those same
``perf_counter`` readings into spans.
"""

from __future__ import annotations

import contextvars
import json
import logging
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("countonce.tracing")

EXPORTERS = ("none", "jsonl", "otlp")
DEFAULT_DIR = Path("runs/traces")


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_ns: int
    end_ns: int = 0
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_ns - self.start_ns) / 1e9)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TracingConfig:
    exporter: str = "none"
    dir: Path = DEFAULT_DIR

    def __post_init__(self) -> None:
        if self.exporter not in EXPORTERS:
            raise ValueError(f"tracing.exporter must be one of {EXPORTERS}, got {self.exporter!r}")
        self.dir = Path(self.dir)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TracingConfig:
        raw = dict(raw or {})
        bad = set(raw) - {"exporter", "dir"}
        if bad:
            raise ValueError(f"unknown tracing key(s): {sorted(bad)}")
        return cls(**raw)

    @property
    def enabled(self) -> bool:
        return self.exporter != "none"


class Tracer:
    """Collects the spans of one trace and exports them when the root closes."""

    def __init__(self, job_id: str, cfg: TracingConfig) -> None:
        self.job_id, self.cfg = job_id, cfg
        self.trace_id = secrets.token_hex(16)
        self.spans: list[Span] = []
        # perf_counter -> epoch offset, so pipeline timings can become spans without a second clock
        self._offset_ns = time.time_ns() - time.perf_counter_ns()

    def ns(self, perf_s: float) -> int:
        return int(perf_s * 1e9) + self._offset_ns

    def new_span(
        self, name: str, parent: Span | None, start_ns: int, attrs: dict[str, Any]
    ) -> Span:
        return Span(
            self.trace_id,
            secrets.token_hex(8),
            parent.span_id if parent else None,
            name,
            start_ns,
            attrs=dict(attrs),
        )

    def export(self) -> Path | None:
        exp = self.cfg.exporter
        if exp == "jsonl":
            return export_jsonl(self.spans, self.cfg.dir / f"{self.job_id}.jsonl")
        if exp == "otlp":
            export_otlp(self.spans, service_name="countonce")
        return None


_TRACER: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar("co_tracer", default=None)
_CURRENT: contextvars.ContextVar[Span | None] = contextvars.ContextVar("co_span", default=None)


def current_span() -> Span | None:
    return _CURRENT.get()


def active() -> bool:
    return _TRACER.get() is not None


def _fail(sp: Span, exc: BaseException) -> None:
    sp.status, sp.error = "error", f"{type(exc).__name__}: {exc}"


def mark_error(exc: BaseException) -> None:
    """Flag the current span as failed (for an exception handled without leaving the span)."""
    sp = _CURRENT.get()
    if sp is not None:
        _fail(sp, exc)


@contextmanager
def trace(
    job_id: str,
    cfg: TracingConfig | None = None,
    name: str = "job",
    *,
    start_perf: float | None = None,
    **attrs: Any,
) -> Iterator[Span | None]:
    """Open a trace with a root span; no-op (yields None) when disabled or one is already active.

    ``start_perf`` backdates the root to a ``time.perf_counter()`` reading (the worker starts it
    after the claim, which is what tells it the job id).
    """
    if cfg is None or not cfg.enabled or _TRACER.get() is not None:
        yield None
        return
    tracer = Tracer(job_id, cfg)
    start = tracer.ns(start_perf) if start_perf is not None else time.time_ns()
    root = tracer.new_span(name, None, start, {"job_id": job_id, **attrs})
    tracer.spans.append(root)
    t_tok, s_tok = _TRACER.set(tracer), _CURRENT.set(root)
    try:
        yield root
    except BaseException as exc:
        _fail(root, exc)
        raise
    finally:
        root.end_ns = time.time_ns()
        _TRACER.reset(t_tok)
        _CURRENT.reset(s_tok)
        try:
            tracer.export()
        except Exception:  # tracing must never fail a job
            log.warning("trace export failed for %s", job_id, exc_info=True)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span | None]:
    """Child of the current span. Yields None and does nothing when no trace is active."""
    tracer = _TRACER.get()
    if tracer is None:
        yield None
        return
    sp = tracer.new_span(name, _CURRENT.get(), time.time_ns(), attrs)
    tracer.spans.append(sp)
    tok = _CURRENT.set(sp)
    try:
        yield sp
    except BaseException as exc:
        _fail(sp, exc)
        raise
    finally:
        sp.end_ns = time.time_ns()
        _CURRENT.reset(tok)


def record_span(
    name: str,
    start_perf: float,
    end_perf: float,
    *,
    error: BaseException | None = None,
    **attrs: Any,
) -> Span | None:
    """Add a finished child span from two existing ``perf_counter`` readings (no new timing)."""
    tracer = _TRACER.get()
    if tracer is None:
        return None
    sp = tracer.new_span(name, _CURRENT.get(), tracer.ns(start_perf), attrs)
    sp.end_ns = tracer.ns(end_perf)
    if error is not None:
        _fail(sp, error)
    tracer.spans.append(sp)
    return sp


def export_jsonl(spans: list[Span], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for sp in spans:
            fh.write(json.dumps(sp.to_dict()) + "\n")
    return path


def export_otlp(spans: list[Span], service_name: str = "countonce") -> bool:
    """Send spans via OpenTelemetry's OTLP exporter. False (with a warning) if not installed."""
    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        log.warning("tracing.exporter is 'otlp' but opentelemetry is not installed; skipping")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    tracer = provider.get_tracer("countonce")
    made: dict[str, Any] = {}
    for sp in sorted(spans, key=lambda s: s.start_ns):  # parents start first
        parent = made.get(sp.parent_id or "")
        ctx = ot_trace.set_span_in_context(parent) if parent is not None else None
        ot = tracer.start_span(
            sp.name,
            context=ctx,
            start_time=sp.start_ns,
            attributes={k: str(v) for k, v in sp.attrs.items()},
        )
        if sp.status == "error":
            ot.set_status(Status(StatusCode.ERROR, sp.error or ""))
        made[sp.span_id] = ot
    for sp in spans:
        made[sp.span_id].end(end_time=sp.end_ns)
    provider.shutdown()
    return True


def load_trace(path: Path) -> list[Span]:
    return [Span(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def waterfall(spans: list[Span], width: int = 40) -> str:
    """Indented tree, one row per span: offset, duration, status and a proportional bar."""
    if not spans:
        return "(empty trace)"
    t0 = min(s.start_ns for s in spans)
    total = max(max(s.end_ns for s in spans) - t0, 1)
    kids: dict[str | None, list[Span]] = {}
    ids = {s.span_id for s in spans}
    for s in sorted(spans, key=lambda s: s.start_ns):
        kids.setdefault(s.parent_id if s.parent_id in ids else None, []).append(s)
    lines = [f"trace {spans[0].trace_id}  total {total / 1e9:.3f}s"]

    def walk(parent: str | None, depth: int) -> None:
        for s in kids.get(parent, []):
            a = int((s.start_ns - t0) / total * width)
            b = max(a + 1, int((s.end_ns - t0) / total * width))
            bar = " " * a + "#" * min(b - a, width - a)
            mark = "" if s.status == "ok" else f"  ERROR {s.error or ''}"
            label = ("  " * depth + s.name).ljust(28)
            lines.append(
                f"{label} +{(s.start_ns - t0) / 1e9:8.3f}s {s.duration_s:8.3f}s "
                f"|{bar:<{width}}|{mark}"
            )
            walk(s.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)
