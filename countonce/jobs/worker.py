"""Queue worker: claim -> ``pipeline.run`` -> result row -> done (PLAN 14.4)."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from countonce.errors import CountOnceError, ErrorCode
from countonce.jobs.store import HEARTBEAT_INTERVAL_S, JobStore
from countonce.monitor import tracing
from countonce.types import CountResult

log = logging.getLogger("countonce.jobs")

# (job row, input paths, progress callback) -> CountResult
Runner = Callable[[dict[str, Any], list[Path], Callable[[str, float], None]], CountResult]


def classify(exc: BaseException) -> ErrorCode:
    """Map an exception to an error code. Unknown bugs are permanent so they burn no retries."""
    if isinstance(exc, CountOnceError):
        return exc.code
    if isinstance(exc, TimeoutError):
        return ErrorCode.ERR_WORKER_TIMEOUT
    if isinstance(exc, FileNotFoundError | ValueError | NotImplementedError):
        return ErrorCode.ERR_INVALID_INPUT
    if isinstance(exc, OSError):
        return ErrorCode.ERR_STORAGE_TRANSIENT
    return ErrorCode.ERR_NO_RECON


def result_row(res: CountResult) -> dict[str, Any]:
    """Flatten a ``CountResult`` into the PLAN 14.3 ``results`` columns."""
    return {
        "schema_version": res.schema_version,
        "count": res.count,
        "interval_low": res.interval.low,
        "interval_high": res.interval.high,
        "interval_level": res.interval.level,
        "interval_method": res.interval.method,
        "n_calibration": res.interval.n_calibration,
        "zero_count": int(res.zero_count),
        "result_json": json.dumps(res.to_dict()),
        "warnings_json": json.dumps(res.warnings),
        "model_versions_json": json.dumps(asdict(res.model_versions)),
        "recon_cache_hit": int(res.recon_cache_hit),
        "oom_fallback_rung": res.oom_fallback_rung,
        "peak_vram_mb": res.peak_vram_mb,
    }


def pipeline_runner(out_root: Path) -> Runner:
    """Default runner: ``countonce.pipeline.run`` with the job's ``options_json`` as Config."""

    def _run(
        job: dict[str, Any], inputs: list[Path], progress: Callable[[str, float], None]
    ) -> CountResult:
        from countonce.config import Config
        from countonce.pipeline import run

        options = json.loads(job["options_json"])
        options["out_dir"] = str(out_root / job["id"])
        progress("pipeline", 0.1)
        arg: Path | list[Path] = inputs[0] if len(inputs) == 1 else inputs
        res = run(arg, job["prompt"], Config(**options))
        progress("export", 0.95)
        return res

    return _run


class Worker:
    def __init__(
        self,
        store: JobStore,
        runner: Runner,
        *,
        worker_id: str | None = None,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        info: dict[str, Any] | None = None,
        tracing_cfg: tracing.TracingConfig | None = None,
    ) -> None:
        self.store = store
        self.runner = runner
        self.tracing_cfg = tracing_cfg  # None or exporter 'none': no spans, no files
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:8]}"
        self.heartbeat_interval_s = heartbeat_interval_s
        store.register_worker(
            self.worker_id, name=self.worker_id, cpu_count=os.cpu_count(), **(info or {})
        )

    def step(self) -> bool:
        """Reap stale jobs, then claim and process at most one. True if a job was processed."""
        self.store.requeue_stale()
        t_claim = time.perf_counter()
        job = self.store.claim(self.worker_id)
        if job is None:
            return False
        # the job id is only known after the claim, so the trace opens backdated to its start
        with tracing.trace(job["id"], self.tracing_cfg, "job", start_perf=t_claim):
            tracing.record_span("claim", t_claim, time.perf_counter(), worker=self.worker_id)
            self._process(job)
        return True

    def _process(self, job: dict[str, Any]) -> None:
        jid = job["id"]
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(self.heartbeat_interval_s):
                self.store.heartbeat(jid, self.worker_id)

        hb = threading.Thread(target=beat, daemon=True)
        hb.start()
        try:
            with tracing.span("download"):
                inputs = [Path(f["key"]) for f in self.store.job_files(jid, "input")]

            def progress(stage: str, frac: float) -> None:
                self.store.set_progress(jid, self.worker_id, stage, frac)

            with tracing.span("run"):  # pipeline stages (and export) nest under this span
                res = self.runner(job, inputs, progress)
            with tracing.span("upload"):
                if not self.store.complete(jid, self.worker_id, result_row(res)):
                    log.warning("job %s was reassigned; dropping result", jid)
        except Exception as exc:  # BaseException (e.g. a killed worker) leaves the job running
            tracing.mark_error(exc)
            code = classify(exc)
            outcome = self.store.fail(jid, self.worker_id, code, str(exc))
            log.info("job %s failed %s -> %s", jid, code.value, outcome.status)
        finally:
            stop.set()

    def run_loop(
        self,
        *,
        stop: threading.Event | None = None,
        poll_s: float = 1.0,
        until_idle: bool = False,
    ) -> int:
        """Process jobs until ``stop`` is set (or, with ``until_idle``, nothing is claimable)."""
        stop = stop or threading.Event()
        done = 0
        while not stop.is_set():
            if self.step():
                done += 1
            elif until_idle:
                break
            else:
                stop.wait(poll_s)
        return done


__all__ = ["Worker", "classify", "pipeline_runner", "result_row"]
