"""FastAPI app factory over the sqlite job store (PLAN 14.2, 14.8 idempotency).

Backend only: sqlite stands in for Postgres, local disk for R2 signed URLs, and per-user API keys
(``Authorization: Bearer co_...`` or ``X-API-Key``) for Google sign-in; see docs/DEVIATIONS.md.

Job lifecycle over HTTP: ``POST /jobs`` creates a ``draft`` job; files are attached through the
chunked-upload endpoints (``/uploads``); ``POST /jobs/{id}/submit`` queues it. A worker (a separate
process on the same database) claims and completes it. Error bodies are
``{"error": {"code", "message", "tip", "retryable"}}`` with the status from ``HTTP_STATUS``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from countonce.errors import CountOnceError, ErrorCode, is_retryable, user_tip
from countonce.jobs.store import TERMINAL, RateLimitError, SqliteJobStore
from countonce.jobs.uploads import UploadLimits, UploadService
from countonce.monitor import tracing
from countonce.storage import BlobStore, LocalBlobStore

HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.ERR_INVALID_INPUT: 400,
    ErrorCode.ERR_UNSUPPORTED_MEDIA: 415,
    ErrorCode.ERR_UNSUPPORTED_CODEC: 415,
    ErrorCode.ERR_FILE_TOO_LARGE: 413,
    ErrorCode.ERR_CORRUPT_MEDIA: 422,
    ErrorCode.ERR_CHUNK_CORRUPT: 422,
    ErrorCode.ERR_TOO_FEW_FRAMES: 422,
    ErrorCode.ERR_NO_RECON: 422,
    ErrorCode.ERR_NO_OVERLAP: 422,
    ErrorCode.ERR_INVALID_CORRECTION: 422,
    ErrorCode.ERR_CANNOT_LINK: 409,
    ErrorCode.ERR_UPLOAD_INCOMPLETE: 409,
    ErrorCode.ERR_GPU_OOM: 503,
    ErrorCode.ERR_STORAGE_TRANSIENT: 503,
    ErrorCode.ERR_WORKER_TIMEOUT: 504,
}

WORKER_ONLINE_S = 120.0  # same window as the heartbeat timeout
PUBLIC_JOB_FIELDS = (
    "id",
    "prompt",
    "status",
    "stage",
    "progress",
    "error_code",
    "error_message",
    "attempts",
    "site_id",
    "captured_at",
    "created_at",
    "started_at",
    "finished_at",
)


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        tip: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.tip = tip
        self.headers = headers or {}


def _error_body(code: str, message: str, tip: str | None, retryable: bool) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "tip": tip, "retryable": retryable}}


class JobCreate(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompts: list[str] = []
    options: dict[str, Any] = {}
    site_id: str | None = None
    captured_at: float | None = None


class FeedbackIn(BaseModel):
    correct: bool
    user_count: int | None = None
    comment: str | None = None


class UploadCreate(BaseModel):
    filename: str
    total_bytes: int
    sha256: str
    chunk_size: int


class FinalizeIn(BaseModel):
    job_id: str | None = None
    prompt: str | None = None


class _RequestLimiter:
    """In-memory token bucket per user for all authenticated requests (None = unlimited)."""

    def __init__(self, per_window: int | None, window_s: float, clock: Callable[[], float]) -> None:
        self.cap = per_window
        self.window_s = window_s
        self.clock = clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, user_id: str) -> None:
        if self.cap is None:
            return
        now = self.clock()
        with self._lock:
            tokens, last = self._state.get(user_id, (float(self.cap), now))
            tokens = min(float(self.cap), tokens + (now - last) * self.cap / self.window_s)
            if tokens < 1.0:
                self._state[user_id] = (tokens, now)
                wait = (1.0 - tokens) * self.window_s / self.cap
                raise _rate_limited(wait)
            self._state[user_id] = (tokens - 1.0, now)


def _rate_limited(retry_after_s: float) -> ApiError:
    return ApiError(
        429,
        "ERR_RATE_LIMITED",
        f"rate limit exceeded; retry in {retry_after_s:.1f}s",
        "Slow down and retry after the Retry-After interval.",
        {"Retry-After": str(max(1, int(retry_after_s + 0.999)))},
    )


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    out = {k: job.get(k) for k in PUBLIC_JOB_FIELDS}
    out["negative_prompts"] = json.loads(job.get("negative_prompts") or "[]")
    out["options"] = json.loads(job.get("options_json") or "{}")
    if job.get("error_code"):
        out["tip"] = user_tip(job["error_code"])
    return out


def _result_view(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("result_json", "warnings_json", "model_versions_json"):
        if isinstance(out.get(key), str):
            out[key] = json.loads(out[key])
    return out


class TraceMiddleware:
    """Pure-ASGI request span: continues an incoming ``traceparent``, tags method/route/status and
    echoes the request span as a ``traceparent`` response header. The span is current while the
    route runs, so ``submit`` can store its context on the job for the worker to continue."""

    def __init__(self, app: Any, cfg: tracing.TracingConfig) -> None:
        self.app, self.cfg = app, cfg

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        parent = tracing.parse_traceparent(headers.get("traceparent"))
        method = scope["method"]
        with tracing.trace(
            f"req-{uuid.uuid4().hex[:12]}",
            self.cfg,
            f"{method} {scope['path']}",
            parent=parent,
            **{"http.method": method},
        ) as root:

            async def send_wrapped(message: Any) -> None:
                if root is not None and message["type"] == "http.response.start":
                    root.attrs["http.status_code"] = message["status"]
                    if message["status"] >= 500:
                        root.status = "error"
                    tp = tracing.format_traceparent(root.trace_id, root.span_id)
                    message = {
                        **message,
                        "headers": [*message["headers"], (b"traceparent", tp.encode())],
                    }
                await send(message)

            try:
                await self.app(scope, receive, send_wrapped)
            finally:
                if root is not None:
                    route = scope.get("route")
                    path = getattr(route, "path", None)
                    if path:  # the route template, not the raw path (no ids in span names)
                        root.name = f"{method} {path}"
                        root.attrs["http.route"] = path


def create_app(
    store: SqliteJobStore,
    uploads: UploadService | None = None,
    *,
    request_rate_limit: int | None = None,
    request_rate_window_s: float = 60.0,
    worker_online_s: float = WORKER_ONLINE_S,
    sse_poll_s: float = 0.5,
    sse_max_s: float = 3600.0,
    tracing_cfg: tracing.TracingConfig | None = None,
) -> FastAPI:
    """Build the app. ``store.rate_limit`` limits job *submits*; ``request_rate_limit`` limits all
    authenticated requests per user (both give 429 with ``Retry-After``)."""
    svc = uploads or UploadService(store, limits=UploadLimits())
    limiter = _RequestLimiter(request_rate_limit, request_rate_window_s, time.time)
    app = FastAPI(title="CountOnce API", version="0.1.0")
    if tracing_cfg is not None and tracing_cfg.enabled:
        app.add_middleware(TraceMiddleware, cfg=tracing_cfg)

    # -- errors ---------------------------------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            _error_body(exc.code, exc.message, exc.tip, exc.status in (429, 503, 504)),
            status_code=exc.status,
            headers=exc.headers,
        )

    @app.exception_handler(CountOnceError)
    async def _typed_error(_: Request, exc: CountOnceError) -> JSONResponse:
        return JSONResponse(
            _error_body(exc.code.value, str(exc), exc.tip, exc.retryable),
            status_code=HTTP_STATUS[exc.code],
        )

    @app.exception_handler(RateLimitError)
    async def _rate(_: Request, exc: RateLimitError) -> JSONResponse:
        err = _rate_limited(exc.retry_after_s)
        return await _api_error(_, err)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        code = ErrorCode.ERR_INVALID_INPUT
        return JSONResponse(_error_body(code.value, msg, user_tip(code), False), status_code=400)

    # -- auth -----------------------------------------------------------------------------
    def current_user(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> str:
        key = x_api_key
        if key is None and authorization and authorization.lower().startswith("bearer "):
            key = authorization[7:].strip()
        user = store.verify_api_key(key) if key else None
        if user is None:
            raise ApiError(
                401, "ERR_UNAUTHENTICATED", "missing or invalid API key",
                "Send a live key as 'Authorization: Bearer <key>' or 'X-API-Key'.",
                {"WWW-Authenticate": "Bearer"},
            )  # fmt: skip
        limiter.check(user)
        return user

    User = Depends(current_user)

    def own_job(job_id: str, user: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None or job["user_id"] != user or job["status"] == "deleted":
            raise ApiError(404, "ERR_NOT_FOUND", f"no such job: {job_id}")
        return job

    def own_session(session_id: str, user: str) -> None:
        if svc.owner(session_id) != user:
            raise ApiError(404, "ERR_NOT_FOUND", f"no such upload session: {session_id}")

    # -- jobs -----------------------------------------------------------------------------
    @app.post("/jobs", status_code=201)
    def create_job(
        body: JobCreate,
        response: Response,
        user: str = User,
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        job_id, created = store.create_draft(
            user,
            body.prompt,
            options=body.options,
            negative_prompts=body.negative_prompts,
            idempotency_key=idempotency_key,
            site_id=body.site_id,
            captured_at=body.captured_at,
        )
        if not created:
            response.status_code = 200
        return {
            "id": job_id,
            "status": own_job(job_id, user)["status"],
            "idempotent_replay": not created,
            "upload": {"create_session": "/uploads", "attach_with": {"job_id": job_id}},
            "submit": f"/jobs/{job_id}/submit",
        }

    @app.post("/jobs/{job_id}/submit")
    def submit_job(
        job_id: str, user: str = User, idempotency_key: str | None = Header(default=None)
    ) -> dict[str, Any]:
        job = own_job(job_id, user)
        if job["status"] == "draft":
            if not store.job_files(job_id):
                raise ApiError(
                    400, ErrorCode.ERR_INVALID_INPUT.value, "the job has no uploaded files",
                    "Upload at least one photo or video before submitting.",
                )  # fmt: skip
            # RateLimitError -> 429. False means a concurrent duplicate submit won the
            # draft -> queued transition, so this response is a replay (found by loadtest-api).
            replay = not store.queue_draft(job_id, tracing.current_traceparent())
        else:
            replay = True  # queue_draft is a state transition: a retry never queues twice
        return {
            **_job_view(own_job(job_id, user)),
            "idempotent_replay": replay,
            "idempotency_key": idempotency_key,
        }

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, user: str = User) -> dict[str, Any]:
        view = _job_view(own_job(job_id, user))
        view["files"] = len(store.job_files(job_id))
        return view

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, user: str = User) -> StreamingResponse:
        own_job(job_id, user)

        async def stream() -> AsyncIterator[str]:
            last: tuple[Any, ...] | None = None
            deadline = time.monotonic() + sse_max_s
            while time.monotonic() < deadline:
                job = store.get(job_id)
                if job is None:
                    break
                state = (job["status"], job["stage"], job["progress"], job["error_code"])
                if state != last:
                    last = state
                    event = "done" if job["status"] in TERMINAL else "progress"
                    yield f"event: {event}\ndata: {json.dumps(_job_view(job))}\n\n"
                    if job["status"] in TERMINAL:
                        return
                else:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(sse_poll_s)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/jobs/{job_id}/result")
    def get_result(job_id: str, user: str = User) -> Any:
        job = own_job(job_id, user)
        if job["status"] == "done":
            outputs = store.job_files(job_id, "output")
            return {
                "job": _job_view(job),
                "result": _result_view(store.get_result(job_id) or {}),
                "outputs": [
                    {"id": f["id"], "name": Path(f["key"]).name, "bytes": f["bytes"],
                     "url": f"/jobs/{job_id}/files/{f['id']}"}
                    for f in outputs
                ],
            }  # fmt: skip
        if job["error_code"]:
            code = ErrorCode(job["error_code"])
            return JSONResponse(
                _error_body(code.value, job["error_message"] or "", user_tip(code),
                            is_retryable(code)),
                status_code=HTTP_STATUS[code],
            )  # fmt: skip
        raise ApiError(
            409, "ERR_NOT_READY", f"job is {job['status']}", "Wait for the job to finish."
        )

    @app.get("/jobs/{job_id}/files/{file_id}")
    def get_file(job_id: str, file_id: str, user: str = User) -> Response:
        own_job(job_id, user)
        blobs = store.blobs
        for f in store.job_files(job_id, "output"):
            if f["id"] != file_id:
                continue
            if blobs is not None and not isinstance(blobs, LocalBlobStore):
                # object storage: hand out a short-lived signed URL, never proxy or expose keys
                if blobs.exists(f["key"]):
                    return RedirectResponse(blobs.presign_get(f["key"]), status_code=307)
            elif Path(f["key"]).is_file():
                return FileResponse(f["key"], filename=Path(f["key"]).name)
        raise ApiError(404, "ERR_NOT_FOUND", f"no such output file: {file_id}")

    @app.post("/jobs/{job_id}/feedback", status_code=201)
    def post_feedback(job_id: str, body: FeedbackIn, user: str = User) -> dict[str, Any]:
        job = own_job(job_id, user)
        if job["status"] != "done":
            raise ApiError(409, "ERR_NOT_READY", "feedback needs a finished job")
        fid = store.add_feedback(job_id, user, body.correct, body.user_count, body.comment)
        return {"id": fid}

    @app.delete("/jobs/{job_id}")
    def delete_job(job_id: str, user: str = User) -> dict[str, Any]:
        own_job(job_id, user)
        try:
            store.delete_job(job_id)
        except ValueError as exc:
            raise ApiError(409, "ERR_JOB_RUNNING", str(exc), "Wait for the job to finish.") from exc
        return {"id": job_id, "status": "deleted"}

    @app.get("/jobs")
    def list_jobs(status: str | None = None, user: str = User) -> dict[str, Any]:
        rows = [j for j in store.list_jobs(status) if j["user_id"] == user]
        rows = [j for j in rows if j["status"] != "deleted" or status == "deleted"]
        return {"jobs": [_job_view(j) for j in rows]}

    # -- sharing --------------------------------------------------------------------------
    @app.post("/jobs/{job_id}/share", status_code=201)
    def create_share(job_id: str, user: str = User) -> dict[str, Any]:
        own_job(job_id, user)
        token = store.create_share_token(job_id)
        return {"token": token, "url": f"/share/{token}"}

    @app.get("/share/{token}")
    def get_share(token: str) -> dict[str, Any]:
        view = store.resolve_share_token(token)
        if view is None:
            raise ApiError(404, "ERR_NOT_FOUND", "unknown or revoked share link")
        res = view["result"]
        return {"job": view["job"], "result": _result_view(res) if res else None}

    @app.delete("/share/{token}")
    def revoke_share(token: str, user: str = User) -> dict[str, Any]:
        view = store.resolve_share_token(token)
        if view is None or own_job(view["job"]["id"], user) is None:
            raise ApiError(404, "ERR_NOT_FOUND", "unknown or revoked share link")
        store.revoke_share_token(token)
        return {"token": token, "revoked": True}

    # -- chunked uploads ------------------------------------------------------------------
    @app.post("/uploads", status_code=201)
    def create_upload(body: UploadCreate, user: str = User) -> dict[str, Any]:
        return svc.create_session(
            user, body.filename, body.total_bytes, body.sha256, body.chunk_size
        )

    @app.get("/uploads/{session_id}")
    def upload_status(session_id: str, user: str = User) -> dict[str, Any]:
        own_session(session_id, user)
        return svc.status(session_id)

    @app.put("/uploads/{session_id}/chunks/{index}")
    async def put_chunk(
        session_id: str,
        index: int,
        request: Request,
        user: str = User,
        x_chunk_sha256: str | None = Header(default=None),
    ) -> dict[str, Any]:
        own_session(session_id, user)
        data = await request.body()
        return svc.put_chunk(session_id, index, data, x_chunk_sha256)

    @app.post("/uploads/{session_id}/finalize")
    def finalize_upload(session_id: str, body: FinalizeIn, user: str = User) -> dict[str, Any]:
        own_session(session_id, user)
        out = svc.finalize(session_id, body.job_id, body.prompt)
        out.pop("path", None)  # server-local path: not for clients
        return out

    # -- health ---------------------------------------------------------------------------
    @app.head("/health", include_in_schema=False)
    @app.get("/health")
    def health() -> dict[str, Any]:
        cutoff = store.clock() - worker_online_s
        workers = store._query(
            "SELECT id, name, kind, last_heartbeat FROM workers WHERE last_heartbeat >= ?",
            (cutoff,),
        )
        counts = store.counts()
        return {
            "status": "ok",
            "workers_online": len(workers),
            "processing_paused": len(workers) == 0,
            "workers": [{"id": w["id"], "name": w["name"], "kind": w["kind"]} for w in workers],
            "queue": {"queued": counts.get("queued", 0), "running": counts.get("running", 0)},
        }

    return app


def open_app(
    db: Path,
    upload_root: Path | None = None,
    *,
    rate_limit: int | None = None,
    request_rate_limit: int | None = None,
    limits_path: Path | None = None,
    store_kind: str = "sqlite",
    pg_url: str | None = None,
    blobs: BlobStore | None = None,
    tracing_cfg: tracing.TracingConfig | None = None,
) -> FastAPI:
    """App for a job database; upload limits from ``configs/uploads.yaml``.

    ``store_kind="postgres"`` uses ``pg_url`` / ``COUNTONCE_PG_URL``; ``db`` then only anchors the
    local ``outputs`` and ``uploads`` folders.
    """
    from countonce.jobs.store import open_store
    from countonce.resources import resolve_config

    store = open_store(
        store_kind,
        db,
        pg_url=pg_url,
        rate_limit=rate_limit,
        output_root=Path(db).parent / "outputs",
        blobs=blobs,
    )
    limits = UploadLimits.from_yaml(resolve_config(limits_path or Path("configs/uploads.yaml")))
    return create_app(
        store,
        UploadService(store, upload_root, limits),
        request_rate_limit=request_rate_limit,
        tracing_cfg=tracing_cfg,
    )
