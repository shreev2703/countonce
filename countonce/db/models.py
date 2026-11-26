"""SQLAlchemy 2.0 typed models for every PLAN 14.3 table (Strong and Moon tiers).

Timestamps are ``DOUBLE PRECISION`` epoch seconds, not ``timestamptz``: the job store injects a
``clock`` callable (tests step it by hand) and shares its SQL with the SQLite store. Flag columns
are real booleans. See ``docs/DEVIATIONS.md`` for what SQLite cannot express.

``PLAN_COLUMNS`` is the PLAN 14.3 column list verbatim; ``tests/test_db_schema.py`` checks the
models against it. ``EXTRA_COLUMNS`` are the columns the queue needs beyond the plan.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Job lifecycle values; ``draft`` is the API's pre-upload state (not in the plan's list).
JOB_STATUSES = ("draft", "queued", "running", "done", "failed", "dead_letter", "deleted")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    email: Mapped[str | None] = mapped_column(Text, unique=True)
    name: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[float | None] = mapped_column(Float)
    training_opt_in: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    deleted_at: Mapped[float | None] = mapped_column(Float)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name="jobs_user_idempotency_key"),
        Index("jobs_queue_idx", "status", "next_attempt_at", "created_at"),
    )
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompts: Mapped[str] = mapped_column(Text, nullable=False, server_default="[]")
    options_json: Mapped[str] = mapped_column(Text, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    stage: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    worker_id: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(Text)
    input_hash: Mapped[str | None] = mapped_column(Text)
    model_routing_id: Mapped[str | None] = mapped_column(Text)
    site_id: Mapped[str | None] = mapped_column(Text)
    captured_at: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    started_at: Mapped[float | None] = mapped_column(Float)
    finished_at: Mapped[float | None] = mapped_column(Float)
    # queue bookkeeping beyond the plan
    heartbeat_at: Mapped[float | None] = mapped_column(Float)
    next_attempt_at: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    trace_context: Mapped[str | None] = mapped_column(Text)  # W3C traceparent of the submit


class JobFile(Base):
    __tablename__ = "job_files"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)  # input | output
    key: Mapped[str] = mapped_column(Text, nullable=False)
    bytes: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(Text)
    exif_stripped: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    privacy_masked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())


class Result(Base):
    __tablename__ = "results"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    schema_version: Mapped[str | None] = mapped_column(Text)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    interval_low: Mapped[int | None] = mapped_column(Integer)
    interval_high: Mapped[int | None] = mapped_column(Integer)
    interval_level: Mapped[float | None] = mapped_column(Float)
    interval_method: Mapped[str | None] = mapped_column(Text)
    n_calibration: Mapped[int | None] = mapped_column(Integer)
    zero_count: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    result_json: Mapped[str | None] = mapped_column(Text)
    warnings_json: Mapped[str | None] = mapped_column(Text)
    model_versions_json: Mapped[str | None] = mapped_column(Text)
    recon_cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    oom_fallback_rung: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    peak_vram_mb: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(Text)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    user_count: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[float | None] = mapped_column(Float)


class Correction(Base):  # Moon tier (UI in Strong)
    __tablename__ = "corrections"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text, nullable=False)  # split | merge | reject
    detection_ids: Mapped[Any] = mapped_column(JSON)
    instance_ids: Mapped[Any] = mapped_column(JSON)
    key: Mapped[str | None] = mapped_column(Text)  # stable identity; repeats are no-ops
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    created_at: Mapped[float | None] = mapped_column(Float)
    exported_version: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (UniqueConstraint("job_id", "key", name="corrections_job_key"),)


class ShareToken(Base):
    __tablename__ = "share_tokens"
    token: Mapped[str] = mapped_column(Text, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False, index=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    revoked_at: Mapped[float | None] = mapped_column(Float)


class Worker(Base):
    __tablename__ = "workers"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text, nullable=False, server_default="laptop")
    last_heartbeat: Mapped[float | None] = mapped_column(Float)
    gpu_name: Mapped[str | None] = mapped_column(Text)
    vram_mb: Mapped[int | None] = mapped_column(Integer)
    cpu_name: Mapped[str | None] = mapped_column(Text)
    cpu_count: Mapped[int | None] = mapped_column(Integer)
    version: Mapped[str | None] = mapped_column(Text)


class ReconCache(Base):
    __tablename__ = "recon_cache"
    input_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    backbone_version: Mapped[str] = mapped_column(Text, primary_key=True)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    expires_at: Mapped[float | None] = mapped_column(Float)
    hits: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class ModelRouting(Base):  # Moon tier
    __tablename__ = "model_routing"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    detector: Mapped[str] = mapped_column(Text, nullable=False)
    backbone: Mapped[str] = mapped_column(Text, nullable=False)
    associator: Mapped[str] = mapped_column(Text, nullable=False)
    calibration: Mapped[str | None] = mapped_column(Text)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    shadow_percent: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")


class ShadowResult(Base):  # Moon tier
    __tablename__ = "shadow_results"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    model_routing_id: Mapped[str] = mapped_column(ForeignKey("model_routing.id"), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    interval_low: Mapped[int | None] = mapped_column(Integer)
    interval_high: Mapped[int | None] = mapped_column(Integer)
    delta_vs_live: Mapped[float | None] = mapped_column(Float)
    timings_json: Mapped[Any] = mapped_column(JSON)


class Site(Base):  # Moon tier
    __tablename__ = "sites"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[float | None] = mapped_column(Float)


class ApiKey(Base):  # Moon tier in the plan; the HTTP API already needs it
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    salt: Mapped[str] = mapped_column(Text, nullable=False)
    hashed_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    last_used_at: Mapped[float | None] = mapped_column(Float)
    revoked_at: Mapped[float | None] = mapped_column(Float)


# -- tables the stores need that PLAN 14.3 does not list --------------------------------------
class RateBucket(Base):
    __tablename__ = "rate_buckets"
    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    tokens: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)


class UploadSession(Base):
    __tablename__ = "upload_sessions"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    n_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open")
    dir: Mapped[str] = mapped_column(Text, nullable=False)
    job_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)


class UploadChunk(Base):
    __tablename__ = "upload_chunks"
    session_id: Mapped[str] = mapped_column(ForeignKey("upload_sessions.id"), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)


# PLAN 14.3, verbatim column lists per table.
PLAN_COLUMNS: dict[str, tuple[str, ...]] = {
    "users": ("id", "email", "name", "created_at", "training_opt_in", "deleted_at"),
    "jobs": (
        "id",
        "user_id",
        "prompt",
        "negative_prompts",
        "options_json",
        "status",
        "stage",
        "progress",
        "error_code",
        "error_message",
        "attempts",
        "worker_id",
        "idempotency_key",
        "input_hash",
        "model_routing_id",
        "site_id",
        "captured_at",
        "created_at",
        "started_at",
        "finished_at",
    ),
    "job_files": (
        "id",
        "job_id",
        "kind",
        "key",
        "bytes",
        "sha256",
        "exif_stripped",
        "privacy_masked",
    ),
    "results": (
        "job_id",
        "schema_version",
        "count",
        "interval_low",
        "interval_high",
        "interval_level",
        "interval_method",
        "n_calibration",
        "zero_count",
        "result_json",
        "warnings_json",
        "model_versions_json",
        "recon_cache_hit",
        "oom_fallback_rung",
        "peak_vram_mb",
    ),
    "feedback": ("id", "job_id", "user_id", "correct", "user_count", "comment", "created_at"),
    "corrections": (
        "id",
        "job_id",
        "user_id",
        "action",
        "detection_ids",
        "instance_ids",
        "created_at",
        "exported_version",
    ),
    "share_tokens": ("token", "job_id", "created_at", "revoked_at"),
    "workers": (
        "id",
        "name",
        "kind",
        "last_heartbeat",
        "gpu_name",
        "vram_mb",
        "cpu_name",
        "cpu_count",
        "version",
    ),
    "recon_cache": (
        "input_hash",
        "backbone_version",
        "storage_key",
        "created_at",
        "expires_at",
        "hits",
    ),
    "model_routing": (
        "id",
        "detector",
        "backbone",
        "associator",
        "calibration",
        "is_default",
        "shadow_percent",
    ),
    "shadow_results": (
        "job_id",
        "model_routing_id",
        "count",
        "interval_low",
        "interval_high",
        "delta_vs_live",
        "timings_json",
    ),
    "sites": ("id", "user_id", "name", "description", "created_at"),
    "api_keys": ("id", "user_id", "hashed_key", "created_at", "last_used_at", "revoked_at"),
}

# Columns/tables beyond the plan, with the reason each exists.
EXTRA_COLUMNS: dict[str, tuple[str, ...]] = {
    "jobs": (
        "heartbeat_at",  # stale requeue
        "next_attempt_at",  # retry backoff
        "trace_context",  # W3C traceparent of the submit (migration 0002)
    ),
    "api_keys": ("salt",),  # PBKDF2 salt for hashed_key
    "corrections": ("key", "consent"),  # idempotent replay, reuse consent (migration 0003)
}
EXTRA_TABLES = ("rate_buckets", "upload_sessions", "upload_chunks")
