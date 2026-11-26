"""Typed error taxonomy and retry policy (PLAN 14.4).

Every failure gets a code. *Retryable* infrastructure errors are retried with exponential backoff;
*permanent* user/media errors go straight to ``dead_letter`` with an actionable tip. Pure module:
no I/O, no heavy imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    ERR_GPU_OOM = "ERR_GPU_OOM"
    ERR_WORKER_TIMEOUT = "ERR_WORKER_TIMEOUT"
    ERR_STORAGE_TRANSIENT = "ERR_STORAGE_TRANSIENT"
    ERR_UNSUPPORTED_MEDIA = "ERR_UNSUPPORTED_MEDIA"
    ERR_UNSUPPORTED_CODEC = "ERR_UNSUPPORTED_CODEC"
    ERR_FILE_TOO_LARGE = "ERR_FILE_TOO_LARGE"
    ERR_CORRUPT_MEDIA = "ERR_CORRUPT_MEDIA"
    ERR_TOO_FEW_FRAMES = "ERR_TOO_FEW_FRAMES"
    ERR_NO_RECON = "ERR_NO_RECON"
    ERR_NO_OVERLAP = "ERR_NO_OVERLAP"
    ERR_CANNOT_LINK = "ERR_CANNOT_LINK"
    ERR_INVALID_CORRECTION = "ERR_INVALID_CORRECTION"
    ERR_INVALID_INPUT = "ERR_INVALID_INPUT"
    ERR_CHUNK_CORRUPT = "ERR_CHUNK_CORRUPT"
    ERR_UPLOAD_INCOMPLETE = "ERR_UPLOAD_INCOMPLETE"


@dataclass(frozen=True)
class ErrorInfo:
    retryable: bool
    tip: str


GPU_OOM_TIP = (
    "The GPU ran out of memory even after freeing the cache, using smaller chunks and lowering "
    "the resolution. Use fewer or shorter frames (lower max_frames or fps), close other "
    "GPU programs, or retry later."
)

ERROR_INFO: dict[ErrorCode, ErrorInfo] = {
    ErrorCode.ERR_GPU_OOM: ErrorInfo(True, GPU_OOM_TIP),
    ErrorCode.ERR_WORKER_TIMEOUT: ErrorInfo(
        True, "The worker timed out. This is usually temporary; retry in a few minutes."
    ),
    ErrorCode.ERR_STORAGE_TRANSIENT: ErrorInfo(
        True, "Storage was briefly unavailable. This is usually temporary; retry shortly."
    ),
    ErrorCode.ERR_UNSUPPORTED_MEDIA: ErrorInfo(
        False, "This file type is not supported. Upload JPEG/PNG/WebP photos or an MP4 video."
    ),
    ErrorCode.ERR_UNSUPPORTED_CODEC: ErrorInfo(
        False, "Your video uses a codec we can't read. Export it as MP4/H.264."
    ),
    ErrorCode.ERR_FILE_TOO_LARGE: ErrorInfo(
        False,
        "The file exceeds the size, resolution or duration limit. Trim the video or "
        "downscale the photos and upload again.",
    ),
    ErrorCode.ERR_CORRUPT_MEDIA: ErrorInfo(
        False, "The file is damaged or incomplete. Re-export or re-upload it."
    ),
    ErrorCode.ERR_TOO_FEW_FRAMES: ErrorInfo(
        False, "Not enough usable frames. Capture more photos or a longer video with overlap."
    ),
    ErrorCode.ERR_NO_RECON: ErrorInfo(
        False, "3D reconstruction failed. Capture again with slower motion and more overlap."
    ),
    ErrorCode.ERR_NO_OVERLAP: ErrorInfo(
        False, "The views do not overlap. Capture the scene so neighbouring views share content."
    ),
    ErrorCode.ERR_CANNOT_LINK: ErrorInfo(
        False,
        "Those detections appear in the same photo, so they must be different objects and "
        "cannot be merged.",
    ),
    ErrorCode.ERR_INVALID_CORRECTION: ErrorInfo(
        False, "The correction refers to detections that do not exist or cannot be changed."
    ),
    ErrorCode.ERR_INVALID_INPUT: ErrorInfo(
        False, "Check the image paths, the prompt and the config keys, then call the tool again."
    ),
    ErrorCode.ERR_CHUNK_CORRUPT: ErrorInfo(
        True, "A chunk arrived damaged. Send that chunk again; finished chunks are kept."
    ),
    ErrorCode.ERR_UPLOAD_INCOMPLETE: ErrorInfo(
        False, "Some chunks are still missing. Resume the upload to send them, then finalize."
    ),
}


def is_retryable(code: ErrorCode | str) -> bool:
    return ERROR_INFO[ErrorCode(code)].retryable


def user_tip(code: ErrorCode | str) -> str:
    return ERROR_INFO[ErrorCode(code)].tip


class CountOnceError(RuntimeError):
    """Base for typed errors; carries ``code``, ``retryable`` and a user ``tip``."""

    code: ErrorCode = ErrorCode.ERR_NO_RECON

    def __init__(
        self, message: str, code: ErrorCode | str | None = None, tip: str | None = None
    ) -> None:
        if code is not None:
            self.code = ErrorCode(code)
        self.tip = tip if tip is not None else user_tip(self.code)
        super().__init__(f"{self.code.value}: {message}")

    @property
    def retryable(self) -> bool:
        return is_retryable(self.code)


class PlannedComponentError(NotImplementedError):
    """A named component that is accepted by ``Config`` but not implemented (docs/STATUS.md)."""


class MediaError(CountOnceError):
    """A permanent user/media error raised by ingest sanitization."""


class CorrectionError(CountOnceError):
    """A user correction was refused (cannot-link violation or invalid reference)."""

    code = ErrorCode.ERR_INVALID_CORRECTION


class GpuOomError(CountOnceError):
    """Typed, retryable infrastructure error raised once the whole OOM ladder is exhausted."""

    code = ErrorCode.ERR_GPU_OOM

    def __init__(self, message: str, tip: str = GPU_OOM_TIP, attempts: list[str] | None = None):
        super().__init__(message, tip=tip)
        self.attempts = attempts or []


# --- retry policy -----------------------------------------------------------------------------

MAX_ATTEMPTS = 4  # total tries, including the first
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0


@dataclass(frozen=True)
class RetryDecision:
    action: str  # "retry" | "dead_letter"
    backoff_s: float = 0.0

    @property
    def retry(self) -> bool:
        return self.action == "retry"


def retry_decision(
    error: CountOnceError | ErrorCode | str,
    attempt: int,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_s: float = BACKOFF_BASE_S,
    cap_s: float = BACKOFF_CAP_S,
) -> RetryDecision:
    """Decide what to do after ``attempt`` (1-based, the try that just failed) failed.

    Permanent errors dead-letter immediately. Retryable ones back off ``base * 2**(attempt-1)``
    seconds (capped) until ``max_attempts`` is reached, then dead-letter.
    """
    code = error.code if isinstance(error, CountOnceError) else ErrorCode(error)
    if not is_retryable(code) or attempt >= max_attempts:
        return RetryDecision("dead_letter")
    return RetryDecision("retry", min(cap_s, base_s * 2 ** (max(attempt, 1) - 1)))
