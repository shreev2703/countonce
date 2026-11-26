"""Classify provider SDK exceptions without importing any SDK.

OpenAI, Anthropic and Google (``google-genai``) all raise exceptions that carry an HTTP status in
``status_code`` or ``code`` plus a message. We read those attributes and the message text, so the
router works with whichever SDKs happen to be installed and with fakes in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ErrorKind(StrEnum):
    RATE_LIMIT = "rate_limit"  # 429 / too many requests: retry after a short wait
    QUOTA = "quota"  # key out of credit or quota: retry only after a long wait
    OVERLOADED = "overloaded"  # 5xx / 529: provider side, usually brief
    AUTH = "auth"  # 401 / 403: a configuration problem, never hidden by fallback
    BAD_REQUEST = "bad_request"  # 400 / 404 / 422: our request is wrong, fallback won't help
    OTHER = "other"


# Kinds where trying a different provider is the right response.
FALLBACK_KINDS = frozenset({ErrorKind.RATE_LIMIT, ErrorKind.QUOTA, ErrorKind.OVERLOADED})


@dataclass(frozen=True)
class Classified:
    kind: ErrorKind
    status: int | None
    retry_after: float | None  # seconds the provider asked us to wait, if it said
    message: str


_QUOTA = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "billing",
    "quota",
)
_RATE = ("rate limit", "rate_limit", "ratelimit", "too many requests", "resource_exhausted")
_OVERLOADED = ("overloaded", "service unavailable", "internal server error", "bad gateway")

_RETRY_TEXT = re.compile(
    r"(?:retry(?:ing)?\s*(?:in|after)|retrydelay['\"]?\s*[:=])\s*['\"]?([0-9]+(?:\.[0-9]+)?)"
    r"\s*(ms|s|sec|seconds|m|min|minutes)?",
    re.IGNORECASE,
)
_UNIT = {"ms": 0.001, "m": 60.0, "min": 60.0, "minutes": 60.0}


def _status(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "code", "http_status"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 599:
                return v
    return None


def _retry_after(exc: BaseException, text: str) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            ms = headers.get("retry-after-ms")
            if ms is not None:
                return max(0.0, float(ms) / 1000.0)
            s = headers.get("retry-after")
            if s is not None:
                return max(0.0, float(s))
        except (TypeError, ValueError, AttributeError):
            pass  # HTTP-date form or an unusable mock: fall through to the message text
    m = _RETRY_TEXT.search(text)
    if m:
        return float(m.group(1)) * _UNIT.get((m.group(2) or "s").lower(), 1.0)
    return None


def classify_error(exc: BaseException) -> Classified:
    """Map an exception from any provider SDK to an ``ErrorKind``."""
    status = _status(exc)
    parts = [type(exc).__name__, str(exc)]
    parts += [str(getattr(exc, a, "")) for a in ("code", "status", "type")]
    text = " ".join(parts).lower()
    retry = _retry_after(exc, text)

    if any(k in text for k in _QUOTA) and "per minute" not in text:
        kind = ErrorKind.QUOTA
    elif status == 429 or any(k in text for k in _RATE):
        kind = ErrorKind.RATE_LIMIT
    elif status in (500, 502, 503, 504, 529) or any(k in text for k in _OVERLOADED):
        kind = ErrorKind.OVERLOADED
    elif status in (401, 403):
        kind = ErrorKind.AUTH
    elif status in (400, 404, 422):
        kind = ErrorKind.BAD_REQUEST
    else:
        kind = ErrorKind.OTHER
    return Classified(kind, status, retry, str(exc))
