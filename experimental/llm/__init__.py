"""Text-LLM utility layer: provider clients, error classification, and fallback routing."""

from __future__ import annotations

from llm.backoff import backoff_delay
from llm.errors import FALLBACK_KINDS, Classified, ErrorKind, classify_error
from llm.providers import LLMClient, LLMResponse, build_llm_client
from llm.router import (
    DEFAULT_CLAUDE_MODEL,
    AllRoutesFailed,
    Attempt,
    LLMRouter,
    Route,
)

__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "FALLBACK_KINDS",
    "AllRoutesFailed",
    "Attempt",
    "Classified",
    "ErrorKind",
    "LLMClient",
    "LLMResponse",
    "LLMRouter",
    "Route",
    "backoff_delay",
    "build_llm_client",
    "classify_error",
]
