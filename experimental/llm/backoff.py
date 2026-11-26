"""Exponential backoff with jitter that defers to a provider's own retry hint."""

from __future__ import annotations

import random
from collections.abc import Callable


def backoff_delay(
    attempt: int,
    base: float,
    cap: float,
    retry_after: float | None = None,
    jitter: float = 0.0,
    rng: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before retry number ``attempt`` (1 is the first retry).

    ``base * 2**(attempt-1)``, or the provider's ``retry_after`` when it gave one, then up to
    ``jitter`` (a fraction) extra so parallel workers do not retry in lockstep. Never above ``cap``.
    """
    if attempt < 1:
        raise ValueError("attempt starts at 1")
    delay = base * 2.0 ** min(attempt - 1, 62) if retry_after is None else max(retry_after, 0.0)
    delay += delay * jitter * rng()
    return min(cap, delay)
