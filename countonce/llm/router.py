"""Prefer Gemini / OpenAI, fall back to Claude when a key is exhausted or throttled.

``LLMRouter`` tries the preferred routes in order, then the fallback route (Claude by default).
A preferred route is passed over when

* its API key variable is unset,
* it raised a rate-limit (429), quota/credit-exhausted, or provider-overload (5xx/529) error, or
* it is still cooling down from one of those errors on an earlier call.

Cooldowns follow the provider's ``Retry-After`` hint when there is one, otherwise exponential
backoff (longer for quota errors, since credit does not come back in a minute). Errors that a
different provider cannot fix (bad request, bad key) are raised to the caller rather than hidden.
The fallback route is always attempted, even while cooling, so a call only fails when Claude does.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from countonce.llm.backoff import backoff_delay
from countonce.llm.errors import FALLBACK_KINDS, Classified, ErrorKind, classify_error
from countonce.llm.providers import (
    LLMClient,
    LLMResponse,
    api_key_var,
    build_llm_client,
    canonical,
)

log = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class Route:
    provider: str
    model: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", canonical(self.provider))
        api_key_var(self.provider)  # rejects unknown providers early


@dataclass(frozen=True)
class Attempt:
    route: Route
    outcome: str  # "skipped: ..." or "failed: ..."
    kind: ErrorKind | None = None
    retry_after: float | None = None


class AllRoutesFailed(RuntimeError):
    """No route produced a response. ``attempts`` says why, in order."""

    def __init__(self, attempts: Sequence[Attempt]) -> None:
        self.attempts = list(attempts)
        hints = [a.retry_after for a in self.attempts if a.retry_after is not None]
        self.retry_after: float | None = min(hints) if hints else None
        detail = "; ".join(f"{a.route.provider}:{a.route.model} {a.outcome}" for a in attempts)
        super().__init__(f"all routes failed: {detail or 'no routes configured'}")


@dataclass
class _Health:
    failures: int = 0
    until: float = 0.0  # monotonic time before which the route is skipped


@dataclass
class LLMRouter:
    preferred: Sequence[Route]
    fallback: Route = field(default_factory=lambda: Route("anthropic", DEFAULT_CLAUDE_MODEL))
    client_factory: Callable[[str, str], LLMClient] = build_llm_client
    env: Mapping[str, str] | None = None
    clock: Callable[[], float] = time.monotonic
    rate_base: float = 30.0  # first cooldown after a 429 / overload with no hint, doubling
    quota_base: float = 600.0  # first cooldown after a quota error with no hint, doubling
    cooldown_cap: float = 3600.0
    jitter: float = 0.1
    attempts: list[Attempt] = field(default_factory=list, init=False)  # last call's skips/failures
    last_route: Route | None = field(default=None, init=False)  # route that served the last call

    def __post_init__(self) -> None:
        self._env: Mapping[str, str] = os.environ if self.env is None else self.env
        self._clients: dict[Route, LLMClient] = {}
        self._health: dict[Route, _Health] = {}

    def cooldown_remaining(self, route: Route) -> float:
        h = self._health.get(route)
        return max(0.0, h.until - self.clock()) if h else 0.0

    def _record_failure(self, route: Route, c: Classified) -> float:
        h = self._health.setdefault(route, _Health())
        h.failures += 1
        base = self.quota_base if c.kind is ErrorKind.QUOTA else self.rate_base
        wait = backoff_delay(h.failures, base, self.cooldown_cap, c.retry_after, self.jitter)
        h.until = self.clock() + wait
        return wait

    def _try(self, route: Route, prompt: str, system: str | None, max_tokens: int) -> LLMResponse:
        client = self._clients.get(route)
        if client is None:
            client = self._clients[route] = self.client_factory(route.provider, route.model)
        resp = client.complete(prompt, system=system, max_output_tokens=max_tokens)
        self._health.pop(route, None)  # a success clears the backoff history
        self.last_route = route
        return resp

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int = 1024
    ) -> LLMResponse:
        self.attempts = []
        routes = [*self.preferred]
        if self.fallback not in routes:
            routes.append(self.fallback)
        for route in routes:
            is_last_resort = route == self.fallback
            var = api_key_var(route.provider)
            if not self._env.get(var):
                self.attempts.append(Attempt(route, f"skipped: {var} is not set"))
                continue
            left = self.cooldown_remaining(route)
            if left > 0 and not is_last_resort:
                self.attempts.append(
                    Attempt(route, f"skipped: cooling down for {left:.0f}s", retry_after=left)
                )
                continue
            try:
                return self._try(route, prompt, system, max_output_tokens)
            except Exception as e:
                c = classify_error(e)
                if c.kind not in FALLBACK_KINDS:
                    raise
                wait = self._record_failure(route, c)
                log.warning(
                    "%s:%s %s (status %s); cooling down %.0fs",
                    route.provider, route.model, c.kind.value, c.status, wait,
                )  # fmt: skip
                self.attempts.append(
                    Attempt(route, f"failed: {c.kind.value}: {c.message[:200]}", c.kind, wait)
                )
        raise AllRoutesFailed(self.attempts)
