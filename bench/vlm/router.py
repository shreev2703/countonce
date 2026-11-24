"""Ordered multi-provider routing with fallback over the ``VLMClient`` adapters.

``FallbackVLMClient`` tries ``(provider, model)`` routes in order and returns the first
success. A route is skipped when its API key variable (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, ``GEMINI_API_KEY``) is unset, and a failed call moves on to the next route.
It satisfies the ``VLMClient`` protocol, so it can stand in for a single client anywhere.

Routing never bypasses the paid-call gate: ``bench.vlm.harness.paid_gate`` still decides whether
a provider may be called at all, and the spend projection there uses the provider you named
first. Nothing here imports an SDK or touches the network until ``complete`` is called.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bench.vlm.client import PROVIDERS, VLMClient, VLMResponse, build_client

log = logging.getLogger(__name__)

# Short aliases accepted in route specs; canonical names are the PROVIDERS keys.
_ALIASES = {"claude": "anthropic", "gemini": "google"}


@dataclass(frozen=True)
class Route:
    provider: str
    model: str


@dataclass(frozen=True)
class Attempt:
    route: Route
    error: str  # exception type and message, or "skipped: <reason>"


class AllRoutesFailed(RuntimeError):
    """Every route was skipped or raised. ``attempts`` lists why, in order."""

    def __init__(self, attempts: Sequence[Attempt]) -> None:
        self.attempts = list(attempts)
        detail = "; ".join(f"{a.route.provider}:{a.route.model} -> {a.error}" for a in attempts)
        super().__init__(f"all routes failed: {detail or 'no routes configured'}")


def parse_routes(spec: str) -> list[Route]:
    """Parse ``"anthropic:MODEL,openai:MODEL,gemini:MODEL"`` into routes (order is priority).

    The model id is whatever the provider accepts; none is assumed here.
    """
    routes: list[Route] = []
    for item in (p.strip() for p in spec.split(",")):
        if not item:
            continue
        provider, sep, model = item.partition(":")
        provider = _ALIASES.get(provider.strip().lower(), provider.strip().lower())
        if not sep or not model.strip():
            raise ValueError(f"route {item!r} must look like provider:model")
        if provider not in PROVIDERS or provider == "fake":
            raise ValueError(f"unknown provider {provider!r} in route {item!r}")
        routes.append(Route(provider, model.strip()))
    if not routes:
        raise ValueError("no routes given")
    return routes


class FallbackVLMClient:
    """Try each route in order; return the first successful ``VLMResponse``."""

    provider = "router"

    def __init__(
        self,
        routes: Sequence[Route],
        client_factory: Callable[[str, str], VLMClient] = build_client,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not routes:
            raise ValueError("at least one route is required")
        self.routes = list(routes)
        self.model = self.routes[0].model
        self._factory = client_factory
        self._env = os.environ if env is None else env
        self._clients: dict[Route, VLMClient] = {}
        self.attempts: list[Attempt] = []  # failures from the most recent complete() call
        self.last_route: Route | None = None  # route that served the most recent call

    def _skip_reason(self, route: Route) -> str | None:
        var = PROVIDERS[route.provider].env_var
        if var and not self._env.get(var):
            return f"skipped: {var} is not set"
        return None

    def complete(self, prompt: str, images: Sequence[Path], max_output_tokens: int) -> VLMResponse:
        self.attempts = []
        for route in self.routes:
            reason = self._skip_reason(route)
            if reason:
                self.attempts.append(Attempt(route, reason))
                continue
            try:
                client = self._clients.get(route) or self._factory(route.provider, route.model)
                self._clients[route] = client
                resp = client.complete(prompt, images, max_output_tokens)
            except Exception as e:  # noqa: BLE001  # any provider error means try the next route
                log.warning("route %s:%s failed: %s", route.provider, route.model, e)
                self.attempts.append(Attempt(route, f"{type(e).__name__}: {e}"))
                continue
            self.last_route = route
            return resp
        raise AllRoutesFailed(self.attempts)


def build_router(
    spec: str,
    client_factory: Callable[[str, str], VLMClient] = build_client,
    env: Mapping[str, str] | None = None,
) -> FallbackVLMClient:
    """``FallbackVLMClient`` from a route spec string (see ``parse_routes``)."""
    return FallbackVLMClient(parse_routes(spec), client_factory, env)
