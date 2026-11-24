"""Fallback routing across providers, with fake clients and no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.vlm.client import FakeVLMClient, VLMClient, VLMResponse
from bench.vlm.router import AllRoutesFailed, Route, build_router, parse_routes

ALL_KEYS = {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "GEMINI_API_KEY": "g"}


class Boom:
    provider = "boom"
    model = "boom-1"

    def complete(self, prompt: str, images: list[Path], max_output_tokens: int) -> VLMResponse:
        raise TimeoutError("upstream timed out")


def factory(clients: dict[str, VLMClient]):  # noqa: ANN201
    def make(provider: str, model: str) -> VLMClient:
        return clients[provider]

    return make


def test_parse_routes_aliases_and_errors() -> None:
    assert parse_routes("claude:m1, gemini:m2,openai:m3") == [
        Route("anthropic", "m1"),
        Route("google", "m2"),
        Route("openai", "m3"),
    ]
    for bad in ("", "openai", "openai:", "mistral:x", "fake:x"):
        with pytest.raises(ValueError):
            parse_routes(bad)


def test_first_route_wins() -> None:
    a, o = FakeVLMClient(["A"]), FakeVLMClient(["O"])
    r = build_router("anthropic:m1,openai:m2", factory({"anthropic": a, "openai": o}), env=ALL_KEYS)
    assert r.complete("p", [], 10).text == "A"
    assert r.last_route == Route("anthropic", "m1") and not o.calls


def test_falls_back_on_error() -> None:
    o = FakeVLMClient(["O"])
    r = build_router(
        "anthropic:m1,openai:m2", factory({"anthropic": Boom(), "openai": o}), env=ALL_KEYS
    )
    assert r.complete("p", [], 10).text == "O"
    assert r.last_route == Route("openai", "m2")
    assert "TimeoutError" in r.attempts[0].error


def test_skips_route_without_key_and_never_builds_client() -> None:
    built: list[str] = []

    def make(provider: str, model: str) -> VLMClient:
        built.append(provider)
        return FakeVLMClient(["G"])

    r = build_router("openai:m1,gemini:m2", make, env={"GEMINI_API_KEY": "g"})
    assert r.complete("p", [], 10).text == "G"
    assert built == ["google"]
    assert "OPENAI_API_KEY" in r.attempts[0].error


def test_all_fail_raises_with_attempts() -> None:
    r = build_router("anthropic:m1,openai:m2", factory({"anthropic": Boom()}), env={})
    with pytest.raises(AllRoutesFailed) as ei:
        r.complete("p", [], 10)
    assert len(ei.value.attempts) == 2
    assert "ANTHROPIC_API_KEY" in str(ei.value)


def test_clients_are_cached_across_calls() -> None:
    built: list[str] = []

    def make(provider: str, model: str) -> VLMClient:
        built.append(provider)
        return FakeVLMClient(["x"])

    r = build_router("anthropic:m1", make, env=ALL_KEYS)
    r.complete("p", [], 10)
    r.complete("p", [], 10)
    assert built == ["anthropic"]
