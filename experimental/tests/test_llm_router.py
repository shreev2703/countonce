"""LLM router: preferred providers first, Claude when they are throttled or out of quota."""

from __future__ import annotations

import pytest
from llm import (
    AllRoutesFailed,
    ErrorKind,
    LLMClient,
    LLMResponse,
    LLMRouter,
    Route,
    backoff_delay,
    classify_error,
)

KEYS = {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "GEMINI_API_KEY": "g"}
GEMINI = Route("gemini", "g-model")
OPENAI = Route("openai", "o-model")
CLAUDE = Route("anthropic", "c-model")


class ProviderError(Exception):
    """Shaped like the SDK errors: an HTTP status, optionally a response with headers."""

    def __init__(self, msg: str, status: int | None = None, headers: dict[str, str] | None = None):
        super().__init__(msg)
        self.status_code = status
        if headers is not None:
            self.response = type("R", (), {"headers": headers})()


class Scripted:
    """Client that replays outcomes (an exception is raised, a string is returned)."""

    def __init__(self, provider: str, *outcomes: str | Exception) -> None:
        self.provider, self.model = provider, f"{provider}-model"
        self._outcomes = list(outcomes)
        self.calls = 0

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int = 1024
    ) -> LLMResponse:
        o = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(o, Exception):
            raise o
        return LLMResponse(o, self.provider, self.model)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def router(clients: dict[str, Scripted], preferred: list[Route], **kw: object) -> LLMRouter:
    def factory(provider: str, model: str) -> LLMClient:
        return clients[provider]

    kw.setdefault("env", KEYS)
    kw.setdefault("jitter", 0.0)
    return LLMRouter(preferred, CLAUDE, factory, **kw)  # type: ignore[arg-type]


# --- classification -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ProviderError("Too Many Requests", 429), ErrorKind.RATE_LIMIT),
        (ProviderError("Error code: 429 - insufficient_quota", 429), ErrorKind.QUOTA),
        (ProviderError("You exceeded your current quota, check billing", 429), ErrorKind.QUOTA),
        (ProviderError("RESOURCE_EXHAUSTED: Quota exceeded per minute", 429), ErrorKind.RATE_LIMIT),
        (ProviderError("credit balance is too low", 400), ErrorKind.QUOTA),
        (ProviderError("Overloaded", 529), ErrorKind.OVERLOADED),
        (ProviderError("bad gateway", 502), ErrorKind.OVERLOADED),
        (ProviderError("invalid api key", 401), ErrorKind.AUTH),
        (ProviderError("max_tokens too large", 400), ErrorKind.BAD_REQUEST),
        (ValueError("boom"), ErrorKind.OTHER),
    ],
)
def test_classify(exc: Exception, kind: ErrorKind) -> None:
    assert classify_error(exc).kind is kind


def test_retry_after_from_header_and_message() -> None:
    assert classify_error(ProviderError("x", 429, {"retry-after": "42"})).retry_after == 42
    assert classify_error(ProviderError("x", 429, {"retry-after-ms": "1500"})).retry_after == 1.5
    assert classify_error(ProviderError("Please retry in 34.5s", 429)).retry_after == 34.5
    assert classify_error(ProviderError("retry after 2 minutes", 429)).retry_after == 120
    assert classify_error(ProviderError("nope", 429)).retry_after is None


def test_backoff_delay() -> None:
    assert [backoff_delay(n, 30, 900) for n in (1, 2, 3, 4)] == [30, 60, 120, 240]
    assert backoff_delay(10, 30, 900) == 900  # capped
    assert backoff_delay(5, 30, 900, retry_after=7) == 7  # provider hint wins
    assert backoff_delay(1, 30, 900, jitter=0.5, rng=lambda: 1.0) == 45
    with pytest.raises(ValueError):
        backoff_delay(0, 30, 900)


# --- routing --------------------------------------------------------------------------------


def test_preferred_provider_used_when_healthy() -> None:
    g, c = Scripted("gemini", "from gemini"), Scripted("anthropic", "from claude")
    r = router({"gemini": g, "anthropic": c}, [GEMINI])
    assert r.complete("hi").text == "from gemini"
    assert r.last_route == GEMINI and c.calls == 0


@pytest.mark.parametrize(
    "err",
    [
        ProviderError("Too Many Requests", 429),
        ProviderError("insufficient_quota", 429),
        ProviderError("RESOURCE_EXHAUSTED", 429),
        ProviderError("overloaded", 529),
    ],
)
def test_falls_back_to_claude_on_throttle_quota_overload(err: Exception) -> None:
    g, c = Scripted("gemini", err), Scripted("anthropic", "from claude")
    r = router({"gemini": g, "anthropic": c}, [GEMINI])
    assert r.complete("hi").text == "from claude"
    assert r.last_route == CLAUDE
    assert r.attempts[0].route == GEMINI and r.attempts[0].kind is not None


def test_chain_gemini_then_openai_then_claude() -> None:
    g = Scripted("gemini", ProviderError("insufficient_quota", 429))
    o = Scripted("openai", ProviderError("Too Many Requests", 429))
    c = Scripted("anthropic", "from claude")
    r = router({"gemini": g, "openai": o, "anthropic": c}, [GEMINI, OPENAI])
    assert r.complete("hi").text == "from claude"
    assert [a.route for a in r.attempts] == [GEMINI, OPENAI]


def test_second_preferred_used_before_claude() -> None:
    g = Scripted("gemini", ProviderError("Too Many Requests", 429))
    o, c = Scripted("openai", "from openai"), Scripted("anthropic", "from claude")
    r = router({"gemini": g, "openai": o, "anthropic": c}, [GEMINI, OPENAI])
    assert r.complete("hi").text == "from openai" and c.calls == 0


def test_missing_key_is_skipped_without_building_client() -> None:
    built: list[str] = []

    def factory(provider: str, model: str) -> LLMClient:
        built.append(provider)
        return Scripted(provider, "ok")

    r = LLMRouter([GEMINI], CLAUDE, factory, env={"ANTHROPIC_API_KEY": "a"})
    assert r.complete("hi").text == "ok"
    assert built == ["anthropic"] and "GEMINI_API_KEY" in r.attempts[0].outcome


def test_non_fallback_errors_are_raised_not_hidden() -> None:
    for err in (ProviderError("invalid api key", 401), ProviderError("bad param", 400)):
        c = Scripted("anthropic", "never")
        r = router({"gemini": Scripted("gemini", err), "anthropic": c}, [GEMINI])
        with pytest.raises(ProviderError):
            r.complete("hi")
        assert c.calls == 0


def test_cooldown_skips_failed_route_then_retries_after_it_expires() -> None:
    clock = Clock()
    g = Scripted("gemini", ProviderError("Too Many Requests", 429, {"retry-after": "60"}), "back")
    c = Scripted("anthropic", "from claude")
    r = router({"gemini": g, "anthropic": c}, [GEMINI], clock=clock)
    assert r.complete("1").text == "from claude" and g.calls == 1
    assert r.cooldown_remaining(GEMINI) == 60

    clock.t += 30  # still cooling: Gemini is not even called
    assert r.complete("2").text == "from claude" and g.calls == 1
    assert "cooling down" in r.attempts[0].outcome

    clock.t += 31  # expired: Gemini is tried again and recovers
    assert r.complete("3").text == "back" and g.calls == 2
    assert r.cooldown_remaining(GEMINI) == 0


def test_repeated_failures_back_off_exponentially_and_quota_waits_longer() -> None:
    clock = Clock()
    g = Scripted("gemini", ProviderError("Too Many Requests", 429))
    r = router({"gemini": g, "anthropic": Scripted("anthropic", "ok")}, [GEMINI], clock=clock)
    waits = []
    for _ in range(3):
        r.complete("x")
        waits.append(r.cooldown_remaining(GEMINI))
        clock.t += waits[-1]
    assert waits == [30, 60, 120]

    q = router(
        {"gemini": Scripted("gemini", ProviderError("insufficient_quota", 429)),
         "anthropic": Scripted("anthropic", "ok")},
        [GEMINI], clock=clock,
    )  # fmt: skip
    q.complete("x")
    assert q.cooldown_remaining(GEMINI) == 600


def test_claude_is_tried_even_while_cooling_and_failure_surfaces_hint() -> None:
    clock = Clock()
    err = ProviderError("Too Many Requests", 429, {"retry-after": "5"})
    r = router(
        {"gemini": Scripted("gemini", err), "anthropic": Scripted("anthropic", err)},
        [GEMINI], clock=clock,
    )  # fmt: skip
    with pytest.raises(AllRoutesFailed) as ei:
        r.complete("x")
    assert [a.route for a in ei.value.attempts] == [GEMINI, CLAUDE]
    assert ei.value.retry_after == 5

    with pytest.raises(AllRoutesFailed):  # Claude still gets a chance although it cooled
        r.complete("x")
    assert r.attempts[0].outcome.startswith("skipped: cooling")
    assert r.attempts[1].outcome.startswith("failed")


def test_claude_as_preferred_is_not_tried_twice() -> None:
    c = Scripted("anthropic", ProviderError("Too Many Requests", 429))
    r = router({"anthropic": c}, [CLAUDE])
    with pytest.raises(AllRoutesFailed):
        r.complete("x")
    assert c.calls == 1


def test_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError):
        Route("mistral", "m")
    assert (
        Route("Claude", "m").provider == "anthropic" and Route("google", "m").provider == "gemini"
    )
