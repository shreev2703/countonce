"""Text-completion clients for Anthropic, OpenAI and Gemini behind one small protocol.

SDKs are imported lazily inside each constructor, so importing this module never needs them and
never touches the network. API keys come from ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY`` and
``GEMINI_API_KEY``. Vision calls live in ``bench.vlm.client``; this layer is text only.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Accepted spellings -> canonical provider name.
ALIASES = {"claude": "anthropic", "google": "gemini"}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int = 1024
    ) -> LLMResponse: ...


def canonical(provider: str) -> str:
    p = provider.strip().lower()
    return ALIASES.get(p, p)


def api_key_var(provider: str) -> str:
    """Name of the environment variable holding the provider's API key."""
    p = canonical(provider)
    if p == "anthropic":
        return "ANTHROPIC_API_KEY"
    if p == "openai":
        return "OPENAI_API_KEY"
    if p == "gemini":
        return "GEMINI_API_KEY"
    raise ValueError(f"unknown provider {provider!r}")


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, model: str) -> None:
        import anthropic  # lazy: optional dependency

        self.model = model
        self._c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int = 1024
    ) -> LLMResponse:
        kwargs = {"system": system} if system else {}
        msg = self._c.messages.create(
            model=self.model,
            max_tokens=max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text, self.provider, self.model, msg.usage.input_tokens, msg.usage.output_tokens
        )


class OpenAILLM:
    provider = "openai"

    def __init__(self, model: str) -> None:
        import openai  # lazy: optional dependency

        self.model = model
        self._c = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int = 1024
    ) -> LLMResponse:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        r = self._c.chat.completions.create(
            model=self.model, max_completion_tokens=max_output_tokens, messages=messages
        )
        u = r.usage
        return LLMResponse(
            r.choices[0].message.content or "",
            self.provider,
            self.model,
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
        )


class GeminiLLM:
    provider = "gemini"

    def __init__(self, model: str) -> None:
        from google import genai  # lazy: optional dependency

        self.model = model
        self._c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int = 1024
    ) -> LLMResponse:
        from google.genai import types

        r = self._c.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens, system_instruction=system
            ),
        )
        u = r.usage_metadata
        return LLMResponse(
            r.text or "",
            self.provider,
            self.model,
            getattr(u, "prompt_token_count", 0) or 0,
            getattr(u, "candidates_token_count", 0) or 0,
        )


_ADAPTERS: dict[str, Callable[[str], LLMClient]] = {
    "anthropic": AnthropicLLM,
    "openai": OpenAILLM,
    "gemini": GeminiLLM,
}


def build_llm_client(provider: str, model: str) -> LLMClient:
    """Construct a network client for ``provider`` (SDK imported here, on first use)."""
    p = canonical(provider)
    if p not in _ADAPTERS:
        raise ValueError(f"no adapter for provider {provider!r}")
    return _ADAPTERS[p](model)
