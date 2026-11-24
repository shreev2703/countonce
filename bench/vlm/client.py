"""VLM client protocol, a deterministic fake, and lazily-imported provider adapters.

Nothing here imports a provider SDK or touches the network at import time. ``build_client`` is
only called by the harness after the paid-call gate has passed (see ``bench.vlm.harness``).
"""

from __future__ import annotations

import base64
import mimetypes
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class VLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self, prompt: str, images: Sequence[Path], max_output_tokens: int
    ) -> VLMResponse: ...


@dataclass(frozen=True)
class ProviderInfo:
    env_var: str | None  # API key variable; None for the fake
    max_images: int  # per-request image cap we enforce (conservative defaults, verify)
    usd_per_mtok_in: float  # estimate used only for the spend cap and the cost column
    usd_per_mtok_out: float
    tokens_per_image: int  # rough input tokens charged per image, for the pre-call projection


# Conservative planning numbers, not quotes: confirm against the provider's price list before a
# paid run, and override per method with ``price_in`` / ``price_out`` in the bench config.
PROVIDERS: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo("ANTHROPIC_API_KEY", 100, 5.0, 25.0, 1600),
    "google": ProviderInfo("GEMINI_API_KEY", 3000, 2.0, 12.0, 1100),
    "openai": ProviderInfo("OPENAI_API_KEY", 500, 5.0, 20.0, 1500),
    "fake": ProviderInfo(None, 100, 1.0, 2.0, 1000),
}


class FakeVLMClient:
    """Scripted client for tests and dry runs: replays ``replies`` (cycling) and records calls."""

    provider = "fake"

    def __init__(
        self,
        replies: Sequence[str | VLMResponse] = ("ANSWER: 7",),
        model: str = "fake-1",
        tokens_in: int = 100,
        tokens_out: int = 20,
    ) -> None:
        self.model = model
        self._replies = list(replies)
        self._tokens = (tokens_in, tokens_out)
        self.calls: list[tuple[str, list[Path], int]] = []

    def complete(
        self, prompt: str, images: Sequence[Path], max_output_tokens: int
    ) -> VLMResponse:
        self.calls.append((prompt, list(images), max_output_tokens))
        r = self._replies[(len(self.calls) - 1) % len(self._replies)]
        return r if isinstance(r, VLMResponse) else VLMResponse(r, *self._tokens)


def _b64(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return mime, base64.b64encode(path.read_bytes()).decode()


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str) -> None:
        import anthropic  # lazy: optional dependency

        self.model = model
        self._c = anthropic.Anthropic(api_key=os.environ[PROVIDERS["anthropic"].env_var or ""])

    def complete(
        self, prompt: str, images: Sequence[Path], max_output_tokens: int
    ) -> VLMResponse:
        content: list[dict] = []
        for p in images:
            mime, data = _b64(p)
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
            )
        content.append({"type": "text", "text": prompt})
        msg = self._c.messages.create(
            model=self.model,
            max_tokens=max_output_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return VLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens)


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str) -> None:
        import openai  # lazy: optional dependency

        self.model = model
        self._c = openai.OpenAI(api_key=os.environ[PROVIDERS["openai"].env_var or ""])

    def complete(
        self, prompt: str, images: Sequence[Path], max_output_tokens: int
    ) -> VLMResponse:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in images:
            mime, data = _b64(p)
            url = f"data:{mime};base64,{data}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        r = self._c.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_output_tokens,
            messages=[{"role": "user", "content": content}],
        )
        u = r.usage
        return VLMResponse(
            r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0),
            getattr(u, "completion_tokens", 0),
        )  # fmt: skip


class GoogleClient:
    provider = "google"

    def __init__(self, model: str) -> None:
        from google import genai  # lazy: optional dependency

        self.model = model
        self._genai = genai
        self._c = genai.Client(api_key=os.environ[PROVIDERS["google"].env_var or ""])

    def complete(
        self, prompt: str, images: Sequence[Path], max_output_tokens: int
    ) -> VLMResponse:
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=p.read_bytes(), mime_type=_b64(p)[0]) for p in images
        ]
        r = self._c.models.generate_content(
            model=self.model,
            contents=[*parts, prompt],
            config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
        )
        u = r.usage_metadata
        return VLMResponse(
            r.text or "", getattr(u, "prompt_token_count", 0) or 0,
            getattr(u, "candidates_token_count", 0) or 0,
        )  # fmt: skip


_ADAPTERS: dict[str, Callable[[str], VLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "google": GoogleClient,
}


def build_client(provider: str, model: str) -> VLMClient:
    """Construct a network client. Only the harness calls this, after the paid gate passed."""
    if provider not in _ADAPTERS:
        raise ValueError(f"no network adapter for provider {provider!r}")
    return _ADAPTERS[provider](model)
