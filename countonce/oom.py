"""CUDA allocator setup and the OOM degradation ladder (PLAN 13 item 6c).

``configure_allocator`` must run before torch is first imported; the CLI entry calls it. The
ladder wraps one GPU stage and retries it on out-of-memory with progressively cheaper settings:

    rung 0  initial        the stage as configured
    rung 1  free_cache     collect garbage and empty the CUDA cache, retry unchanged
    rung 2  smaller_chunks retry reconstruction in chunks (``chunked_recon``), half the frames
    rung 3  lower_res      keep the chunks, step down through the resolution tiers

If every rung fails the ladder raises ``GpuOomError`` (code ``ERR_GPU_OOM``). This module never
imports torch, so the ladder logic is testable on CPU.
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from countonce.errors import GPU_OOM_TIP, ErrorCode, GpuOomError  # noqa: F401  (re-export)

log = logging.getLogger(__name__)

ALLOC_ENV = "PYTORCH_CUDA_ALLOC_CONF"
ALLOC_SETTING = "expandable_segments:True"
ERR_GPU_OOM = ErrorCode.ERR_GPU_OOM.value
USER_TIP = GPU_OOM_TIP
RESOLUTION_TIERS: tuple[float, ...] = (0.75, 0.5)
MIN_CHUNK = 3  # must exceed the default chunk overlap floor (2)

T = TypeVar("T")


def is_wsl() -> bool:
    """True under Windows Subsystem for Linux (its kernel release names Microsoft)."""
    return "microsoft" in platform.release().lower()


def configure_allocator() -> bool:
    """Set ``expandable_segments:True`` in ``PYTORCH_CUDA_ALLOC_CONF`` if torch is not loaded yet.

    Keeps any other options the user already set, and leaves an explicit
    ``expandable_segments`` choice alone. Does nothing on WSL, where the setting made the
    five-view VGGT run fail with ``CUDA driver error: device not ready`` on an 8 GB card.
    Returns True unless the call changed nothing for one of those two environment reasons
    (torch already loaded, or WSL).
    """
    if "torch" in sys.modules:
        log.warning("torch was imported before %s was set; the setting has no effect", ALLOC_ENV)
        return False
    if is_wsl():
        return False
    current = os.environ.get(ALLOC_ENV, "")
    if "expandable_segments" not in current:
        os.environ[ALLOC_ENV] = f"{current},{ALLOC_SETTING}" if current else ALLOC_SETTING
    return True


class RungUnavailable(Exception):
    """Raised by a stage when it cannot honour a rung (e.g. no resolution control); skipped."""


@dataclass(frozen=True)
class Attempt:
    """Settings for one try. ``chunk_size`` None = unchunked; ``resolution_scale`` 1.0 = native."""

    rung: int
    name: str
    chunk_size: int | None = None
    resolution_scale: float = 1.0


@dataclass(frozen=True)
class LadderResult(Generic[T]):
    value: T
    rung: int  # 0 = no fallback used; matches ``oom_fallback_rung``
    name: str
    attempts: tuple[str, ...]  # names tried, in order, including the one that succeeded


def is_oom(exc: BaseException) -> bool:
    """True for CUDA OOM without importing torch (class-name and message match)."""
    if any(c.__name__ == "OutOfMemoryError" for c in type(exc).__mro__):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def free_cache() -> None:
    """Collect garbage and release cached CUDA blocks when torch/CUDA are active."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def build_attempts(
    n_frames: int, resolution_tiers: tuple[float, ...] = RESOLUTION_TIERS
) -> list[Attempt]:
    chunk = max(MIN_CHUNK, n_frames // 2)
    out = [
        Attempt(0, "initial"),
        Attempt(1, "free_cache"),
        Attempt(2, "smaller_chunks", chunk_size=chunk),
    ]
    out += [
        Attempt(3, f"lower_res_{s:g}", chunk_size=chunk, resolution_scale=s)
        for s in resolution_tiers
    ]
    return out


def run_with_oom_ladder(
    stage: Callable[[Attempt], T],
    *,
    stage_name: str = "stage",
    n_frames: int = 0,
    resolution_tiers: tuple[float, ...] = RESOLUTION_TIERS,
    free: Callable[[], None] = free_cache,
) -> LadderResult[T]:
    """Run ``stage(attempt)`` down the ladder until it stops raising OOM.

    The cache is freed before every retry (rung 1 is that retry with no other change). Non-OOM
    exceptions propagate unchanged. The rung that succeeded is logged.
    """
    tried: list[str] = []
    for attempt in build_attempts(n_frames, resolution_tiers):
        if attempt.rung > 0:
            free()
        try:
            value = stage(attempt)
        except RungUnavailable:
            log.info(
                "%s: rung %d (%s) unavailable, skipping", stage_name, attempt.rung, attempt.name
            )
            continue
        except Exception as exc:
            if not is_oom(exc):
                raise
            tried.append(attempt.name)
            log.warning("%s: OOM at rung %d (%s)", stage_name, attempt.rung, attempt.name)
            continue
        tried.append(attempt.name)
        if attempt.rung:
            log.warning(
                "%s: succeeded at fallback rung %d (%s)", stage_name, attempt.rung, attempt.name
            )
        else:
            log.info("%s: succeeded at rung 0 (no fallback)", stage_name)
        return LadderResult(value, attempt.rung, attempt.name, tuple(tried))
    raise GpuOomError(
        f"{stage_name} ran out of GPU memory at every rung ({', '.join(tried)})", attempts=tried
    )
