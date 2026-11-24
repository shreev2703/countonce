"""CountOnce: count anything in 3D from a few photos, without counting it twice."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Lazy so ``import countonce`` stays light (no torch, no pipeline modules).
_EXPORTS = {
    "run": ("countonce.pipeline", "run"),
    "Config": ("countonce.config", "Config"),
    "CountResult": ("countonce.types", "CountResult"),
    "load_result": ("countonce.explain", "load_result_dir"),
}

__all__ = ["Config", "CountResult", "__version__", "load_result", "run"]

if TYPE_CHECKING:
    from countonce.config import Config
    from countonce.explain import load_result_dir as load_result
    from countonce.pipeline import run
    from countonce.types import CountResult


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'countonce' has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
