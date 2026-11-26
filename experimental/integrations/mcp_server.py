"""MCP tool for LLM agents (PLAN section 15): ``count_objects(images, prompt, config)``.

``count_objects`` is a pure function over :func:`countonce.pipeline.run` and never raises for
user mistakes: it returns ``{"ok": False, "error": {code, message, tip, retryable}}``. The MCP
server is a thin wrapper that needs the optional ``mcp`` package (``pip install countonce[mcp]``);
it is imported lazily so this module always imports cleanly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from countonce.config import Config
from countonce.errors import CountOnceError, ErrorCode, is_retryable, user_tip
from countonce.pipeline import run

TOOL_NAME = "count_objects"
TOOL_DESCRIPTION = (
    "Count distinct objects matching a text prompt across several overlapping photos of one "
    "scene, counting each physical object once. Returns the count, an interval and warnings."
)


def _error(code: ErrorCode, message: str, tip: str | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code.value,
            "message": message,
            "tip": tip if tip is not None else user_tip(code),
            "retryable": is_retryable(code),
        },
    }


def count_objects(
    images: list[str],
    prompt: str,
    config: dict[str, Any] | None = None,
    *,
    detector: Any = None,
    reconstructor: Any = None,
) -> dict[str, Any]:
    """Count ``prompt`` objects in ``images``; ``detector``/``reconstructor`` are test overrides.

    Success: ``{"ok": True, "count", "interval": {low, high, level, method}, "warnings",
    "result_path", "zero_count"}``. ``config`` takes any :class:`Config` key; ``out_dir``
    defaults to a fresh temporary directory so ``result_path`` always exists.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return _error(ErrorCode.ERR_INVALID_INPUT, "prompt must be a non-empty string")
    if not isinstance(images, list) or not images or not all(isinstance(p, str) for p in images):
        return _error(ErrorCode.ERR_INVALID_INPUT, "images must be a non-empty list of paths")
    missing = [p for p in images if not Path(p).expanduser().exists()]
    if missing:
        return _error(ErrorCode.ERR_INVALID_INPUT, f"path not found: {', '.join(missing)}")
    try:
        cfg = Config(**(config or {}))
    except (ValidationError, TypeError) as exc:
        return _error(ErrorCode.ERR_INVALID_INPUT, f"invalid config: {exc}")
    if not (config or {}).get("out_dir"):
        cfg = cfg.with_overrides(out_dir=Path(tempfile.mkdtemp(prefix="countonce-mcp-")))
    try:
        res = run(
            [Path(p).expanduser() for p in images],
            prompt.strip(),
            cfg,
            detector=detector,
            reconstructor=reconstructor,
        )
    except CountOnceError as exc:
        return _error(exc.code, str(exc), exc.tip)
    except (FileNotFoundError, ValueError) as exc:
        return _error(ErrorCode.ERR_INVALID_INPUT, str(exc))
    iv = res.interval
    return {
        "ok": True,
        "count": res.count,
        "zero_count": res.zero_count,
        "interval": {"low": iv.low, "high": iv.high, "level": iv.level, "method": iv.method},
        "warnings": list(res.warnings),
        "result_path": str((Path(cfg.out_dir) / "result.json").resolve()),
    }


def build_server() -> Any:
    """Create the FastMCP server with ``count_objects`` registered (needs the ``mcp`` extra)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The MCP server needs the optional 'mcp' package. "
            "Install it with: pip install 'countonce[mcp]'"
        ) from exc
    server = FastMCP("countonce")
    server.tool(name=TOOL_NAME, description=TOOL_DESCRIPTION)(count_objects)
    return server


def main() -> None:
    """Console script ``countonce-mcp``: serve over stdio."""
    build_server().run()


if __name__ == "__main__":
    main()
