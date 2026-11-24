"""Versioned prompt files: ``<name>.v<N>.txt`` next to this module, with an ``{object}`` slot."""

from __future__ import annotations

import re
from pathlib import Path

PROMPT_DIR = Path(__file__).parent
PROMPTS = {"count": "count_unique", "enumerate": "locate_label"}  # short name -> file stem


def load_prompt(name: str, version: int = 1) -> tuple[str, str]:
    """Return ``(template, version_tag)`` for a prompt name (``count`` or ``enumerate``)."""
    if name not in PROMPTS:
        raise ValueError(f"unknown VLM prompt {name!r}; choose from {sorted(PROMPTS)}")
    path = PROMPT_DIR / f"{PROMPTS[name]}.v{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"prompt file missing: {path}")
    return path.read_text().strip(), f"{name}.v{version}"


def render(template: str, obj: str) -> str:
    return re.sub(r"\{object\}", obj, template)
