"""Locate config files that default to a CWD-relative ``configs/...`` path.

Lookup order: the path as given (relative to the CWD), then the copy shipped inside the wheel
(``countonce/configs``), then a clear error. Only configs the installed runtime needs ship as
package data; benchmark/training configs live in the repo checkout only.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

PACKAGED = ("alerts.yaml", "colmap.yaml", "cost.yaml", "uploads.yaml")


def packaged_config(name: str) -> Path | None:
    ref = Path(str(resources.files("countonce") / "configs" / name))
    return ref if ref.is_file() else None


def resolve_config(path: Path | str) -> Path:
    """Return an existing file for ``path``; raise ``FileNotFoundError`` with guidance if none."""
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_absolute() and p.parts[:1] == ("configs",) and len(p.parts) == 2:
        found = packaged_config(p.name)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"config {p} not found in the current directory, and it is not shipped with the "
            f"installed package (packaged: {', '.join(PACKAGED)}). Run from a repository "
            "checkout or pass an explicit path."
        )
    raise FileNotFoundError(f"config not found: {p}")
