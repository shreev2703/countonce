# Ported from ~/vggtProject (vggt-sparse-recon): vggt_bench/methods/base.py. Adapted for countonce.
"""Common interface so the benchmark can swap reconstruction methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from countonce.recon.result import Recon


class ReconMethod(Protocol):
    name: str

    def reconstruct(self, image_paths: list[Path]) -> Recon: ...

    # Every result file records the settings that produced it, so the sweep has
    # to reach this through the interface rather than a concrete class.
    def config(self) -> dict[str, Any]: ...
