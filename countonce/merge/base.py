"""Shared associator interface (docs/PLAN.md §8.7).

v1 (clustering), v2 (geometric matching) and the future v3 (learned) all expose
``associate(lifted, recon, frames) -> ClusterResult``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from countonce.types import Frame, Instance, LiftedDetection, MatchEvidence, Recon


@dataclass
class ClusterResult:
    instances: list[Instance]
    detection_status: dict[str, str] = field(default_factory=dict)  # new/duplicate/rejected
    reject_reasons: dict[str, str] = field(default_factory=dict)
    evidence: list[MatchEvidence] = field(default_factory=list)


@runtime_checkable
class Associator(Protocol):
    name: str

    def associate(
        self,
        lifted: Sequence[LiftedDetection],
        recon: Recon | None = None,
        frames: Mapping[int, Frame] | None = None,
    ) -> ClusterResult: ...
