"""Load, validate and convert ``identity_labels.json`` (PLAN section 9.1 identity labels).

File format (schema version ``1``)::

    {"schema_version": "1", "scene": str, "labels": {detection_id: instance_id},
     "rejected": [detection_id, ...], "labeler": str, "timestamp": ISO-8601 str}

``labels`` maps a detection to a human-assigned instance id; ``rejected`` lists detections marked
"not target". Validation enforces cannot-link: two detections of the same view (frame) may not
share an instance id. ``to_gt`` yields the ``Detection.id -> GT id | None`` mapping that
``bench.gt_matching.match_detections_to_gt`` produces and ``bench.metrics`` consumes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench import metrics

SCHEMA_VERSION = "1"
FILENAME = "identity_labels.json"


class LabelError(ValueError):
    """Invalid identity-label file or labels inconsistent with the run."""


@dataclass
class IdentityLabels:
    scene: str
    labels: dict[str, str] = field(default_factory=dict)  # detection id -> instance id
    rejected: list[str] = field(default_factory=list)
    labeler: str = ""
    timestamp: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene": self.scene,
            "labels": dict(self.labels),
            "rejected": list(self.rejected),
            "labeler": self.labeler,
            "timestamp": self.timestamp,
        }


def frame_of(det_id: str) -> int:
    """View/frame index of a ``"<frame>:<k>"`` detection id."""
    try:
        return int(det_id.split(":", 1)[0])
    except ValueError as exc:
        raise LabelError(f"malformed detection id {det_id!r}") from exc


def validate(labels: IdentityLabels, known_ids: set[str] | None = None) -> None:
    """Raise ``LabelError`` on unknown ids, overlaps or cannot-link violations."""
    if labels.schema_version != SCHEMA_VERSION:
        raise LabelError(f"unsupported schema_version {labels.schema_version!r}")
    if not labels.scene:
        raise LabelError("scene must be non-empty")
    if len(set(labels.rejected)) != len(labels.rejected):
        raise LabelError("duplicate ids in rejected")
    both = set(labels.labels) & set(labels.rejected)
    if both:
        raise LabelError(f"ids both labeled and rejected: {sorted(both)}")
    if known_ids is not None:
        unknown = (set(labels.labels) | set(labels.rejected)) - known_ids
        if unknown:
            raise LabelError(f"unknown detection ids: {sorted(unknown)}")
    seen: dict[tuple[int, str], str] = {}
    for det, inst in sorted(labels.labels.items()):
        key = (frame_of(det), str(inst))
        if key in seen:
            raise LabelError(
                f"cannot-link violated: {seen[key]} and {det} are in the same view "
                f"but share instance {inst!r}"
            )
        seen[key] = det


def save_labels(
    labels: IdentityLabels, path: Path | str, known_ids: set[str] | None = None
) -> Path:
    validate(labels, known_ids)
    p = Path(path)
    p.write_text(json.dumps(labels.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return p


def load_labels(path: Path | str, known_ids: set[str] | None = None) -> IdentityLabels:
    """Parse and validate a labels file (``known_ids``: the run's detection ids, if available)."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelError(f"{p}: not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise LabelError(f"{p}: top level must be an object")
    missing = {"schema_version", "scene", "labels", "rejected"} - set(raw)
    if missing:
        raise LabelError(f"{p}: missing keys {sorted(missing)}")
    if not isinstance(raw["labels"], dict) or not isinstance(raw["rejected"], list):
        raise LabelError(f"{p}: 'labels' must be an object and 'rejected' a list")
    labels = IdentityLabels(
        scene=str(raw["scene"]),
        labels={str(k): str(v) for k, v in raw["labels"].items()},
        rejected=[str(x) for x in raw["rejected"]],
        labeler=str(raw.get("labeler", "")),
        timestamp=str(raw.get("timestamp", "")),
        schema_version=str(raw["schema_version"]),
    )
    validate(labels, known_ids)
    return labels


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def seed_from_result(result: Any, scene: str, labeler: str = "prelabel:v2") -> IdentityLabels:
    """Pre-labelled mode: instance membership of an existing run becomes the starting labels.

    Rejected detections (``detection_status == "rejected"``) start as "not target".
    """
    labels: dict[str, str] = {}
    for inst in result.instances:
        for m in inst.members:
            labels[m] = f"i{inst.id}"
    rejected = sorted(d for d, s in result.detection_status.items() if s == "rejected")
    for d in rejected:
        labels.pop(d, None)
    return IdentityLabels(scene, labels, rejected, labeler, now_iso())


def to_gt(labels: IdentityLabels, all_ids: set[str] | None = None) -> dict[str, str | None]:
    """``Detection.id -> instance id`` (the ``gt_matching`` format); rejected/unlabelled -> None."""
    out: dict[str, str | None] = {d: None for d in (all_ids or set())}
    out.update({d: None for d in labels.rejected})
    out.update(labels.labels)
    return out


def clusters_from_result(result: Any) -> dict[str, int]:
    """Detection id -> predicted instance id of a run (non-rejected detections only)."""
    return {m: inst.id for inst in result.instances for m in inst.members}


def association_scores(clusters: Mapping[str, Any], labels: IdentityLabels) -> dict[str, float]:
    """Pairwise precision/recall/F1 and cluster purity of predicted clusters against labels.

    Only detections that are both predicted-clustered and human-labelled are scored.
    """
    gt = to_gt(labels)
    ids = sorted(d for d in clusters if gt.get(d) is not None)
    pred = [clusters[d] for d in ids]
    truth = [gt[d] for d in ids]
    p, r, f1 = metrics.pairwise_prf(pred, truth)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "purity": metrics.cluster_purity(pred, truth),
        "n_scored": float(len(ids)),
    }
