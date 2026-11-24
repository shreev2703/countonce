"""Consented correction pairs as matcher training data (PLAN §16B.3).

``countonce correct --pairs`` writes ``LabeledPair`` JSONL: ``hard_positive`` (a merge: same
object), ``hard_negative`` (a split: different objects) and ``false_positive`` (a reject, which
has no pairwise label and is not used by the matcher). The pairs of one job become one small
group whose detections carry the persisted geometry (xyz, size, frame). Persisted results hold
no ray or camera centre, so those inputs are zero and the box/score/lift-confidence columns are
neutral; the model still sees the 3D position and size the pair was judged on.

The loss is binary cross-entropy on the model's symmetric match probability of each pair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from countonce.match.model import NOMINAL_OBJECT_SIZE, DetInputs, mask_inputs

PAIR_KINDS = ("hard_positive", "hard_negative")


@dataclass
class CorrectionGroup:
    job_id: str
    inputs: DetInputs
    positives: np.ndarray  # (P, 2) detection indices
    negatives: np.ndarray  # (Q, 2)


def load_pairs(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not valid JSON ({exc})") from exc
        if row.get("kind") not in (*PAIR_KINDS, "false_positive"):
            raise ValueError(f"{path}:{n}: unknown pair kind {row.get('kind')!r}")
        rows.append(row)
    return rows


def build_groups(
    rows: list[dict[str, Any]], feature_group: str = "full", appearance_dim: int = 0
) -> list[CorrectionGroup]:
    """One group per job; pairs whose detections lack geometry or share a frame are dropped."""
    by_job: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["kind"] in PAIR_KINDS and r.get("det_b") is not None:
            by_job.setdefault(r["job_id"], []).append(r)
    groups = []
    for job, pairs in sorted(by_job.items()):
        recs: dict[str, dict[str, Any]] = {}
        for r in pairs:
            for k in (r["det_a"], r["det_b"]):
                g = (r.get("geometry") or {}).get(k)
                if g and g.get("xyz") is not None:
                    recs[k] = g
        ids = sorted(recs)
        pos_of = {d: i for i, d in enumerate(ids)}
        pos: list[tuple[int, int]] = []
        neg: list[tuple[int, int]] = []
        for r in pairs:
            a, b = r["det_a"], r["det_b"]
            if a in pos_of and b in pos_of and recs[a]["frame_id"] != recs[b]["frame_id"]:
                (pos if r["kind"] == "hard_positive" else neg).append((pos_of[a], pos_of[b]))
        if not (pos or neg):
            continue
        xyz = np.array([recs[d]["xyz"] for d in ids], dtype=float)
        shift = xyz.mean(axis=0)
        scale = float(np.max(np.linalg.norm(xyz - shift, axis=1))) or 1.0
        n = len(ids)
        size = np.array([recs[d].get("size") or NOMINAL_OBJECT_SIZE for d in ids], dtype=float)
        d_in = DetInputs(
            xyz=(xyz - shift) / scale,
            size=size / scale,
            box=np.zeros((n, 4)),
            score=np.ones(n),
            lift_conf=np.ones(n),
            ray=np.zeros((n, 3)),
            cam=np.zeros((n, 3)),
            view=np.array([recs[d]["frame_id"] for d in ids], dtype=np.int64),
        )
        groups.append(
            CorrectionGroup(
                job,
                mask_inputs(d_in, feature_group, appearance_dim),
                np.array(pos, dtype=np.int64).reshape(-1, 2),
                np.array(neg, dtype=np.int64).reshape(-1, 2),
            )
        )
    return groups


def correction_loss(model: Any, group: CorrectionGroup) -> Any:
    """Mean BCE over the group's labelled pairs (positives want p=1, negatives p=0)."""
    import torch

    p = model.match_probs(group.inputs)
    terms = []
    if len(group.positives):
        pp = p[group.positives[:, 0], group.positives[:, 1]]
        terms.append(-torch.log(pp.clamp_min(1e-6)))
    if len(group.negatives):
        pn = p[group.negatives[:, 0], group.negatives[:, 1]]
        terms.append(-torch.log((1 - pn).clamp_min(1e-6)))
    return torch.cat(terms).mean()
