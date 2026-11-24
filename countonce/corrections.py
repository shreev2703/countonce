"""User correction tools and the correction dataset (PLAN 14.8, 16B.3).

``apply_corrections`` replays Split / Merge / Reject actions on a stored result (schema 1.1) and
returns a new result; the input is never mutated. Cannot-link is enforced: no instance may hold
two detections from the same frame. ``export_training_pairs`` turns consented corrections into
labelled pairs (Split -> hard negative, Merge -> hard positive, Reject -> false positive).

Operates on the persisted ``ResultV11`` rather than the in-memory ``CountResult`` because
corrections arrive after a job finished (see docs/DEVIATIONS.md).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from countonce.errors import CorrectionError, ErrorCode
from countonce.schemas import InstanceModel, ResultV11

ACTIONS = ("split", "merge", "reject")
REJECT_REASON = "user_reject"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Correction:
    job_id: str
    action: str  # split | merge | reject
    detection_ids: tuple[str, ...]
    user: str | None = None
    timestamp: str = field(default_factory=_now)
    consent: bool = False  # user agreed to use this correction for training

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")
        object.__setattr__(self, "detection_ids", tuple(self.detection_ids))
        if not self.detection_ids:
            raise ValueError("detection_ids must not be empty")
        if self.action == "merge" and len(set(self.detection_ids)) < 2:
            raise ValueError("merge needs at least two distinct detection ids")

    @property
    def key(self) -> str:
        """Stable identity used for idempotent re-application (ignores user/time/consent)."""
        ids = sorted(set(self.detection_ids))
        raw = json.dumps([self.job_id, self.action, ids])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detection_ids"] = list(self.detection_ids)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Correction:
        kw: dict[str, Any] = {
            "job_id": str(d["job_id"]),
            "action": str(d["action"]),
            "detection_ids": tuple(d["detection_ids"]),
            "user": d.get("user"),
            "consent": bool(d.get("consent", False)),
        }
        if d.get("timestamp"):
            kw["timestamp"] = str(d["timestamp"])
        return cls(**kw)


def load_corrections(path: Path | str) -> list[Correction]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("corrections", [])
    return [Correction.from_dict(r) for r in raw]


@dataclass
class LabeledPair:
    kind: str  # hard_negative | hard_positive | false_positive
    job_id: str
    det_a: str
    det_b: str | None  # None for false_positive
    geometry: dict[str, Any]  # persisted per-detection records, when available
    user: str | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------ applying
def _frame(det_id: str) -> int:
    return int(det_id.split(":")[0])


def _check_cannot_link(members: Iterable[str], what: str) -> None:
    seen: dict[int, str] = {}
    for m in members:
        f = _frame(m)
        if f in seen:
            raise CorrectionError(
                f"{what} would put {seen[f]} and {m} (same frame {f}) in one instance",
                code=ErrorCode.ERR_CANNOT_LINK,
            )
        seen[f] = m


def _owner(instances: Mapping[int, InstanceModel], det_id: str) -> int:
    for iid, inst in instances.items():
        if det_id in inst.members:
            return iid
    raise CorrectionError(f"detection {det_id!r} is not in any instance (unknown or rejected)")


def _recompute_instance(
    inst: InstanceModel,
    members: list[str],
    pos: Mapping[str, Sequence[float]],
    size: Mapping[str, float],
) -> InstanceModel:
    pts = [pos[m] for m in members if m in pos]
    xyz = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3)) if pts else inst.xyz
    sz = [size[m] for m in members if m in size]
    return inst.model_copy(
        update={
            "members": members,
            "n_views": len({_frame(m) for m in members}),
            "xyz": xyz,
            "size": sum(sz) / len(sz) if sz else inst.size,
        }
    )


def apply_corrections(result: ResultV11, corrections: Sequence[Correction]) -> ResultV11:
    """Return a corrected copy of ``result``; already-applied corrections are skipped."""
    out = result.model_copy(deep=True)
    prov = out.config
    applied: list[dict[str, Any]] = list(prov.get("corrections_applied", []))
    done = {a["key"] for a in applied}
    instances = {i.id: i for i in out.instances}
    pos = {d.id: d.xyz for d in out.detections if d.xyz is not None}
    size = {d.id: d.size for d in out.detections if d.size is not None}
    status = dict(out.detection_status)
    reasons = dict(out.reject_reasons)
    touched: set[int] = set()
    n0 = len(instances)

    for c in corrections:
        if c.key in done:
            continue
        ids = list(dict.fromkeys(c.detection_ids))
        if c.action == "reject":
            for d in ids:
                if d in status and status[d] == "rejected":
                    continue
                if d not in status:
                    raise CorrectionError(f"unknown detection {d!r}")
                iid = _owner(instances, d) if status[d] != "rejected" else -1
                inst = instances[iid]
                rest = [m for m in inst.members if m != d]
                if rest:
                    instances[iid] = _recompute_instance(inst, rest, pos, size)
                    touched.add(iid)
                else:
                    del instances[iid]
                    touched.discard(iid)
                status[d] = "rejected"
                reasons[d] = REJECT_REASON
        elif c.action == "split":
            owners = {_owner(instances, d) for d in ids}
            _check_cannot_link(ids, "split")
            moved = set(ids)
            for iid in owners:
                if set(instances[iid].members) <= moved:
                    raise CorrectionError(
                        f"split would leave instance {iid} empty; pick a subset of its members"
                    )
            for iid in owners:
                inst = instances[iid]
                instances[iid] = _recompute_instance(
                    inst, [m for m in inst.members if m not in moved], pos, size
                )
                touched.add(iid)
            new_id = max(instances, default=-1) + 1
            base = instances[next(iter(sorted(owners)))]
            instances[new_id] = _recompute_instance(base, ids, pos, size).model_copy(
                update={"id": new_id}
            )
            touched.add(new_id)
        else:  # merge
            owners_sorted = sorted({_owner(instances, d) for d in ids})
            if len(owners_sorted) < 2:
                continue  # already one instance
            members = [m for iid in owners_sorted for m in instances[iid].members]
            _check_cannot_link(members, "merge")
            keep = owners_sorted[0]
            for iid in owners_sorted[1:]:
                del instances[iid]
                touched.discard(iid)
            instances[keep] = _recompute_instance(instances[keep], members, pos, size)
            touched.add(keep)
        applied.append(
            {
                "key": c.key,
                "action": c.action,
                "detection_ids": ids,
                "user": c.user,
                "timestamp": c.timestamp,
            }
        )
        done.add(c.key)

    for iid in touched:  # one 'new' (latest frame) per changed instance, the rest duplicates
        ms = sorted(instances[iid].members, key=lambda m: (_frame(m), m))
        for m in ms:
            status[m] = "duplicate"
        status[ms[-1]] = "new"

    out.instances = [instances[i] for i in sorted(instances)]
    out.detection_status = status
    out.reject_reasons = reasons
    delta = len(instances) - n0
    out.count = len(instances)
    out.zero_count = out.count == 0
    out.interval = out.interval.model_copy(
        update={"low": max(0, out.interval.low + delta), "high": max(0, out.interval.high + delta)}
    )
    counts: dict[str, int] = {k: 0 for k in out.per_frame_counts}
    for d, s in status.items():
        if s != "rejected":
            counts[str(_frame(d))] = counts.get(str(_frame(d)), 0) + 1
    out.per_frame_counts = counts
    prov = dict(prov)
    prov["corrections_applied"] = applied
    out.config = prov
    return out


# ------------------------------------------------------------------------------ dataset
def export_training_pairs(
    corrections: Sequence[Correction],
    results: Mapping[str, ResultV11],
    opted_in_users: Collection[str] | None = None,
) -> list[LabeledPair]:
    """Labelled pairs from consented corrections. ``results`` are the ORIGINAL results by job id.

    ``opted_in_users`` (e.g. ``JobStore.opted_in_users()``, PLAN 14.5 ``training_opt_in``) is a
    second gate: when given, only corrections by those users are exported, and corrections with
    no user are excluded. ``None`` keeps the per-correction ``consent`` flag as the only gate.
    """
    pairs: list[LabeledPair] = []
    for c in corrections:
        res = results.get(c.job_id)
        if not c.consent or res is None:
            continue
        if opted_in_users is not None and (c.user is None or c.user not in opted_in_users):
            continue
        recs = {d.id: d.model_dump(mode="json") for d in res.detections}
        inst_of = {m: i for i in res.instances for m in i.members}
        ids = list(dict.fromkeys(c.detection_ids))

        def add(
            kind: str, a: str, b: str | None, c: Correction = c, recs: dict[str, Any] = recs
        ) -> None:
            geo = {k: recs[k] for k in (a, b) if k is not None and k in recs}
            pairs.append(LabeledPair(kind, c.job_id, a, b, geo, c.user, c.timestamp))

        if c.action == "reject":
            for d in ids:
                add("false_positive", d, None)
        elif c.action == "split":
            moved = set(ids)
            seen: set[tuple[str, str]] = set()
            for d in ids:
                inst = inst_of.get(d)
                if inst is None:
                    continue
                for m in inst.members:
                    if m not in moved and (d, m) not in seen:
                        seen.add((d, m))
                        add("hard_negative", d, m)
        else:
            groups: list[InstanceModel] = []
            for d in ids:
                inst = inst_of.get(d)
                if inst is not None and inst.id not in [g.id for g in groups]:
                    groups.append(inst)
            for gi, a in enumerate(groups):
                for b in groups[gi + 1 :]:
                    for ma in a.members:
                        for mb in b.members:
                            add("hard_positive", ma, mb)
    return pairs


def write_training_pairs_jsonl(pairs: Sequence[LabeledPair], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for pr in pairs:
            fh.write(json.dumps(pr.to_dict(), sort_keys=True) + "\n")
    return p


def write_corrected(
    result: ResultV11, corrected_dir: Path | str, scene_scale: float | None
) -> None:
    """Write result.json / markers.json / counts.csv for a corrected result."""
    from countonce.export.writers import _write_json, to_markers, write_counts_csv

    d = Path(corrected_dir)
    _write_json(d / "result.json", result)
    _write_json(d / "markers.json", to_markers(result, scene_scale))
    write_counts_csv(result, d / "counts.csv")
