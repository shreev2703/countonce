"""Frozen test split (PLAN §9.8): scene hashes recorded once in ``bench/splits.json``."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

SPLITS_VERSION = 1
DEFAULT_PATH = Path(__file__).with_name("splits.json")


def scene_hash(scene_dir: str | Path) -> str:
    """SHA-256 over sorted relative file names and bytes; renames and edits both change it."""
    root = Path(scene_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode() + b"\0")
        h.update(f.read_bytes() + b"\0")
    return h.hexdigest()


def assign_splits(scene_ids: list[str], test_frac: float = 0.3, seed: int = 0) -> dict[str, str]:
    """Deterministic scene -> "val" | "test" assignment, independent of input order.

    Scenes are ranked by sha256(seed:id); the first ``round(test_frac * n)`` (at least 1 when
    n >= 2) are test. Adding a scene later can reshuffle, which is why the result is frozen.
    """
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("duplicate scene ids")
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    ranked = sorted(scene_ids, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    n_test = round(test_frac * len(ranked))
    if len(ranked) >= 2:
        n_test = min(max(n_test, 1), len(ranked) - 1)
    test = set(ranked[:n_test])
    return {s: ("test" if s in test else "val") for s in scene_ids}


def write_splits(
    scene_hashes: Mapping[str, str],
    path: str | Path = DEFAULT_PATH,
    test_frac: float = 0.3,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, dict[str, str]]:
    """Write the frozen split file. Refuses to touch an existing file unless ``overwrite``."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} is frozen; pass overwrite=True only before any tuning")
    assignment = assign_splits(list(scene_hashes), test_frac, seed)
    splits: dict[str, dict[str, str]] = {"val": {}, "test": {}}
    for sid in sorted(scene_hashes):
        splits[assignment[sid]][sid] = scene_hashes[sid]
    doc = {"version": SPLITS_VERSION, "seed": seed, "test_frac": test_frac, "splits": splits}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return splits


def load_splits(path: str | Path = DEFAULT_PATH) -> dict[str, dict[str, str]]:
    doc = json.loads(Path(path).read_text())
    if doc.get("version") != SPLITS_VERSION:
        raise ValueError(f"unsupported splits version {doc.get('version')!r}")
    return {k: dict(v) for k, v in doc["splits"].items()}


def verify_splits(current_hashes: Mapping[str, str], path: str | Path = DEFAULT_PATH) -> list[str]:
    """Compare recorded hashes with ``current_hashes`` (scene id -> hash now on disk).

    Returns human-readable problems; an empty list means the frozen split is intact. Checks:
    scenes in two splits, recorded scenes that are missing or changed. Scenes on disk but not
    in the file are reported too, so a new scene can't silently skip assignment.
    """
    splits = load_splits(path)
    problems: list[str] = []
    seen: dict[str, str] = {}
    for name, scenes in splits.items():
        for sid, recorded in scenes.items():
            if sid in seen:
                problems.append(f"{sid}: in both '{seen[sid]}' and '{name}'")
            seen[sid] = name
            now = current_hashes.get(sid)
            if now is None:
                problems.append(f"{sid}: recorded in '{name}' but missing")
            elif now != recorded:
                problems.append(f"{sid}: hash changed in '{name}'")
    for sid in current_hashes:
        if sid not in seen:
            problems.append(f"{sid}: not in any split")
    return problems


def split_of(scene_id: str, path: str | Path = DEFAULT_PATH) -> str | None:
    for name, scenes in load_splits(path).items():
        if scene_id in scenes:
            return name
    return None
