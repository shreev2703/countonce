"""Hugging Face release scaffolding (PLAN §16): dataset layout + card, model card. No network.

Nothing here uploads. Both commands only write local files; every metric stays
``TODO: fill from results`` unless a results JSON is given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TODO = "TODO: fill from results"
DEFAULT_LICENSE = "other"


class ReleaseError(ValueError):
    pass


def _front_matter(fields: dict[str, Any]) -> str:
    lines = ["---"]
    for k, v in fields.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            lines.extend(f"- {x}" for x in v)
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _strip_title(text: str) -> str:
    """Drop the template's own H1 so the generated card has a single title."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _read(path: Path, what: str) -> str:
    if not path.is_file():
        raise ReleaseError(f"{what} not found: {path}")
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- dataset
def build_synth_dataset(
    out: Path | str,
    shards: Path | str,
    *,
    n_scenes: int | None = None,
    split: str | None = None,
    datasheet: Path | str = "docs/DATASHEET.md",
    license_id: str = DEFAULT_LICENSE,
    max_bytes: int = 1 << 30,
) -> dict[str, Any]:
    """Pack a subset of DAG shards into ``<out>/data/*.tar`` plus ``<out>/README.md``."""
    from synth.recipe import RECIPE_VERSION
    from synth.shards import ShardError, pack_shards, read_shards

    out = Path(out)
    src = Path(shards)
    sheet = _read(Path(datasheet), "datasheet")
    if not (src / "index.json").is_file():
        raise ReleaseError(f"no shard index at {src}: run `countonce dag run` first")

    def subset() -> Any:
        n = 0
        for s in read_shards(src):
            if split is not None and s.metadata.get("split") != split:
                continue
            if n_scenes is not None and n >= n_scenes:
                return
            n += 1
            yield s

    try:
        index = pack_shards(subset(), out / "data", max_bytes=max_bytes)
    except ShardError as exc:
        raise ReleaseError(str(exc)) from exc
    if index["n_samples"] == 0:
        raise ReleaseError("no scenes selected (empty shards or no match for --split)")
    meta = {
        "license": license_id,
        "task_categories": ["object-detection"],
        "tags": ["counting", "multi-view", "synthetic"],
        "pretty_name": "CountOnce-Synth",
        "countonce_recipe_version": f'"{RECIPE_VERSION}"',
        "countonce_n_scenes": index["n_samples"],
    }
    body = (
        "# CountOnce-Synth\n\n"
        f"- Recipe version: `{RECIPE_VERSION}`\n"
        f"- License: `{license_id}`"
        + (" (TODO: decide, see Uses and distribution)" if license_id == DEFAULT_LICENSE else "")
        + "\n"
        f"- Scenes in this subset: {index['n_samples']}"
        + (f" (split `{split}`)" if split else "")
        + "\n"
        f"- Shards: {len(index['shards'])} WebDataset-layout tars in `data/`, "
        "hashes in `data/index.json`\n"
        "- Each sample: `<scene>.json` (metadata) + `<scene>.npz` "
        "(images, depth, instance_masks, camera_centers)\n\n"
        + _strip_title(sheet).replace("# CountOnce-Bench datasheet (template)", "")
        + "\n"
    )
    (out / "README.md").write_text(_front_matter(meta) + body, encoding="utf-8")
    return {"out": str(out), "n_scenes": index["n_samples"], "shards": len(index["shards"])}


# ----------------------------------------------------------------------------- model card
_METRIC_ROWS = [
    ("Count MAE", "count_mae", "mae"),
    ("MAPE", "mape", "mape"),
    ("Association pairwise F1", "assoc_f1", "assoc_f1"),
]


def _num(x: Any) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else TODO


def render_model_card(
    entry: dict[str, Any],
    template: str,
    *,
    results: dict[str, Any] | None = None,
    license_id: str = DEFAULT_LICENSE,
) -> str:
    """HF model card: front-matter + registry facts + the template body.

    Metrics come from ``results`` (a results JSON) or the registry entry's ``metrics``; any
    missing metric stays ``TODO: fill from results``.
    """
    metrics = {**(entry.get("metrics") or {}), **(results or {})}
    flat: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat.update({f"{k}_{kk}": vv for kk, vv in v.items()})
        flat[k] = v
    meta = {
        "license": license_id,
        "library_name": "countonce",
        "tags": ["counting", "multi-view", "association"],
    }
    details = (
        "# CountOnce-Match\n\n"
        "## Registry entry\n"
        f"- Version: `{entry.get('version', TODO)}`\n"
        f"- SHA-256: `{entry.get('sha256') or TODO}`\n"
        f"- Dataset version: `{entry.get('dataset_version') or TODO}`\n"
        f"- Recipe version: `{entry.get('recipe_version') or TODO}`\n"
        f"- Git commit: `{entry.get('git_sha') or TODO}`\n"
        f"- Registered at: {entry.get('registered_at') or TODO}\n"
        f"- License: `{license_id}`\n\n"
    )
    body = _strip_title(template)
    # Fill the evaluation table rows we have a number for; leave the rest as TODO.
    out_lines = []
    for line in body.splitlines():
        for label, *keys in _METRIC_ROWS:
            if line.startswith(f"| {label} |"):
                val = next((flat[k] for k in keys if k in flat), None)
                if val is not None:
                    line = f"| {label} | {_num(val)} | {_num(flat.get(f'{keys[0]}_ci'))} |"
        out_lines.append(line)
    return _front_matter(meta) + details + "\n".join(out_lines) + "\n"


def build_model_card(
    out: Path | str,
    *,
    registry_root: Path | str = "models",
    version: str | None = None,
    template: Path | str = "docs/MODEL_CARD.md",
    results: Path | str | None = None,
    license_id: str = DEFAULT_LICENSE,
) -> Path:
    from countonce.match.registry import Registry, RegistryError

    reg = Registry(registry_root)
    try:
        entry = reg.get(version) if version else reg.current()
        if entry is None:
            models = reg.models()
            if not models:
                raise ReleaseError("registry is empty: register a checkpoint first")
            entry = models[-1]
    except RegistryError as exc:
        raise ReleaseError(str(exc)) from exc
    res: dict[str, Any] | None = None
    if results is not None and Path(results).is_file():  # a missing results file => TODOs
        res = json.loads(Path(results).read_text(encoding="utf-8"))
    card = render_model_card(
        entry, _read(Path(template), "model card template"), results=res, license_id=license_id
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    return out
