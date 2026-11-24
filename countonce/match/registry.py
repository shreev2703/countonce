"""Local versioned matcher-checkpoint registry with a promotion gate (PLAN §16B.3, §10.6).

Layout under ``root`` (default ``models/``)::

    registry.json        {"current": "v002", "models": [{version, sha256, ...}, ...]}
    checkpoints/vNNN.pt  immutable copy of each registered checkpoint
    promotion_log.jsonl  one JSON line per promotion decision, pass or fail

``register`` stores a checkpoint with its hash, dataset/recipe version, git sha, config and
metrics. ``promote`` runs the gate: the candidate and the current model are scored on the frozen
split (a ``bench.evaluate`` result) and on the CPU golden set (a ``bench.golden`` result); the
candidate is refused if it regresses beyond the tolerances. The evaluators are injected, so the
gate itself is torch-free and unit-testable with fake checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from countonce.provenance import git_commit

Evaluator = Callable[[Path], dict[str, Any]]

DEFAULT_ROOT = Path("models")


class RegistryError(ValueError):
    """Unknown version, hash mismatch, or an unusable registry file."""


@dataclass
class GateConfig:
    """Regression tolerances. A candidate may be worse than the incumbent by at most these."""

    frozen_mae_rel: float = 0.05  # relative increase of the frozen-split MAE
    frozen_assoc_f1_abs: float = 0.01  # absolute drop of the frozen-split association F1
    golden: dict[str, float | None] = field(
        default_factory=lambda: {
            "mape": 0.02,
            "assoc_f1": 0.02,
            "peak_vram_mb": None,  # hardware noise, shown but never fails
            "latency_s": None,
        }
    )
    require_frozen: bool = True  # refuse (not skip) when no frozen evaluator is available


@dataclass
class Decision:
    candidate: str
    incumbent: str | None
    passed: bool
    reasons: list[str]
    checks: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "candidate": self.candidate,
            "incumbent": self.incumbent,
            "passed": self.passed,
            "reasons": self.reasons,
            "checks": self.checks,
        }


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _num(x: Any) -> float | None:
    if isinstance(x, dict):  # a bootstrap CI dict: use its point estimate
        x = x.get("estimate")
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def frozen_summary(result: dict[str, Any]) -> dict[str, float | None]:
    return {"mae": _num(result.get("mae")), "assoc_f1": _num(result.get("assoc_f1"))}


def compare_frozen(
    base: dict[str, Any], cand: dict[str, Any], cfg: GateConfig
) -> tuple[dict[str, Any], list[str]]:
    """Frozen-split comparison; returns (metrics table, regression descriptions)."""
    problems: list[str] = []
    for name, res in (("candidate", cand), ("incumbent", base)):
        if res.get("frozen_problems"):
            problems.append(
                f"frozen: {name} not scored on the frozen split: {res['frozen_problems']}"
            )
    b, c = frozen_summary(base), frozen_summary(cand)
    if b["mae"] is not None:
        if c["mae"] is None:
            problems.append("frozen/mae: missing for candidate")
        else:
            limit = b["mae"] * (1 + cfg.frozen_mae_rel) + 1e-12
            if c["mae"] > limit:
                problems.append(
                    f"frozen/mae: {b['mae']:.4g} -> {c['mae']:.4g} "
                    f"(tolerance +{cfg.frozen_mae_rel:.0%})"
                )
    if b["assoc_f1"] is not None:
        if c["assoc_f1"] is None:
            problems.append("frozen/assoc_f1: missing for candidate")
        elif b["assoc_f1"] - c["assoc_f1"] > cfg.frozen_assoc_f1_abs + 1e-12:
            problems.append(
                f"frozen/assoc_f1: {b['assoc_f1']:.4g} -> {c['assoc_f1']:.4g} "
                f"(tolerance -{cfg.frozen_assoc_f1_abs})"
            )
    return {"incumbent": b, "candidate": c}, problems


class Registry:
    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.path = self.root / "registry.json"
        self.log_path = self.root / "promotion_log.jsonl"

    # ------------------------------------------------------------ storage
    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema": 1, "current": None, "models": []}
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{self.path}: not valid JSON ({exc})") from exc
        if not isinstance(doc, dict) or "models" not in doc:
            raise RegistryError(f"{self.path}: not a registry file")
        return doc

    def _save(self, doc: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def _log(self, decision: Decision) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict(), sort_keys=True) + "\n")

    # ------------------------------------------------------------ queries
    def models(self) -> list[dict[str, Any]]:
        doc = self._load()
        return [{**m, "current": m["version"] == doc["current"]} for m in doc["models"]]

    def get(self, version: str) -> dict[str, Any]:
        for m in self._load()["models"]:
            if m["version"] == version:
                return dict(m)
        raise RegistryError(f"unknown model version {version!r}")

    def current(self) -> dict[str, Any] | None:
        doc = self._load()
        return None if doc["current"] is None else self.get(doc["current"])

    def checkpoint_path(self, version: str) -> Path:
        return self.root / str(self.get(version)["path"])

    def log(self) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        return [json.loads(x) for x in self.log_path.read_text().splitlines() if x.strip()]

    # ------------------------------------------------------------ register
    def register(
        self,
        checkpoint: Path | str,
        *,
        dataset_version: str = "",
        recipe_version: str = "",
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        git_sha: str | None = None,
    ) -> dict[str, Any]:
        src = Path(checkpoint)
        if not src.is_file():
            raise FileNotFoundError(f"checkpoint {src} not found")
        digest = file_sha256(src)
        doc = self._load()
        for m in doc["models"]:
            if m["sha256"] == digest:
                raise RegistryError(f"checkpoint already registered as {m['version']}")
        version = f"v{len(doc['models']) + 1:03d}"
        rel = Path("checkpoints") / f"{version}.pt"
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, self.root / rel)
        entry = {
            "version": version,
            "sha256": digest,
            "path": rel.as_posix(),
            "dataset_version": dataset_version,
            "recipe_version": recipe_version,
            "git_sha": git_sha if git_sha is not None else git_commit(),
            "config": config or {},
            "metrics": metrics or {},
            "registered_at": _now(),
        }
        doc["models"].append(entry)
        self._save(doc)
        return entry

    # ------------------------------------------------------------ promote
    def promote(
        self,
        version: str,
        *,
        gate: GateConfig | None = None,
        frozen_eval: Evaluator | None = None,
        golden_eval: Evaluator | None = None,
        force_first: bool = True,
    ) -> Decision:
        """Gate and (if it passes) promote ``version``. Every outcome is appended to the log.

        ``frozen_eval`` / ``golden_eval`` map a checkpoint path to a ``bench.evaluate`` /
        ``bench.golden`` result dict. With no incumbent the first model is promoted without a
        comparison (``force_first``), but its integrity check still runs.
        """
        gate = gate or GateConfig()
        doc = self._load()
        cand = self.get(version)  # RegistryError if unknown
        incumbent = doc["current"]
        reasons: list[str] = []
        checks: dict[str, Any] = {}
        cpath = self.root / cand["path"]
        if not cpath.is_file() or file_sha256(cpath) != cand["sha256"]:
            reasons.append(f"integrity: checkpoint for {version} is missing or its hash changed")
        elif incumbent == version:
            reasons.append(f"{version} is already the current model")
        elif incumbent is None:
            checks["note"] = "no incumbent; promoted without comparison"
            if not force_first:
                reasons.append("no incumbent to compare against")
        else:
            ipath = self.checkpoint_path(incumbent)
            if frozen_eval is None:
                if gate.require_frozen:
                    reasons.append("frozen: no frozen-split evaluator supplied (required)")
            else:
                table, problems = compare_frozen(frozen_eval(ipath), frozen_eval(cpath), gate)
                checks["frozen"] = table
                reasons += problems
            if golden_eval is not None:
                from bench.golden import compare

                _, problems = compare(golden_eval(ipath), golden_eval(cpath), gate.golden)
                checks["golden"] = {"regressions": problems}
                reasons += [f"golden: {p}" for p in problems]
            else:
                checks["golden"] = "skipped (no evaluator)"
        decision = Decision(version, incumbent, not reasons, reasons, checks, _now())
        if decision.passed:
            doc["current"] = version
            self._save(doc)
        self._log(decision)
        return decision


# ---------------------------------------------------------------- default evaluators
def frozen_evaluator(
    gt_path: Path | str, predictions_dir: Path | str, n_boot: int = 1000
) -> Evaluator:
    """Score ``<predictions_dir>/<version>.json`` (Predictions format) with ``bench.evaluate``.

    The checkpoint's file stem is its version, so the same directory serves the candidate and
    the incumbent. Predictions come from running that checkpoint on the frozen split.
    """
    from bench.evaluate import evaluate, load_gt, load_predictions

    gt = load_gt(gt_path)

    def run(ckpt: Path) -> dict[str, Any]:
        f = Path(predictions_dir) / f"{ckpt.stem}.json"
        if not f.is_file():
            raise FileNotFoundError(f"no frozen-split predictions for {ckpt.stem}: {f}")
        return evaluate(load_predictions(f), gt, n_boot=n_boot)

    return run


def golden_evaluator(config: Path | str | None = None) -> Evaluator:
    """Run the CPU synthetic golden set with the checkpoint as the v3 matcher."""
    from bench.golden import DEFAULT_CONFIG, build_result, load_config, run_synthetic

    cfg = load_config(config or DEFAULT_CONFIG)

    def run(ckpt: Path) -> dict[str, Any]:
        return build_result(run_synthetic(cfg, checkpoint=ckpt), "cpu-synthetic")

    return run
