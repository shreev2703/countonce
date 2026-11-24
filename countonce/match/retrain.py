"""Correction-driven retrain and promotion loop (PLAN §16B.3 "Retrain" and "Report").

Chains the existing pieces: the promoted checkpoint in the registry is fine-tuned on synthetic
episodes plus opted-in user corrections (Split -> hard negative, Merge -> hard positive,
Reject -> false-positive label, which the matcher does not use pairwise), the candidate is
registered, and it is promoted only through ``Registry.promote`` (the frozen-split and CPU golden
gate). Each run appends a before/after row to ``results/retrain_log.csv``.

Consent: a correction is used only when its own ``consent`` flag is set AND its user is in the
opted-in set (``JobStore.opted_in_users()``). With no opt-in source the set is empty, so nothing
is used. The frozen split is scored for the candidate and the incumbent; the gate runs on
``gate_split`` (default ``val``) and ``test`` is only reported, so it never steers promotion.
Evaluators and the trainer are injectable, which keeps the tests CPU only.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from countonce.corrections import (
    Correction,
    export_training_pairs,
    load_corrections,
    write_training_pairs_jsonl,
)
from countonce.match.registry import (
    Decision,
    Evaluator,
    GateConfig,
    Registry,
    RegistryError,
    file_sha256,
)
from countonce.provenance import git_commit

Trainer = Callable[[Any], Any]

LOG_COLUMNS = [
    "timestamp_utc",
    "incumbent",
    "candidate",
    "promoted",
    "reasons",
    "n_corrections",
    "n_hard_positive",
    "n_hard_negative",
    "n_false_positive",
    "data_version",
    "git_sha",
]
_METRICS = ("mae", "mae_lo", "mae_hi", "assoc_f1", "assoc_f1_lo", "assoc_f1_hi")
_GOLDEN = ("mape", "assoc_f1")


class RetrainError(ValueError):
    """The loop cannot run (no incumbent, missing inputs, unusable candidate)."""


@dataclass
class RetrainConfig:
    registry_root: str = "models"
    out_dir: str = "runs/retrain"
    log_csv: str = "results/retrain_log.csv"
    corrections_file: str = ""  # JSON list of corrections (countonce.corrections format)
    results_root: str = ""  # dir of <job_id>/result.json, the ORIGINAL results
    jobs_db: str = ""  # job store holding users.training_opt_in
    train: dict[str, Any] = field(default_factory=dict)  # TrainConfig overrides
    frozen: dict[str, dict[str, str]] = field(default_factory=dict)  # split -> {gt, predictions}
    gate_split: str = "val"
    golden: bool = True
    golden_config: str = ""
    n_boot: int = 1000
    gate: dict[str, Any] = field(default_factory=dict)  # GateConfig overrides

    @classmethod
    def from_yaml(cls, path: Path | str) -> RetrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        bad = sorted(set(raw) - known)
        if bad:
            raise ValueError(f"unknown retrain config keys: {bad}")
        return cls(**raw)


@dataclass
class CorrectionSet:
    used: list[Correction]
    pairs_path: Path | None
    counts: dict[str, int]
    sha: str = ""


@dataclass
class RetrainOutcome:
    incumbent: str
    candidate: str
    decision: Decision
    row: dict[str, Any]


def eligible_corrections(
    corrections: list[Correction], opted_in: set[str], known_jobs: set[str]
) -> list[Correction]:
    """Corrections that carry consent, belong to an opted-in user, and have a result."""
    return [
        c
        for c in corrections
        if c.consent and c.user is not None and c.user in opted_in and c.job_id in known_jobs
    ]


def build_correction_set(
    corrections: list[Correction],
    results: dict[str, Any],
    opted_in: set[str],
    out_dir: Path,
) -> CorrectionSet:
    used = eligible_corrections(corrections, opted_in, set(results))
    pairs = export_training_pairs(used, results, opted_in_users=opted_in)
    counts = {"hard_positive": 0, "hard_negative": 0, "false_positive": 0}
    for p in pairs:
        counts[p.kind] = counts.get(p.kind, 0) + 1
    if not pairs:
        return CorrectionSet(used, None, counts)
    path = write_training_pairs_jsonl(pairs, out_dir / "corrections.jsonl")
    return CorrectionSet(used, path, counts, file_sha256(path))


def _load_inputs(cfg: RetrainConfig) -> tuple[list[Correction], dict[str, Any], set[str]]:
    if not (cfg.corrections_file and cfg.results_root):
        return [], {}, set()
    from countonce.explain import load_result_dir

    corrections = load_corrections(cfg.corrections_file)
    results: dict[str, Any] = {}
    for jid in sorted({c.job_id for c in corrections}):
        d = Path(cfg.results_root) / jid
        if (d / "result.json").is_file():
            results[jid] = load_result_dir(d)
    opted: set[str] = set()
    if cfg.jobs_db:
        from countonce.jobs.store import SqliteJobStore

        opted = set(SqliteJobStore(cfg.jobs_db).opted_in_users())
    return corrections, results, opted


def _f(x: Any) -> float | None:
    return None if x is None else float(x)


def _summary(res: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("mae", "assoc_f1"):
        v = res.get(key)
        if isinstance(v, dict):
            out[key] = _f(v.get("estimate"))
            out[f"{key}_lo"] = _f(v.get("low"))
            out[f"{key}_hi"] = _f(v.get("high"))
        else:
            out[key] = _f(v)
            out[f"{key}_lo"] = out[f"{key}_hi"] = None
    return out


def memoize(ev: Evaluator) -> Evaluator:
    """Cache by checkpoint path so the gate and the log share one evaluation."""
    cache: dict[Path, dict[str, Any]] = {}

    def run(ckpt: Path) -> dict[str, Any]:
        key = Path(ckpt).resolve()
        if key not in cache:
            cache[key] = ev(ckpt)
        return cache[key]

    return run


def _default_trainer(train_cfg: Any) -> Any:
    from countonce.match.train import train

    return train(train_cfg)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def append_log_row(path: Path | str, row: dict[str, Any]) -> None:
    """Append one row. Columns are the union of the old header and the row; old rows keep blanks."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if p.is_file():
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
    cols = list(LOG_COLUMNS)
    for r in [*rows, row]:
        cols += [c for c in r if c not in cols]
    rows.append({k: _cell(v) for k, v in row.items()})
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


def retrain(
    cfg: RetrainConfig,
    *,
    frozen_evals: dict[str, Evaluator] | None = None,
    golden_eval: Evaluator | None = None,
    trainer: Trainer | None = None,
    corrections: list[Correction] | None = None,
    results: dict[str, Any] | None = None,
    opted_in_users: set[str] | None = None,
) -> RetrainOutcome:
    """Fine-tune the incumbent, register the candidate, gate it, and log the before/after row.

    ``corrections`` / ``results`` / ``opted_in_users`` override what the config would load.
    """
    from countonce.match.train import TrainConfig, data_version

    reg = Registry(cfg.registry_root)
    incumbent = reg.current()
    if incumbent is None:
        raise RetrainError(f"{reg.path}: no promoted model to fine-tune; promote one first")
    if not frozen_evals or cfg.gate_split not in frozen_evals:
        raise RetrainError(
            f"no frozen-split evaluator for gate_split {cfg.gate_split!r}; "
            "the gate needs the frozen split (bench/splits.json)"
        )
    if corrections is None:
        corrections, results, loaded_opt = _load_inputs(cfg)
        if opted_in_users is None:
            opted_in_users = loaded_opt
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cset = build_correction_set(corrections, results or {}, set(opted_in_users or ()), out_dir)

    tcfg = TrainConfig(**{**cfg.train, "out_dir": str(out_dir / "train")})
    tcfg.init_checkpoint = str(reg.checkpoint_path(incumbent["version"]))
    tcfg.corrections = str(cset.pairs_path) if cset.pairs_path else ""
    tcfg.resume = False
    tcfg.validated()
    (trainer or _default_trainer)(tcfg)
    best = Path(tcfg.out_dir) / "best.pt"
    if not best.is_file():
        raise RetrainError(f"training produced no checkpoint at {best}")

    dirs_ok = Path(tcfg.train_dir).is_dir() and Path(tcfg.val_dir).is_dir()
    base_version = data_version(tcfg.train_dir, tcfg.val_dir) if dirs_ok else ""
    version = f"{base_version}+corr-{cset.sha[:8]}" if cset.sha else base_version
    try:
        entry = reg.register(
            best,
            dataset_version=version,
            recipe_version="retrain",
            config={"train": cfg.train, "n_corrections": len(cset.used), **cset.counts},
        )
    except RegistryError as exc:
        raise RetrainError(f"candidate not registered: {exc}") from exc
    cand_v, inc_v = entry["version"], incumbent["version"]
    cand_path, inc_path = reg.checkpoint_path(cand_v), reg.checkpoint_path(inc_v)

    frozen = {k: memoize(v) for k, v in frozen_evals.items()}
    golden = memoize(golden_eval) if golden_eval else None
    gate = GateConfig(**cfg.gate) if cfg.gate else GateConfig()
    decision = reg.promote(
        cand_v, gate=gate, frozen_eval=frozen[cfg.gate_split], golden_eval=golden
    )

    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "incumbent": inc_v,
        "candidate": cand_v,
        "promoted": decision.passed,
        "reasons": " | ".join(decision.reasons),
        "n_corrections": len(cset.used),
        "n_hard_positive": cset.counts["hard_positive"],
        "n_hard_negative": cset.counts["hard_negative"],
        "n_false_positive": cset.counts["false_positive"],
        "data_version": version,
        "git_sha": git_commit(),
    }
    for split, ev in frozen.items():
        for who, path in (("before", inc_path), ("after", cand_path)):
            s = _summary(ev(path))
            for m in _METRICS:
                row[f"{split}_{who}_{m}"] = s[m]
    if golden:
        for who, path in (("before", inc_path), ("after", cand_path)):
            overall = golden(path).get("overall", {})
            for m in _GOLDEN:
                row[f"golden_{who}_{m}"] = _f(overall.get(m))
    append_log_row(cfg.log_csv, row)
    return RetrainOutcome(inc_v, cand_v, decision, row)


def evaluators_from_config(
    cfg: RetrainConfig,
) -> tuple[dict[str, Evaluator], Evaluator | None]:
    """Default evaluators: ``bench.evaluate`` on ``<predictions>/<version>.json`` per frozen split
    and the CPU golden set. Raises ``RetrainError`` while the frozen split is not configured."""
    from countonce.match.registry import frozen_evaluator, golden_evaluator

    if not cfg.frozen:
        raise RetrainError("config has no `frozen:` splits (gt + predictions dir per split)")
    frozen: dict[str, Evaluator] = {}
    for split, spec in cfg.frozen.items():
        if "gt" not in spec or "predictions" not in spec:
            raise RetrainError(f"frozen.{split} needs `gt` and `predictions`")
        frozen[split] = frozen_evaluator(spec["gt"], spec["predictions"], n_boot=cfg.n_boot)
    golden = golden_evaluator(cfg.golden_config or None) if cfg.golden else None
    return frozen, golden
