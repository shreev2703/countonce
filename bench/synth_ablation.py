"""Config-driven v3 ablation runner (PLAN §10.3 "pairwise / + cycle / + sync / both" and §10.5
feature-group attribution) on synthetic episodes.

Arms (``arm:*``): pairwise only, + cycle loss (training), + synchronization (inference on the
pairwise model), both. Feature groups (``group:*``): 3D only, 3D + rays, full; the appearance
slot is zero-masked when the episodes carry none. Each trained configuration is fit once per
seed and reused by every arm that shares it. Per-seed rows and a summary (mean ± std over seeds,
percentile bootstrap CI over test episodes of the per-episode seed-mean) are written as CSV.
Nothing here is a result until it is run on real splits.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bench.stats import bootstrap_ci
from bench.synth_assoc import episode_to_lifted, score_result
from countonce.match.model import FEATURE_GROUPS, load_checkpoint
from countonce.match.train import TrainConfig, train
from countonce.merge.v3 import V3Associator
from synth.geometry import Episode, load_episode

METRICS = ("assoc_f1", "duplicate_rate", "miss_rate", "count_error")
ARMS = ("pairwise", "cycle", "sync", "both")


@dataclass
class AblationConfig:
    train_dir: str = "data/synth/train"
    val_dir: str = "data/synth/val"
    test_dir: str = "data/synth/test"
    work_dir: str = "runs/ablation"
    out_csv: str = "results/synth_ablation.csv"
    per_seed_csv: str = "results/synth_ablation_per_seed.csv"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    arms: list[str] = field(default_factory=lambda: list(ARMS))
    feature_groups: list[str] = field(default_factory=lambda: list(FEATURE_GROUPS))
    cycle_weight: float = 0.5
    epochs: int = 30
    batch_size: int = 8
    lr: float = 3e-4
    device: str = "auto"
    model: dict[str, Any] = field(default_factory=dict)
    n_boot: int = 2000
    ci_level: float = 0.95
    max_test_episodes: int | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> AblationConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        bad = sorted(set(raw) - {f.name for f in fields(cls)})
        if bad:
            raise ValueError(f"unknown ablation config keys: {bad}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if len(set(self.seeds)) < 3:
            raise ValueError("the ablation needs at least 3 distinct seeds")
        if sorted(set(self.arms) - set(ARMS)):
            raise ValueError(f"unknown arms {sorted(set(self.arms) - set(ARMS))}; use {ARMS}")
        if sorted(set(self.feature_groups) - set(FEATURE_GROUPS)):
            raise ValueError(f"unknown feature groups; use {FEATURE_GROUPS}")


@dataclass(frozen=True)
class Variant:
    """One reported row: which trained model (cycle weight, input group) and inference mode."""

    name: str
    cycle: bool
    group: str
    synchronize: bool


def variants(cfg: AblationConfig) -> list[Variant]:
    flags = {"pairwise": (False, False), "cycle": (True, False), "sync": (False, True),
             "both": (True, True)}  # fmt: skip
    out = [Variant(f"arm:{a}", *flags[a][:1], "full", flags[a][1]) for a in cfg.arms]
    out += [Variant(f"group:{g}", False, g, False) for g in cfg.feature_groups]
    return out


def _train_model(cfg: AblationConfig, cycle: bool, group: str, seed: int) -> Path:
    tag = f"{'cycle' if cycle else 'nocycle'}_{group}_seed{seed}"
    tc = TrainConfig(
        train_dir=cfg.train_dir,
        val_dir=cfg.val_dir,
        out_dir=str(Path(cfg.work_dir) / tag),
        results_csv=str(Path(cfg.work_dir) / tag / "history.csv"),
        run_name=tag,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        cycle_weight=cfg.cycle_weight if cycle else 0.0,
        feature_group=group,
        seed=seed,
        device=cfg.device,
        model=dict(cfg.model),
    )
    train(tc)
    return Path(tc.out_dir) / "best.pt"


def _evaluate(
    ckpt: Path, group: str, synchronize: bool, cases: list[tuple[Episode, Any, Any]], device: str
) -> dict[str, list[float]]:
    assoc = V3Associator(
        model=load_checkpoint(ckpt, "cpu" if device == "auto" else device),
        synchronize=synchronize,
        feature_group=group,
        device="cpu" if device == "auto" else device,
    )
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for ep, lifted, recon in cases:
        row = score_result(ep, lifted, assoc.associate(lifted, recon, None))
        for m in METRICS:
            out[m].append(float(getattr(row, m)))
    return out


def run_ablation(cfg: AblationConfig) -> tuple[Path, Path]:
    """Train every distinct configuration per seed, evaluate every variant on the test episodes,
    and write the per-seed and summary CSVs. Returns (summary_path, per_seed_path)."""
    cfg.validate()
    files = sorted(Path(cfg.test_dir).glob("*.npz"))[: cfg.max_test_episodes]
    if not files:
        raise FileNotFoundError(f"no .npz episodes in {cfg.test_dir}")
    cases = []
    for p in files:
        ep = load_episode(p)
        lifted, recon = episode_to_lifted(ep)
        cases.append((ep, lifted, recon))

    vs = variants(cfg)
    ckpts: dict[tuple[bool, str, int], Path] = {}
    evals: dict[tuple[bool, str, bool, int], dict[str, list[float]]] = {}
    per_ep: dict[tuple[str, int], dict[str, list[float]]] = {}
    for seed in cfg.seeds:
        for v in vs:
            tkey = (v.cycle, v.group, seed)
            if tkey not in ckpts:
                ckpts[tkey] = _train_model(cfg, v.cycle, v.group, seed)
            ekey = (*tkey[:2], v.synchronize, seed)
            if ekey not in evals:
                evals[ekey] = _evaluate(ckpts[tkey], v.group, v.synchronize, cases, cfg.device)
            per_ep[(v.name, seed)] = evals[ekey]

    per_seed_rows = []
    for v in vs:
        for seed in cfg.seeds:
            r = per_ep[(v.name, seed)]
            per_seed_rows.append(
                {"variant": v.name, "seed": seed, "n_episodes": len(cases)}
                | {m: _nanmean(r[m]) for m in METRICS}
            )
    summary = []
    for v in vs:
        row: dict[str, Any] = {"variant": v.name, "n_seeds": len(cfg.seeds),
                               "n_episodes": len(cases)}  # fmt: skip
        for m in METRICS:
            mat = np.array([per_ep[(v.name, s)][m] for s in cfg.seeds], dtype=float)
            seed_means = np.array([_nanmean(r) for r in mat])
            ep_means = _nanmean_cols(mat)
            row[f"{m}_mean"] = (
                float(np.nanmean(seed_means)) if np.isfinite(seed_means).any() else np.nan
            )
            row[f"{m}_std"] = float(np.nanstd(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            lo = hi = float("nan")
            if np.isfinite(ep_means).any():
                ci = bootstrap_ci(ep_means.tolist(), n_boot=cfg.n_boot, level=cfg.ci_level)
                lo, hi = ci.low, ci.high
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = lo, hi
        summary.append(row)
    return _write(cfg.out_csv, summary), _write(cfg.per_seed_csv, per_seed_rows)


def _nanmean(x: Any) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _nanmean_cols(mat: np.ndarray) -> np.ndarray:
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=0)
    tot = np.where(ok, mat, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _write(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p
