"""Train CountOnce-Match v3 (PLAN §10.4, cycle consistency §10.3, tracking §10.6).

Loss per episode: pair NLL of the true partial assignment (with dustbin) + ``count_weight`` x
count-consistency + ``cycle_weight`` x triplet cycle consistency. AdamW, per-step cosine
schedule, gradient clipping, bf16 autocast on CUDA only. Everything random is derived from
``(seed, epoch)``, so a run resumed from ``last.pt`` reproduces the uninterrupted one.
The checkpoint ``best.pt`` is chosen by validation pairwise association F1 and loads with
``countonce.match.model.load_checkpoint``.

Each run writes ``<out_dir>/log.jsonl`` (config, git commit, data version, per-epoch metrics)
and a final per-epoch CSV in ``results/``. torch is imported lazily.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from countonce.match.model import (
    DetInputs,
    episode_inputs,
    make_model,
    mask_inputs,
    save_checkpoint,
)
from countonce.provenance import git_commit, git_is_dirty
from synth.geometry import Episode, load_episode


@dataclass
class TrainConfig:
    train_dir: str = "data/synth/train"
    val_dir: str = "data/synth/val"
    out_dir: str = "runs/match"
    results_csv: str = ""  # default: results/<run_name>.csv
    run_name: str = "match"
    data_version: str = ""  # default: hash of the episode file names and sizes
    epochs: int = 30
    batch_size: int = 8  # episodes per optimiser step
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    count_weight: float = 0.1
    cycle_weight: float = 0.0  # 0 disables the cycle loss (the "pairwise only" ablation)
    max_triplets: int = 8  # view triplets sampled per episode per step
    match_threshold: float = 0.5  # for validation association F1
    corrections: str = ""  # JSONL of consented correction pairs mixed in as hard pos/neg
    corrections_weight: float = 1.0
    feature_group: str = "full"  # 3d | 3d_rays | full (§10.5 attribution ablation)
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda
    resume: bool = True
    stop_after_epoch: int | None = None  # pause a run early (the cosine horizon stays `epochs`)
    model: dict[str, Any] = field(default_factory=dict)  # make_model kwargs

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        bad = sorted(set(raw) - known)
        if bad:
            raise ValueError(f"unknown train config keys: {bad}")
        return cls(**raw)


# ------------------------------------------------------------------ losses
def cycle_consistency_loss(p_ij: Any, p_jk: Any, p_ik: Any) -> Any:
    """Mean squared gap between ``P_ij @ P_jk`` and ``P_ik``.

    All three are the real-detection blocks of the assignments (dustbin row/column removed),
    so dustbin mass never enters. Zero when the three are consistent permutation matrices.
    """
    return ((p_ij @ p_jk - p_ik) ** 2).mean()


def _core(logp: Any) -> Any:
    return logp[:-1, :-1].exp()


def triplet_cycle_loss(
    logps: dict[tuple[int, int], Any], n_views: int, rng: np.random.Generator, max_triplets: int
) -> Any | None:
    """Cycle loss averaged over up to ``max_triplets`` sampled view triplets a<b<c."""
    triplets = [
        (a, b, c)
        for a in range(n_views)
        for b in range(a + 1, n_views)
        for c in range(b + 1, n_views)
        if {(a, b), (b, c), (a, c)} <= logps.keys()
    ]
    if not triplets:
        return None
    pick = rng.permutation(len(triplets))[:max_triplets]
    terms = [
        cycle_consistency_loss(_core(logps[(a, b)]), _core(logps[(b, c)]), _core(logps[(a, c)]))
        for a, b, c in (triplets[i] for i in pick)
    ]
    return sum(terms) / len(terms)


def soft_count(logps: dict[tuple[int, int], Any], views: np.ndarray) -> Any:
    """Differentiable instance count: sum_i 1 / (1 + sum_j P_ij) over detections."""
    import torch

    ids = sorted(set(views.tolist()))
    row = None
    for (a, b), logp in logps.items():
        core = _core(logp)
        if row is None:
            row = torch.zeros(len(views), device=core.device)
        ia, ib = np.flatnonzero(views == ids[a]), np.flatnonzero(views == ids[b])
        row = row.index_add(0, torch.as_tensor(ia, device=core.device), core.sum(1))
        row = row.index_add(0, torch.as_tensor(ib, device=core.device), core.sum(0))
    if row is None:
        return torch.zeros(())
    return (1.0 / (1.0 + row)).sum()


def episode_terms(
    model: Any, ep: Episode, d: DetInputs, cfg: TrainConfig, rng: np.random.Generator
) -> dict[str, Any] | None:
    """Loss terms for one episode, or None when it has no view pair to learn from."""
    from countonce.match.model import pair_nll

    logps = model.pair_log_assignments(d)
    if not logps:
        return None
    ids = sorted(set(ep.view.tolist()))
    nll = _stack_mean(
        [pair_nll(lp, *ep.pair_targets(ids[a], ids[b])) for (a, b), lp in logps.items()]
    )
    out: dict[str, Any] = {"nll": nll}
    if cfg.count_weight > 0:
        truth = max(ep.true_count(), 1)
        out["count"] = (soft_count(logps, d.view) - ep.true_count()).abs() / truth
    if cfg.cycle_weight > 0:
        cyc = triplet_cycle_loss(logps, len(ids), rng, cfg.max_triplets)
        if cyc is not None:
            out["cycle"] = cyc
    return out


def _stack_mean(xs: list[Any]) -> Any:
    import torch

    return torch.stack(xs).mean()


def _total(terms: dict[str, Any], cfg: TrainConfig) -> Any:
    loss = terms["nll"]
    if "count" in terms:
        loss = loss + cfg.count_weight * terms["count"]
    if "cycle" in terms:
        loss = loss + cfg.cycle_weight * terms["cycle"]
    return loss


# ------------------------------------------------------------------ data + validation
def load_split(
    directory: str | Path, feature_group: str = "full", appearance_dim: int = 0
) -> list[tuple[Episode, DetInputs]]:
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz episodes in {directory} (run `countonce synth` first)")
    return [
        (ep, mask_inputs(episode_inputs(ep), feature_group, appearance_dim))
        for ep in map(load_episode, files)
    ]


def data_version(*directories: str | Path) -> str:
    h = hashlib.sha1()
    for d in directories:
        for p in sorted(Path(d).glob("*.npz")):
            h.update(f"{p.name}:{p.stat().st_size};".encode())
    return h.hexdigest()[:12]


def pair_labels(ep: Episode) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (i, j), i<j, of cross-view detection pairs and their same-object labels."""
    iu, ju = np.triu_indices(ep.n_detections, 1)
    cross = ep.view[iu] != ep.view[ju]
    iu, ju = iu[cross], ju[cross]
    same = (ep.identity[iu] == ep.identity[ju]) & (ep.identity[iu] >= 0)
    return np.stack([iu, ju]), same


def collect_pair_probs(
    model: Any, split: list[tuple[Episode, DetInputs]]
) -> tuple[np.ndarray, np.ndarray]:
    """Flattened cross-view match probabilities and GT labels (input to calibration)."""
    import torch

    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for ep, d in split:
            (iu, ju), same = pair_labels(ep)
            p = model.match_probs(d).cpu().numpy()
            probs.append(p[iu, ju])
            labels.append(same)
    if not probs:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(probs), np.concatenate(labels)


def association_f1(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    tp = int((pred & labels).sum())
    denom = 2 * tp + int((pred & ~labels).sum()) + int((~pred & labels).sum())
    return 2 * tp / denom if denom else 1.0


def evaluate(
    model: Any, split: list[tuple[Episode, DetInputs]], cfg: TrainConfig
) -> dict[str, float]:
    import torch

    model.eval()
    rng = np.random.default_rng([cfg.seed, 99])
    nll, n = 0.0, 0
    with torch.no_grad():
        for ep, d in split:
            t = episode_terms(model, ep, d, cfg, rng)
            if t is not None:
                nll += float(t["nll"])
                n += 1
    probs, labels = collect_pair_probs(model, split)
    return {
        "val_nll": nll / max(n, 1),
        "val_f1": association_f1(probs, labels, cfg.match_threshold),
    }


# ------------------------------------------------------------------ training loop
def _device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _make_optim(model: Any, cfg: TrainConfig, total_steps: int) -> tuple[Any, Any]:
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    return opt, sched


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_results_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for h in history for k in h}, key=lambda k: (k != "epoch", k))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(history)


def train(cfg: TrainConfig) -> list[dict[str, Any]]:
    """Run (or resume) training; returns the per-epoch metric history."""
    import torch

    adim = int(cfg.model.get("appearance_dim", 0))
    train_set = load_split(cfg.train_dir, cfg.feature_group, adim)
    val_set = load_split(cfg.val_dir, cfg.feature_group, adim)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last, best = out / "last.pt", out / "best.pt"
    log = out / "log.jsonl"
    device = _device(cfg.device)
    use_amp = device == "cuda"
    version = cfg.data_version or data_version(cfg.train_dir, cfg.val_dir)
    groups: list[Any] = []
    if cfg.corrections:
        from countonce.match.correction_data import build_groups, load_pairs
        from countonce.match.registry import file_sha256

        groups = build_groups(load_pairs(cfg.corrections), cfg.feature_group, adim)
        version = f"{version}+corr-{file_sha256(cfg.corrections)[:8]}"

    torch.manual_seed(cfg.seed)
    model = make_model(**cfg.model).to(device)
    steps_per_epoch = math.ceil(len(train_set) / cfg.batch_size)
    opt, sched = _make_optim(model, cfg, cfg.epochs * steps_per_epoch)

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    start = 0
    if cfg.resume and last.exists():
        blob = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        history, best_f1, start = blob["history"], blob["best_f1"], blob["epoch"]
        _append_jsonl(log, {"event": "resume", "epoch": start})
    else:
        for stale in (last, best, log):
            stale.unlink(missing_ok=True)
        _append_jsonl(
            log,
            {
                "event": "run",
                "config": asdict(cfg),
                "git_commit": git_commit(),
                "git_dirty": git_is_dirty(),
                "data_version": version,
                "device": device,
            },
        )

    stop = cfg.epochs if cfg.stop_after_epoch is None else min(cfg.epochs, cfg.stop_after_epoch)
    for epoch in range(start, stop):
        model.train()
        rng = np.random.default_rng([cfg.seed, epoch])
        order = rng.permutation(len(train_set))
        sums = {"loss": 0.0, "nll": 0.0, "count": 0.0, "cycle": 0.0, "corr": 0.0}
        n_eps = 0
        n_corr = 0
        step_no = 0
        for s in range(0, len(order), cfg.batch_size):
            opt.zero_grad(set_to_none=True)
            batch_loss = []
            for i in order[s : s + cfg.batch_size]:
                ep, d = train_set[int(i)]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    terms = episode_terms(model, ep, d, cfg, rng)
                if terms is None:
                    continue
                loss = _total(terms, cfg)
                batch_loss.append(loss)
                n_eps += 1
                sums["loss"] += float(loss.detach())
                for k in ("nll", "count", "cycle"):
                    if k in terms:
                        sums[k] += float(terms[k].detach())
            if groups:
                from countonce.match.correction_data import correction_loss

                closs = correction_loss(model, groups[(epoch * 7919 + step_no) % len(groups)])
                batch_loss.append(cfg.corrections_weight * closs)
                sums["corr"] += float(closs.detach())
                n_corr += 1
                step_no += 1
            if batch_loss:
                _stack_mean(batch_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            sched.step()
        val = evaluate(model, val_set, cfg)
        rec: dict[str, Any] = {"epoch": epoch + 1, "lr": sched.get_last_lr()[0], **val}
        rec.update({f"train_{k}": v / max(n_eps, 1) for k, v in sums.items() if k != "corr"})
        if groups:
            rec["train_corr"] = sums["corr"] / max(n_corr, 1)
        history.append(rec)
        if rec["val_f1"] > best_f1:
            best_f1 = rec["val_f1"]
            save_checkpoint(model, best)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "history": history,
                "best_f1": best_f1,
                "epoch": epoch + 1,
            },
            last,
        )
        _append_jsonl(log, {"event": "epoch", **rec})

    if history and history[-1]["epoch"] == cfg.epochs:
        csv_path = Path(cfg.results_csv or f"results/{cfg.run_name}.csv")
        write_results_csv(csv_path, history)
        _fit_calibration(cfg, best, val_set, out / "calibration.json")
    return history


def _fit_calibration(
    cfg: TrainConfig, ckpt: Path, val_set: list[tuple[Episode, DetInputs]], path: Path
) -> None:
    """Temperature fitted on the validation split only, using the best checkpoint."""
    from countonce.match.calibration import (
        apply_temperature,
        ece,
        fit_temperature,
        reliability_data,
    )
    from countonce.match.model import load_checkpoint

    if not ckpt.exists():
        return
    model = load_checkpoint(ckpt, _device(cfg.device))
    p, y = collect_pair_probs(model, val_set)
    if p.size == 0:
        return
    t = fit_temperature(p, y)
    path.write_text(
        json.dumps(
            {
                "split": "val",
                "temperature": t,
                "ece_before": ece(p, y),
                "ece_after": ece(apply_temperature(p, t), y),
                "reliability_before": reliability_data(p, y).to_dict(),
                "reliability_after": reliability_data(apply_temperature(p, t), y).to_dict(),
            },
            indent=2,
        )
    )
