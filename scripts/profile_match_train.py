"""Profile matcher training (cProfile + torch.profiler) on N episodes; prints top-20 tables."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np
import torch

from countonce.match.model import make_model
from countonce.match.train import (
    TrainConfig,
    _stack_mean,
    _total,
    episode_terms,
    load_split,
    make_layouts,
)


def run(model, eps, cfg, opt, device, lays):
    rng = np.random.default_rng(0)
    for s in range(0, len(eps), cfg.batch_size):
        opt.zero_grad(set_to_none=True)
        losses = []
        for k in range(s, min(s + cfg.batch_size, len(eps))):
            ep, d = eps[k]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                t = episode_terms(model, ep, d, cfg, rng, lays[k])
            if t is not None:
                losses.append(_total(t, cfg))
        if losses:
            _stack_mean(losses).backward()
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/match_v3_capped.yaml")
    ap.add_argument("--episodes", type=int, default=50)
    a = ap.parse_args()
    cfg = TrainConfig.from_yaml(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load_split(cfg.train_dir, cfg.feature_group, 0)[: a.episodes]
    model = make_model(**cfg.model).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lays = make_layouts(eps, device)
    run(model, eps[:8], cfg, opt, device, lays)  # warm-up
    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    run(model, eps, cfg, opt, device, lays)
    pr.disable()
    print(f"wall {time.perf_counter() - t0:.2f}s for {len(eps)} episodes on {device}")
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(20)
    print(buf.getvalue())
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with profile(activities=acts) as prof:
        run(model, eps, cfg, opt, device, lays)
    key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=20))


if __name__ == "__main__":
    main()
