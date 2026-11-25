# Profiling report

Skeleton. Every cell is `TODO: fill from results` until `countonce profile` runs on the target
GPU. See PLAN section 13. Related: [[HANDOFF]], [[DECISIONS]].

## How to reproduce

```
countonce profile --config configs/profile.yaml   # writes results/profile.csv
```

Columns: `repeat, stage, wall_s, peak_vram_mb, n_frames, backbone, detector, assoc, cache_state,
recon_cache_hit, det_cache_hit`. The last three are append-only and empty except for
`countonce profile --variant cache` rows (cold / new_prompt / repeat; flags are 0 or 1).
`peak_vram_mb` is empty without CUDA, and only `recon` and `total` carry a value.

## Baseline breakdown

Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 616.92, WSL2), torch 2.11.0+cu128,
real `facebook/VGGT-1B` weights (`HF_HUB_OFFLINE=1`),
Frames: 10 (the largest measured count). Medians of 3 repeats after 1 warmup, from
`scripts/profile_gpu_baseline.py` (tiny synthetic scene, 128x96 images, dedup off).

Detect is the **color_blob** backend (a colour-threshold stand-in, not CountGD), so its time
says nothing about a real detector. `lift` is not timed as its own stage by the pipeline; it is
inside `associate`. `confidence` is the sensitivity-interval stage.

| Stage | Wall time (s, median) | Share of total | Peak VRAM (MB) |
|---|---|---|---|
| ingest | 0.018 | 0.1% | n/a |
| detect (color_blob) | 0.036 | 0.2% | n/a |
| recon (VGGT-1B) | 20.957 | 88.7% | 4834.2 |
| associate (incl. lift) | 0.496 | 2.1% | n/a |
| confidence | 1.992 | 8.4% | n/a |
| total | 23.628 | 100% | 4834.2 |

### By frame count

| Frames | ingest (s) | detect (s) | recon (s) | associate (s) | confidence (s) | total (s) | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| 3 | 0.008 | 0.009 | 17.751 | 0.143 | 0.613 | 18.520 | 3494.1 |
| 5 | 0.008 | 0.014 | 26.213 | 0.250 | 1.082 | 27.579 | 3987.6 |
| 10 | 0.018 | 0.036 | 20.957 | 0.496 | 1.992 | 23.628 | 4834.2 |

The bundled tiny scene has 5 views, so the 10-frame run cycles each view twice (identical
images); recon time and VRAM depend on frame count, not content. No OOM occurred up to 10
frames on the 8 GB card. Recon wall time is **not monotonic** in frame count (per-repeat recon:
3 frames 23.9/17.7/17.8 s, 5 frames 26.2/26.2/18.0 s, 10 frames 22.3/20.6/21.0 s), so at these
sizes it is not dominated by frame count; the cause is not yet isolated (TODO: fill from
results). Peak VRAM rises steadily with frames. Correctness check: `countonce run examples/tiny_scene --backbone vggt --detector color_blob`
(default dedup, which kept 1 usable frame of 5) and the fixture backbone both gave count 5
(ground truth 5).

License: the VGGT-1B weights are released under a **non-commercial** license; check the model
card before any commercial use of results or of a service built on them.

## Optimization before/after

| Change | Stage | Before (s) | After (s) | Speedup | Peak VRAM before/after (MB) |
|---|---|---|---|---|---|
| SDPA / FlashAttention | recon | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| bf16 | recon | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| `torch.compile` | recon | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| Stage caching | all | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |

## OOM fallback ladder

Rungs: 0 initial, 1 free cache, 2 smaller chunks, 3 lower resolution, then `ERR_GPU_OOM`
(`countonce/oom.py`). Rung reached per frame count:

| Frames | Rung that succeeded | Peak VRAM (MB) |
|---|---|---|
| TODO: fill from results | TODO: fill from results | TODO: fill from results |

## Activation checkpointing memory (matcher training)

`activation_checkpointing: none|attention|all` in `configs/match.yaml` recomputes the attention
(and, for `all`, the MLP) of each matcher block in the backward pass. It only applies in training
mode with grad enabled, so eval and inference are unchanged (PLAN §13 item 11).

CPU check, no GPU needed:

```
countonce train --config configs/match.yaml --profile-memory
```

It runs one forward and backward step per mode on the first training episode and prints
`saved` (MB of tensors autograd keeps for backward, counted with `saved_tensors_hooks`),
step time, and the CUDA peak when a GPU is used. `saved` should fall from `none` to `attention`
to `all`, while the extra recompute raises step time. Gradients match the baseline to 1e-5
(`tests/test_activation_checkpointing.py`).

| Mode | Saved activations CPU (MB) | Step time CPU (s) | Peak VRAM GPU (MB) | Max batch / views in 8-12 GB |
|---|---|---|---|---|
| none | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| attention | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| all | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |

<!-- profile-variants:start -->
## Matcher training speed (`countonce/match/train.py`)

Profile of 50 training episodes (batch 8, bf16 autocast, RTX 5070 Laptop, `.venv-gpu`) from
`configs/match_v3_capped.yaml` data, via `scripts/profile_match_train.py` (cProfile plus
`torch.profiler`; CUPTI is unavailable under WSL2 so the torch.profiler table has CPU times only).

**Before** (per-view-pair loop): 32.9 s wall for 50 episodes under cProfile. `log_sinkhorn` ran
once per view pair (689 calls, 41,340 `logsumexp` launches, 12.4 s cumulative); `as_tensor`
host-to-device copies took 3.5 s; backward 13.0 s, mostly the same tiny kernels.

| ncalls | tottime (s) | cumtime (s) | function |
|---:|---:|---:|---|
| 1 | 0.174 | 32.859 | `scripts/profile_match_train.py:18(run)` |
| 50 | 0.012 | 19.529 | `countonce/match/train.py:143(episode_terms)` |
| 50 | 0.001 | 17.687 | `countonce/match/model.py:275(pair_log_assignments)` |
| 50 | 0.034 | 17.686 | `countonce/match/model.py:277(<dictcomp>)` |
| 7 | 0.000 | 13.042 | `torch/_tensor.py:576(backward)` |
| 7 | 0.000 | 13.041 | `torch/autograd/__init__.py:253(backward)` |
| 7 | 0.000 | 13.040 | `torch/autograd/graph.py:860(_engine_run_backward)` |
| 7 | 13.040 | 13.040 | `{method 'run_backward' of 'torch._C._EngineBase' objects}` |
| 689 | 2.779 | 12.367 | `countonce/match/sinkhorn.py:17(log_sinkhorn)` |
| 41340 | 8.800 | 8.800 | `{built-in method torch.logsumexp}` |
| 739 | 0.070 | 5.282 | `countonce/match/model.py:261(_pairs)` |
| 5023 | 3.535 | 3.535 | `{built-in method torch.as_tensor}` |
| 689 | 0.520 | 1.420 | `countonce/match/model.py:251(_pair_scores)` |
| 50 | 0.011 | 1.419 | `countonce/match/train.py:154(<listcomp>)` |
| 689 | 0.711 | 1.306 | `countonce/match/model.py:300(pair_nll)` |
| 1828 | 0.026 | 0.905 | `torch/nn/modules/container.py:248(forward)` |
| 50 | 0.021 | 0.774 | `countonce/match/model.py:233(encode)` |
| 4106 | 0.027 | 0.768 | `torch/nn/modules/linear.py:130(forward)` |

**After** (all view pairs of an episode in one padded Sinkhorn; per-episode tensors, indices and
targets built once and kept on the device; epoch sums accumulated as device tensors instead of one
`.item()` per episode): 4.0 s wall for the same 50 episodes under cProfile.

| ncalls | tottime (s) | cumtime (s) | function |
|---:|---:|---:|---|
| 1 | 0.025 | 4.011 | `scripts/profile_match_train.py:25(run)` |
| 50 | 0.008 | 2.140 | `countonce/match/train.py:161(episode_terms)` |
| 50 | 0.004 | 2.039 | `countonce/match/model.py:404(pair_log_assignments_padded)` |
| 7 | 0.000 | 1.798 | `torch/_tensor.py:576(backward)` |
| 7 | 0.000 | 1.798 | `torch/autograd/__init__.py:253(backward)` |
| 7 | 0.000 | 1.796 | `torch/autograd/graph.py:860(_engine_run_backward)` |
| 7 | 1.796 | 1.796 | `{method 'run_backward' of 'torch._C._EngineBase' objects}` |
| 50 | 0.216 | 1.269 | `countonce/match/sinkhorn.py:45(log_sinkhorn_padded)` |
| 50 | 0.039 | 0.673 | `countonce/match/model.py:343(encode)` |
| 3250/400 | 0.009 | 0.649 | `torch/nn/modules/module.py:1775(_wrapped_call_impl)` |
| 3000 | 0.648 | 0.648 | `{built-in method torch.logsumexp}` |
| 3250/400 | 0.016 | 0.647 | `torch/nn/modules/module.py:1783(_call_impl)` |
| 200 | 0.001 | 0.554 | `countonce/match/model.py:323(forward)` |
| 400 | 0.002 | 0.553 | `countonce/match/model.py:317(_run)` |
| 200 | 0.039 | 0.436 | `countonce/match/model.py:304(_attn)` |
| 1550 | 0.009 | 0.277 | `torch/nn/modules/linear.py:130(forward)` |
| 1550 | 0.265 | 0.265 | `{built-in method torch._C._nn.linear}` |
| 550 | 0.006 | 0.261 | `torch/nn/modules/container.py:248(forward)` |
| 3150 | 0.195 | 0.195 | `{built-in method torch.where}` |
| 200 | 0.159 | 0.159 | `{built-in method torch._C._nn.scaled_dot_product_attention}` |

What changed: `log_sinkhorn_padded` (`countonce/match/sinkhorn.py`), `PairLayout` /
`build_layout` / `pair_nll_padded` (`countonce/match/model.py`), `episode_terms` and
`soft_count_padded` (`countonce/match/train.py`). The old per-pair path is kept as
`episode_terms_loop` and `pair_log_assignments_loop`; `tests/test_match_train_equiv.py` checks
losses and gradients agree within 1e-5 on CPU (8 seeded episodes, with and without the cycle loss).

Measured one epoch of the 800-episode set (plus the 100-episode val pass), `timeout 1500`:

| path | s / epoch | source |
|---|---:|---|
| before (per-pair loop) | 703.4 | `runs/match_v3_capped/log.jsonl`, epoch 1 `wall_s` |
| after (padded batch) | 85.4 | one-epoch run of `configs/match_v3_capped.yaml` with `epochs: 1` |

## Optimization variants

Generated by `scripts/make_profile_report.py` from the variants CSV; do not edit by hand. Speedup is baseline p50 / variant p50 of end-to-end wall time. A count delta of 0 means the variant counted the same as the baseline (accuracy proxy only; benchmark MAPE is separate).

TODO: fill from results (`/nonexistent.csv` not found; run `countonce profile` with `variants:` on the GPU).
<!-- profile-variants:end -->

<!-- profile-cache:start -->
## Repeat-job caching

Generated by `scripts/make_profile_report.py` from the profile CSV; do not edit by hand. Each repeat starts with empty caches: a cold job, the same photos with a new prompt (reconstruction reused), then an exact repeat (reconstruction and detections reused). Savings are cold p50 divided by the state's p50; GPU stages are detect + recon wall time. Measured, not assumed (PLAN section 13 item 6b).

TODO: fill from results (no cache-state rows in `/nonexistent2.csv`; run `countonce profile --variant cache` on the GPU).
<!-- profile-cache:end -->

## GroundingDINO-tiny (HF) detector sanity and runtime

Measured by `scripts/sanity_fuji_hf.py` (2026-11-24, RTX 5070 Laptop, `.venv-gpu`, fp32, eager),
`IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71`, prompt `apple`,
score threshold 0.3 (fixed before the run, not tuned), first 20 annotated images of the Fuji-SfM
mask-set `validation` split (no FSC-147 data on this machine). GT = polygon-instance count per image
(mean 13.25 apples/image).

| n images | MAE [95% bootstrap CI] | RMSE | per-image runtime, mean / median (warm, model load excluded) |
|---|---|---|---|
| 20 | 2.20 [1.10, 3.65] | 3.66 | 0.32 s / 0.33 s |

This checks the wrapper only, on 20 crops from one orchard; it is not a counting benchmark. Rows are
in `results/detector_sanity.csv` (gitignored).
