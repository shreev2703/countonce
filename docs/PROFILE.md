# Profiling report

Skeleton. Every cell is `TODO: fill from results` until `countonce profile` runs on the target
GPU. See PLAN section 13. Related: [[HANDOFF]], [[DECISIONS]].

## How to reproduce

```
countonce profile --config configs/profile.yaml   # writes results/profile.csv
```

Columns: `repeat, stage, wall_s, peak_vram_mb, n_frames, backbone, detector, assoc`.
`peak_vram_mb` is empty without CUDA, and only `recon` and `total` carry a value.

## Baseline breakdown

Hardware: TODO: fill from results. Frames: TODO: fill from results.

| Stage | Wall time (s, median) | Share of total | Peak VRAM (MB) |
|---|---|---|---|
| ingest | TODO: fill from results | TODO: fill from results | n/a |
| detect | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| recon | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| associate | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| confidence | TODO: fill from results | TODO: fill from results | n/a |
| total | TODO: fill from results | 100% | TODO: fill from results |

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
