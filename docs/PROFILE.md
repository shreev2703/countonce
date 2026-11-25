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
## Optimization variants

Generated by `scripts/make_profile_report.py` from the variants CSV; do not edit by hand. Speedup is baseline p50 / variant p50 of end-to-end wall time. A count delta of 0 means the variant counted the same as the baseline (accuracy proxy only; benchmark MAPE is separate).

TODO: fill from results (`/nonexistent.csv` not found; run `countonce profile` with `variants:` on the GPU).
<!-- profile-variants:end -->

<!-- profile-cache:start -->
## Repeat-job caching

Generated by `scripts/make_profile_report.py` from the profile CSV; do not edit by hand. Each repeat starts with empty caches: a cold job, the same photos with a new prompt (reconstruction reused), then an exact repeat (reconstruction and detections reused). Savings are cold p50 divided by the state's p50; GPU stages are detect + recon wall time. Measured, not assumed (PLAN section 13 item 6b).

TODO: fill from results (no cache-state rows in `/nonexistent2.csv`; run `countonce profile --variant cache` on the GPU).
<!-- profile-cache:end -->
