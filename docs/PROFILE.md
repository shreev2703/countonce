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
