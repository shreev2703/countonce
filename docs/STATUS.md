# Component status

One row per component. Columns are evidence levels, strictest last. Cells say what has been
done, not what is hoped for. README claims must not go beyond this table; if they disagree, this
file wins and the README is the bug. Related: [[HANDOFF]], [[DECISIONS]], `docs/GATES.md`,
`docs/DEVIATIONS.md`.

- **Implemented**: code exists and the code path runs.
- **Synthetic-tested**: unit/integration tests on toy or rendered synthetic inputs (CPU).
- **Real-model-tested**: run once or more with real weights (GPU) on at least one real or tiny scene.
- **Held-out real benchmark**: scored on held-out real scenes (Fuji-SfM test trees, `results/bench.csv`).
  Own-capture scenes: none yet.

Legend: yes / no / planned (selecting it raises an error that says so) / n/a.

## Detectors

| Component | Config name | Implemented | Synthetic-tested | Real-model-tested | Held-out real benchmark |
|---|---|---|---|---|---|
| GroundingDINO via HuggingFace (`IDEA-Research/grounding-dino-tiny`) | `gdino` (**default**), `groundingdino` | yes | yes (fake model) | yes (tiny scene, GPU marked test) | yes (Fuji Core, v1 and v2 rows) |
| GroundingDINO original package path | `detector_options.backend: groundingdino` | yes | gating only | no | no |
| CountGD | `countgd` | **planned**: forward pass is a stub, weights gated | no | no | no |
| CountGD++ | `countgdpp` | **planned**: not implemented | no | no | no |
| Colour blob (test backend only) | `color_blob` | yes | yes | n/a | n/a |

Selecting `countgd` or `countgdpp` raises `PlannedComponentError` from `build_detector`; the CLI prints
the message and exits non-zero. It never silently falls back to another detector.

## Reconstruction backbones

| Component | Config name | Implemented | Synthetic-tested | Real-model-tested | Held-out real benchmark |
|---|---|---|---|---|---|
| VGGT-1B | `vggt` | yes | adapter only | yes (tiny scene, GPU marked test) | yes (Fuji Core) |
| COLMAP | `colmap` | yes | adapter only | no | no |
| Fixture (rendered-scene cameras and depth) | `fixture` | yes (test backend) | yes | n/a | n/a |
| VGGT-Omega, VGG-T3 | `vggt_omega`, `vggt_t3` | **planned** (names accepted, no adapter) | no | no | no |

## Association

| Component | Config name | Implemented | Synthetic-tested | Real-model-tested | Held-out real benchmark |
|---|---|---|---|---|---|
| v1 3D clustering | `v1` | yes | yes | yes (with the real detector and VGGT) | yes (Fuji Core) |
| v2 reprojection + Hungarian | `v2` | yes | yes | yes | yes (Fuji Core) |
| Permutation sync | `sync` | yes | yes | no | no |
| v3 learned matcher | `v3` | yes | yes (trained on synthetic data only) | no | no |

## Baselines, pipeline and tooling

| Component | Implemented | Synthetic-tested | Real-model-tested | Held-out real benchmark |
|---|---|---|---|---|
| sum / max / median baselines | yes | yes | yes | yes (Fuji Core) |
| Ingest, lift, export, `result.json` | yes | yes | yes (tiny scene) | yes (Fuji Core) |
| Sensitivity interval | yes | yes | yes | no (coverage not evaluated) |
| Conformal / jackknife+ intervals | yes | yes | no | no |
| Chunked reconstruction (chain, pose graph) | yes | yes | no | no |
| SAM 2 masks | yes | yes | yes (GPU marked test) | no |
| Benchmark harness, Fuji loader and report | yes | yes | yes | yes (Fuji Core only) |
| API, jobs, MCP server | yes | yes | no | n/a |

## Known open problem

The real detector + VGGT path over-counts the tiny scene (7 vs truth 5); it is tracked as task M2
in `TASKS_OVERNIGHT.md`. The synthetic-fixture path counts 5. Do not read the synthetic rows as
evidence about real-model accuracy.
