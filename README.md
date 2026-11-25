# CountOnce

[![CI](https://github.com/shreev2703/countonce/actions/workflows/ci.yml/badge.svg)](https://github.com/shreev2703/countonce/actions/workflows/ci.yml)

Count objects in 3D from a few photos or a short video, without counting any object twice.

> **Status: draft.** The pipeline and benchmark harness are implemented and tested on synthetic
> and toy inputs. The real path (HF GroundingDINO, VGGT, v1/v2) has been run on the tiny scene and
> on the Fuji-SfM Core test trees; see `docs/STATUS.md` for exactly what each component has and
> has not been tested on. Figures and table cells without a measured value stay
> `TODO: fill from results`.

![Headline figure: MAPE vs. number of views](TODO: fill from results)

## Key findings

- `TODO: fill from results` (headline accuracy vs. the sum/max/median baselines)
- `TODO: fill from results` (effect of ordered vs. shuffled photos)
- `TODO: fill from results` (v1 vs. v2 vs. sync association)
- `TODO: fill from results` (at least one weakness, e.g. the worst error slice)

## Demo

Demo link: `TODO: fill from results`. Screenshot: `TODO: fill from results`.

## Results

Methods × views on the test split. Cells are filled by `countonce bench --figures` output,
never by hand.

| Method | Views | MAPE | Within 10% | Association F1 | Runtime |
|---|---|---|---|---|---|
| sum (baseline) | 3 / 5 / 10 / 20 / 40 | TODO: fill from results | TODO: fill from results | n/a | TODO: fill from results |
| max (baseline) | 3 / 5 / 10 / 20 / 40 | TODO: fill from results | TODO: fill from results | n/a | TODO: fill from results |
| median (baseline) | 3 / 5 / 10 / 20 / 40 | TODO: fill from results | TODO: fill from results | n/a | TODO: fill from results |
| CountOnce v1 (VGGT) | 3 / 5 / 10 / 20 / 40 | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| CountOnce v2 (VGGT) | 3 / 5 / 10 / 20 / 40 | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| CountOnce sync (VGGT) | 3 / 5 / 10 / 20 / 40 | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| CountOnce v1 (COLMAP) | 3 / 5 / 10 / 20 / 40 | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |

Confidence intervals are scene-level bootstrap CIs (`bench/stats.py`).
Sensitivity-range coverage: `TODO: fill from results`.

## How it works

```
photos / video
   │  ingest      drop blurry and near-duplicate frames, cap frame count
   ▼
open-vocabulary detector (per frame)      "apple" → boxes + scores        countonce/detect/
   │
   ▼
feed-forward reconstruction (VGGT)        cameras + depth + point maps    countonce/recon/
   │
   ▼
lift each detection to a 3D point + size  countonce/lift.py
   │
   ▼
associate detections across views         v1 3D clustering | v2 reprojection + Hungarian | sync
   │  (two detections in one photo are never the same object)
   ▼
count = number of instances, plus a sensitivity range, quality warnings,
overlays, counts.csv, scene.ply, result.json                              countonce/export/
```

A longer walk-through is in [docs/EXPLAINER.md](docs/EXPLAINER.md). The full plan is in
[docs/PLAN.md](docs/PLAN.md); deviations from it are logged in
[docs/DEVIATIONS.md](docs/DEVIATIONS.md).

## Limitations

- **Static scenes only.** Objects that move between photos break the geometry and are counted
  twice or missed. Moving-object detection is not implemented (`possibly_moving` stays at its default).
- **Chunk drift.** Captures larger than the backbone's frame budget are reconstructed in
  overlapping chunks and chained with Sim(3) alignment. Errors accumulate with the chunk count;
  there is no pose-graph optimization yet (see `docs/DEVIATIONS.md`).
- **Relative scale.** VGGT output has no real-world units. All thresholds are relative: merge
  distance is measured in object sizes, and scene scale is the median camera-to-scene distance.
- **The sensitivity range is not coverage.** The reported interval is the 10th–90th percentile of
  counts over a grid of merge strictness and detector thresholds. It shows how fragile the count
  is. It is not a statistical guarantee that the true count lies inside it
  (`confidence_method="sensitivity"`).
- Quality warnings and confidence scores are heuristics and are not calibrated.
- The default detector is HuggingFace GroundingDINO (text prompts only). CountGD and CountGD++ are
  planned, not implemented: selecting them exits with an error. Per-component evidence is in
  `docs/STATUS.md`, which wins if this README disagrees.

## Reproduce

Setup (Python ≥ 3.11; the reconstruction backbone needs a GPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,recon]"
countonce env --out results/env.json     # check Python / torch / CUDA / VRAM
```

Count objects in a folder of photos or a video:

```bash
countonce run --images path/to/photos --prompt "apple" --assoc v2 --out results/run1   # defaults: gdino + vggt
countonce run --video path/to/clip.mp4 --prompt "apple" --fps 2 --out results/run1
```

Or from Python (`import countonce` is light; torch loads only when `run` is first used):

```python
from pathlib import Path

import countonce

cfg = countonce.Config(out_dir="results/run1")
result = countonce.run(Path("path/to/photos"), "apple", cfg)  # -> countonce.CountResult
print(result.count, result.interval)
again = countonce.load_result("results/run1")  # reads results/run1/result.json
```

No GPU or weights? `examples/tiny_scene/` bundles five rendered views of five red spheres (plus
blue and green distractors) with known cameras and depth. The test-only `color_blob` detector and
`fixture` backbone count it on CPU (expected: 5):

```bash
countonce run examples/tiny_scene --prompt "red sphere" --detector color_blob --backbone fixture --assoc v2 --out results/tiny
```

These two backends exist only to exercise the CLI end to end (`tests/test_cli_e2e.py`); they are
not real detection or reconstruction. Regenerate the scene with `python examples/tiny_scene/generate.py`.

Flags: `--detector {gdino,color_blob}` (`countgd` and `countgdpp` are planned and exit with an error), `--backbone {vggt,vggt_omega,vggt_t3,colmap}`,
`--assoc {v1,v2,v3,sync}`, `--config file.yaml` (flags override the file). Not every choice has an
implementation yet; unimplemented ones exit with an error.

Download the Fuji-SfM dataset (about 6.9 GB). It is CC BY-NC-SA 4.0: research and education only, no commercial use. Do not redistribute its images or any GT derived from them, and do not upload them to a Hugging Face dataset. The loader reads `data/fuji_sfm` (gitignored):

```bash
scripts/download_fuji_sfm.sh data/raw/fuji_sfm
```

Run the benchmark. Scenes go in `data/scenes/<id>/{images/,gt.json}`:

```bash
countonce bench --config configs/bench.yaml --dry-run     # print the job grid size
countonce bench --config configs/bench.yaml               # run the sweep -> results/bench.csv
countonce bench --config configs/bench.yaml --figures results/figures
```

Tests and lint:

```bash
ruff check .
pytest -m "not slow and not gpu"
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, test markers and how to add a detector or baseline.

Install the git hooks once per clone (they are local to this repository, not global):

```bash
pip install pre-commit && pre-commit install
```

On commit they run ruff, `ruff format --check`, end-of-file / trailing-whitespace fixers and
`mypy countonce` (strict, a local hook that needs the dev extras). Run them on demand with
`pre-commit run --all-files`.

Schema compatibility fixtures live in `tests/fixtures/schemas/v<version>/`. After a
`schema_version` bump, add an upgrader and run `python scripts/regen_schema_fixture.py`; it writes
only the newest version's directory and refuses to overwrite an existing one.

## Credits, citations and licenses

This repo ships download scripts, not datasets or model weights. Verify each license before use.

| Item | Role | License status |
|---|---|---|
| VGGT (Wang et al., CVPR 2025) | Reconstruction backbone | Original checkpoint is non-commercial; a commercial variant exists by application |
| VGGT-Ω, VGG-T³ | Optional backbones (not wired) | Research only / `TODO: verify` |
| CountGD (Amini-Naieni et al.) | Open-vocabulary counting detector | MIT |
| CountGD++, CountVid, GroundingDINO, DINOv2 | Detector / video baselines / appearance features | `TODO: verify` each (see `docs/MODEL_SURVEY.md`) |
| Fuji-SfM | Benchmark scenes | CC BY-NC-SA 4.0 (research/education only): no redistribution of its images or derived GT, no HF dataset upload |
| WGISD, MinneApple, AppleGrowthVision, FSC-147 | Other candidate datasets | See `docs/DATASETS.md` |
| Kubric | Planned synthetic data | Open source; asset licenses to be checked |
| Pachauri et al., NeurIPS 2013 | Spectral permutation synchronization (`countonce/merge/sync.py`) | Paper |

Reconstruction wrapper, memory guard, frame extraction, COLMAP wrapper and plotting style are
ported from the author's VGGT benchmark repo (`TODO: fill from results` — add its URL once
public). The project's own license: `TODO: choose a license before release`.

Related work and open novelty questions: [docs/RELATED_WORK.md](docs/RELATED_WORK.md).

<!-- fuji-core:begin -->
### Fuji-SfM Core (test trees 01-08)

Generated by `python -m bench.fuji_report`; details in `docs/RESULTS_FUJI.md`.

| Method | k | MAE (apples) | signed bias (apples) | MAPE (%) | assoc F1 (explains errors) | duplicate rate | miss rate |
|---|---|---|---|---|---|---|---|
| countonce_v1 | 3 | 61.62 [21.68, 110.14] | -54.25 [-107.20, -12.94] | 292.0 [42.8, 765.9] | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v1 | 5 | 67.25 [27.24, 114.82] | -33.75 [-96.25, +17.26] | 1036.0 [157.0, 2207.1] | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v1 | 10 | 52.62 [39.37, 69.88] | +39.88 [+19.75, +61.19] | 315.3 [103.2, 597.7] | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v1 | 20 | 100.00 [66.12, 130.14] | -3.62 [-52.44, +54.06] | 73.4 [36.4, 113.4] | TODO: fill from results | TODO: fill from results | 0.955 [0.941, 0.970] |
| countonce_v2 | 3 | 62.75 [22.38, 111.77] | -55.88 [-109.06, -14.56] | 279.2 [45.9, 724.0] | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v2 | 5 | 65.69 [25.06, 113.51] | -36.31 [-98.08, +14.45] | 942.3 [148.4, 2022.0] | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v2 | 10 | 40.12 [28.00, 53.00] | +16.88 [-3.75, +37.12] | 204.6 [68.5, 384.6] | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v2 | 20 | 101.81 [68.23, 135.13] | -57.19 [-99.62, -7.62] | 54.7 [35.1, 74.2] | TODO: fill from results | TODO: fill from results | 0.956 [0.943, 0.971] |
| sum | 3 | 61.00 [21.12, 108.89] | -53.62 [-105.95, -12.37] | 291.7 [42.4, 765.7] | n/a | n/a | n/a |
| sum | 5 | 68.19 [28.06, 115.63] | -32.56 [-95.56, +19.69] | 1150.1 [156.9, 2478.4] | n/a | n/a | n/a |
| sum | 10 | 68.19 [51.93, 88.88] | +64.69 [+45.81, +85.94] | 413.3 [143.4, 756.1] | n/a | n/a | n/a |
| sum | 20 | 157.44 [120.18, 203.06] | +113.31 [+54.46, +183.03] | 143.0 [79.3, 220.5] | n/a | n/a | n/a |
| max | 3 | 66.25 [23.69, 118.75] | -62.25 [-118.00, -19.99] | 167.6 [50.4, 382.0] | n/a | n/a | n/a |
| max | 5 | 60.12 [17.68, 110.45] | -45.38 [-103.33, +2.44] | 498.0 [79.7, 1098.0] | n/a | n/a | n/a |
| max | 10 | 31.06 [14.06, 50.19] | -22.06 [-45.07, -1.06] | 74.7 [29.7, 142.7] | n/a | n/a | n/a |
| max | 20 | 153.75 [109.81, 187.25] | -153.75 [-187.25, -109.81] | 67.5 [58.6, 76.0] | n/a | n/a | n/a |
| median | 3 | 72.56 [26.81, 127.44] | -71.19 [-126.57, -25.37] | 146.2 [80.8, 265.6] | n/a | n/a | n/a |
| median | 5 | 57.88 [14.44, 111.50] | -53.38 [-109.00, -7.69] | 212.0 [96.3, 398.4] | n/a | n/a | n/a |
| median | 10 | 43.81 [21.37, 67.29] | -43.25 [-67.16, -20.63] | 66.3 [43.7, 85.6] | n/a | n/a | n/a |
| median | 20 | 185.00 [139.06, 219.47] | -185.00 [-219.47, -139.06] | 90.9 [86.7, 94.4] | n/a | n/a | n/a |

Paired MAE difference over trees (a - b, negative favours a):

| a vs b | k | diff [95% CI] | p (bootstrap) | p (Wilcoxon) |
|---|---|---|---|---|
| v2 vs countonce_v1 | 3 | 1.12 [0.25, 2.19] | 0.012 | 0.094 |
| v2 vs sum | 3 | 1.75 [0.31, 3.88] | 0.010 | 0.094 |
| v2 vs max | 3 | -3.50 [-7.69, 0.00] | 0.052 | 0.117 |
| v2 vs median | 3 | -9.81 [-16.88, -2.87] | 0.002 | 0.055 |
| v2 vs countonce_v1 | 5 | -1.56 [-3.44, 0.44] | 0.130 | 0.312 |
| v2 vs sum | 5 | -2.50 [-5.06, 0.31] | 0.078 | 0.156 |
| v2 vs max | 5 | 5.56 [-1.44, 13.31] | 0.132 | 0.375 |
| v2 vs median | 5 | 7.81 [-4.19, 20.38] | 0.234 | 0.461 |
| v2 vs countonce_v1 | 10 | -12.50 [-20.50, -4.62] | 0.000 | 0.031 |
| v2 vs sum | 10 | -28.06 [-44.38, -11.25] | 0.002 | 0.031 |
| v2 vs max | 10 | 9.06 [-7.44, 26.75] | 0.292 | 0.297 |
| v2 vs median | 10 | -3.69 [-27.01, 23.42] | 0.726 | 0.844 |
| v2 vs countonce_v1 | 20 | 1.81 [-26.07, 20.63] | 0.816 | 0.367 |
| v2 vs sum | 20 | -55.62 [-122.38, -0.25] | 0.044 | 0.250 |
| v2 vs max | 20 | -51.94 [-81.70, -20.25] | 0.006 | 0.023 |
| v2 vs median | 20 | -83.19 [-112.82, -50.69] | 0.000 | 0.016 |
<!-- fuji-core:end -->
