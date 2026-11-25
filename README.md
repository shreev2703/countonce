# CountOnce

[![CI](https://github.com/shreev2703/countonce/actions/workflows/ci.yml/badge.svg)](https://github.com/shreev2703/countonce/actions/workflows/ci.yml)

Count objects in 3D from a few photos or a short video, without counting any object twice.

> **Status: draft.** The pipeline and benchmark harness are implemented and tested on synthetic
> and toy inputs. No real-data benchmark has been run yet, so every number, figure and table cell
> below is `TODO: fill from results`. Nothing here is a measured result.

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
- Detector adapters for CountGD / GroundingDINO have not been run against real weights in this repo
  yet; the CountGD forward pass is a stub until weights are available.

## Reproduce

Setup (Python ≥ 3.11; the reconstruction backbone needs a GPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,recon]"
countonce env --out results/env.json     # check Python / torch / CUDA / VRAM
```

Count objects in a folder of photos or a video:

```bash
countonce run --images path/to/photos --prompt "apple" --assoc v2 --out results/run1
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

Flags: `--detector {countgd,gdino,countgdpp}`, `--backbone {vggt,vggt_omega,vggt_t3,colmap}`,
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

| Method | k | MAE (apples) | MAPE (%) | assoc F1 | duplicate rate | miss rate |
|---|---|---|---|---|---|---|
| countonce_v1 | 3 | 61.94 [21.74, 110.64] | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v1 | 5 | 67.25 [27.24, 114.88] | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v1 | 10 | 53.00 [39.81, 70.06] | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v1 | 20 | 99.62 [66.06, 129.38] | 72.0 [36.7, 110.9] | TODO: fill from results | TODO: fill from results | 0.954 [0.940, 0.969] |
| countonce_v2 | 3 | 63.06 [22.56, 112.27] | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v2 | 5 | 65.56 [24.94, 113.56] | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v2 | 10 | 39.62 [27.37, 51.62] | TODO: fill from results | TODO: fill from results | TODO: fill from results | TODO: fill from results |
| countonce_v2 | 20 | 101.50 [68.49, 132.89] | 53.1 [34.9, 71.5] | TODO: fill from results | TODO: fill from results | 0.955 [0.942, 0.969] |
| sum | 3 | 61.31 [21.44, 109.39] | TODO: fill from results | n/a | n/a | n/a |
| sum | 5 | 68.19 [28.00, 115.69] | TODO: fill from results | n/a | n/a | n/a |
| sum | 10 | 68.31 [51.87, 88.82] | TODO: fill from results | n/a | n/a | n/a |
| sum | 20 | 156.94 [119.61, 202.25] | 140.5 [79.8, 216.9] | n/a | n/a | n/a |
| max | 3 | 66.56 [23.75, 119.31] | TODO: fill from results | n/a | n/a | n/a |
| max | 5 | 60.12 [17.62, 110.56] | TODO: fill from results | n/a | n/a | n/a |
| max | 10 | 31.19 [13.81, 50.63] | TODO: fill from results | n/a | n/a | n/a |
| max | 20 | 153.75 [110.12, 186.94] | 67.9 [59.3, 76.0] | n/a | n/a | n/a |
| median | 3 | 72.62 [26.81, 127.26] | TODO: fill from results | n/a | n/a | n/a |
| median | 5 | 57.88 [14.44, 111.44] | TODO: fill from results | n/a | n/a | n/a |
| median | 10 | 43.81 [21.18, 67.75] | TODO: fill from results | n/a | n/a | n/a |
| median | 20 | 185.00 [139.25, 219.19] | 91.0 [86.9, 94.4] | n/a | n/a | n/a |

Paired MAE difference over trees (a - b, negative favours a):

| a vs b | k | diff [95% CI] | p (bootstrap) | p (Wilcoxon) |
|---|---|---|---|---|
| v2 vs countonce_v1 | 3 | 1.12 [0.19, 2.31] | 0.016 | 0.125 |
| v2 vs sum | 3 | 1.75 [0.19, 4.00] | 0.014 | 0.125 |
| v2 vs max | 3 | -3.50 [-7.62, 0.00] | 0.052 | 0.125 |
| v2 vs median | 3 | -9.56 [-16.38, -2.88] | 0.002 | 0.055 |
| v2 vs countonce_v1 | 5 | -1.69 [-3.56, 0.25] | 0.106 | 0.250 |
| v2 vs sum | 5 | -2.62 [-5.13, 0.06] | 0.058 | 0.156 |
| v2 vs max | 5 | 5.44 [-1.56, 13.19] | 0.148 | 0.375 |
| v2 vs median | 5 | 7.69 [-4.26, 20.25] | 0.242 | 0.461 |
| v2 vs countonce_v1 | 10 | -13.38 [-22.31, -5.00] | 0.000 | 0.039 |
| v2 vs sum | 10 | -28.69 [-45.88, -11.37] | 0.002 | 0.039 |
| v2 vs max | 10 | 8.44 [-8.06, 24.64] | 0.284 | 0.312 |
| v2 vs median | 10 | -4.19 [-27.25, 22.44] | 0.710 | 0.844 |
| v2 vs countonce_v1 | 20 | 1.88 [-25.25, 19.62] | 0.796 | 0.383 |
| v2 vs sum | 20 | -55.44 [-122.14, -1.24] | 0.042 | 0.250 |
| v2 vs max | 20 | -52.25 [-79.81, -22.37] | 0.006 | 0.023 |
| v2 vs median | 20 | -83.50 [-110.81, -51.72] | 0.000 | 0.016 |
<!-- fuji-core:end -->
