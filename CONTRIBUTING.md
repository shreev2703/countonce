# Contributing to CountOnce

## Dev setup (uv)

```bash
git clone https://github.com/shreev2703/countonce.git && cd countonce
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev,test]"      # add ,recon for torch/pycolmap and ,mcp for the MCP server
pre-commit install                   # hooks are local to this clone
cp .env.example .env                 # only needed for VLM baselines
```

Plain `python -m venv` and `pip install -e .[dev,test]` work too. For a CPU-only torch, install it
from `https://download.pytorch.org/whl/cpu` first (the CI `full-cpu` job does this).

## Checks before a PR

```bash
ruff check . && ruff format --check .
mypy countonce                       # strict
pytest -m "not slow and not gpu"     # what hosted CI runs
```

`pre-commit run --all-files` runs the lint and type hooks.

## Test markers

| Marker | Meaning |
|---|---|
| `slow` | long-running; excluded from hosted CI |
| `gpu` | needs CUDA; run locally or on the self-hosted runner |
| `golden` | CPU golden synthetic-geometry regression tests |

Tests that need optional packages (torch, matplotlib, cv2, pycolmap, mcp) skip when they are
missing. `scripts/check_skips.py` fails CI on any other skip reason.

## Never make up numbers

Every metric, table cell, figure and claim in the README, docs, model card and datasheet comes
from a script that actually ran, or reads `TODO: fill from results`. Do not paste numbers from
memory or from a paper into a results table. Test fixtures may use toy values, but never present
them as results. PRs that add unmeasured numbers will be asked to revert them.

## Environment variables

Any variable the code reads must be listed in `.env.example` with a placeholder value.
`tests/test_env_example.py` fails otherwise. Never commit secrets.

## Adding a detector

1. Implement the `Detector` protocol from `countonce/detect/base.py` in `countonce/detect/<name>.py`
   (`name`, `version`, `config()`, `detect(frames, prompt, exemplars)`). Return all raw detections
   with scores; thresholds are applied later. Import heavy packages (torch) lazily.
2. Register it in `build_detector` in `countonce/pipeline.py` and add it to the `--detector`
   choices in `countonce/cli.py` (and to `Config` if it needs new options).
3. Test it with a fake or fixture input. No test may need weights or a GPU unless marked `gpu`.
4. If weights are gated or non-commercial, say so in `docs/DATASETS.md` / `docs/MODEL_SURVEY.md`.

A reconstruction backbone is the same pattern via `countonce/recon/adapter.py` (`load_method`).

## Adding a baseline

1. Add the function to `bench/baselines.py` and register it in `BASELINES`.
2. Produce its per-scene predictions in the format of `bench/schemas/predictions.schema.json`
   (`method`, `kind`, `split`, `scenes[]` with `scene_id`, `count`, optional `interval` and
   `assignments`) and score them with `python -m bench.evaluate --predictions preds.json --gt <split>`.
3. Only frozen splits (`bench/splits.json`) are accepted by `bench/leaderboard.py`.
4. Add a test with toy predictions that have a known score.

## Benchmark submissions

Open a "Benchmark submission" issue, or a PR adding your predictions file. Validate it locally
with `bench.evaluate` first.

## Commits and PRs

Small, focused commits with a message that says what and why. Fill in the PR template. Note any
departure from `docs/PLAN.md` in `docs/DEVIATIONS.md`.

## License

The repository does not have a license file yet; the choice is pending with the maintainer.
Please do not assume one until it is added.
