# Deviations

Where the real code or an engineering judgment call differs from `docs/PLAN.md`.
One entry per decision: date, what the plan said, what was done, why.

## 2026-11-24: Bootstrap

- **Python version:** plan says Python 3.11. `pyproject.toml` declares `requires-python = ">=3.11"`
  and tool targets of 3.11; the dev machine's system Python is 3.12, which is compatible.
- **Packaging:** setuptools backend with a flat `countonce/` package; dev tools under the
  `dev` optional extra. Runtime deps limited to numpy and pydantic until later tasks need more.
- **Subpackages:** created `ingest`, `detect`, `recon`, `merge`, `export`, `schemas`, `match`
  (names from PLAN §8 and §10) as empty packages; `py.typed` added for mypy consumers.
- **Lint rules:** ruff `E,F,I,UP,B,W`, line length 100. mypy runs in strict mode with the pydantic plugin.
- **Tooling check:** ruff and pytest verified via the read-only `~/vggtProject/.venv`; mypy is not installed there and the system Python lacks venv/pip, so mypy config is unverified. Install with `pip install -e .[dev]` once a venv exists.

## 2026-11-24: Port from vggtProject
- `Recon` from vggt-sparse-recon lives in `countonce/recon/result.py` (the ported, method-level reconstruction result). The §7 data-model `Recon` in `countonce/types.py` is a separate task and may wrap or replace it.
- Memory guard (`profiling.py`) placed in `countonce/recon/profiling.py`; `torch` is imported at module top, so it needs the `recon` extra.
- Plotting: ported only generic helpers (style, `save`, `depth_to_rgb`, `point_cloud_preview`, `vram_vs_frames`). Benchmark-specific plots (`band_lines`, `scene_type_bars`) were left behind; they belong to Track B.
- Env check: Python pin relaxed from `==3.11` to `>=3.11` to match `pyproject.toml`.
- Frame extraction: dropped the unused `EXAMPLES` constant. `configs/colmap.yaml` copied verbatim.
- Added `recon` optional-dependency extra and `pythonpath = ["."]` for pytest. `ruff`/`pytest` were run from `~/vggtProject/.venv` (3.11, has torch/pycolmap) since countonce has no venv yet.
- Video-clip helper (`clip.py`) was not ported: it is demo-site specific and not in the task list.
