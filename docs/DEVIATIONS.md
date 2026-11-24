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
