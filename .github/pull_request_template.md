## What and why

<!-- One or two sentences. Link the issue or PLAN section if there is one. -->

## Checklist

- [ ] `ruff check .`, `ruff format --check .`, `mypy countonce` and `pytest -m "not slow and not gpu"` pass locally
- [ ] New behaviour has a test (mark it `slow` / `gpu` / `golden` if it needs those)
- [ ] No invented numbers: every metric, table cell or claim comes from a script that ran, or says `TODO: fill from results`
- [ ] No secrets, datasets, weights or large binaries added
- [ ] New environment variables are listed in `.env.example`
- [ ] Docs updated (README, `docs/`, `docs/DEVIATIONS.md` if the PLAN was deviated from)

## How it was tested

<!-- Commands and results. Say what was NOT run (for example GPU tests). -->
