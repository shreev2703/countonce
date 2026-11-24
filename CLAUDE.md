# CountOnce — Autonomous Agent Rules

## Context & References
- Master spec: `docs/PLAN.md` (follow §4 Rules strictly).
- Reference repo (READ-ONLY — never modify): `~/vggtProject`
  - Port env check, VGGT wrapper, memory guard, frame extraction, COLMAP wrapper, and plotting style into `countonce/`. Credit the source in comments.
- Obsidian handoff file: `/mnt/c/Users/super_nga/second-brain/countonce-handoff.md`

## Autonomous Decision Policy
- You are running unattended overnight. **Never ask for confirmation or wait for input.**
- Whenever there is a design choice, ambiguity, or multiple valid options, **pick the recommended / cleanest engineering choice**, execute it, and log the decision in `docs/DEVIATIONS.md`.
- Never fabricate metrics or benchmark numbers. Use `TODO: fill from results` until a script actually runs.
- Never push to a remote repository overnight. Commit locally on the active branch only.

## Verification & Roadblock Policy
1. Work strictly on the single task assigned to this session.
2. Before committing, run `ruff check .` and `pytest` (if tests exist) and fix any failures.
3. Make one clean git commit when the task is complete.
4. Mark the task `[x]` in `TASKS_OVERNIGHT.md` and append a short handoff note (what was built, files changed, any decisions/next steps) to `/mnt/c/Users/super_nga/second-brain/countonce-handoff.md`.
5. **If you hit a genuine roadblock** (missing credentials, gated weights, broken external download, or 2 failed fix attempts): do NOT spin. Log the exact error and what you tried in `docs/GATES.md`, leave the repo clean (`git status` clean or WIP committed), and exit immediately so the orchestrator can move to the next task.
