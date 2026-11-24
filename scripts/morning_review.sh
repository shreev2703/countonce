#!/usr/bin/env bash
cd ~/countonce
CLAUDE_CONFIG_DIR=~/.claude claude -p \
  "Review 'git diff main..HEAD', TASKS_OVERNIGHT.md, and docs/GATES.md against docs/PLAN.md.
  List: (1) completed tasks & test status, (2) any bugs, missing type hints, or deviations from PLAN.md, (3) exact next steps for me today." \
  --model opus
