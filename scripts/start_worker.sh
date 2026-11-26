#!/usr/bin/env bash
# One command to start the laptop GPU worker as a self-restarting service.
#
#   scripts/start_worker.sh                      # local sqlite store (./jobs.db), outputs next to it
#   COUNTONCE_STORE=postgres COUNTONCE_PG_URL=... COUNTONCE_STORAGE=s3 scripts/start_worker.sh
#
# Reads .env if present. Extra arguments go to `countonce worker` (e.g. --retention-days 14).
# Stop with Ctrl-C or SIGTERM: the current job finishes first. For boot-time start run
#   countonce worker --print-unit > ~/.config/systemd/user/countonce-worker.service
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
ARGS=(--supervise --db "${COUNTONCE_DB:-jobs.db}")
[ -n "${COUNTONCE_STORE:-}" ] && ARGS+=(--store "$COUNTONCE_STORE")
exec "$PY" -m countonce.cli worker "${ARGS[@]}" "$@"
