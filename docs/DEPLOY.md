# Deploying CountOnce on Vercel

Two Vercel projects plus a worker that is not on Vercel. Related: [PLAN](PLAN.md) §0.2 (M18),
[DEVIATIONS](DEVIATIONS.md).

| Piece | Where | Root directory | Does |
|---|---|---|---|
| Site (`web/`) | Vercel project `countonce` (`countonce.<domain>`) | `web/` | Pages, sign-in, same-origin proxy `/api/backend/*` |
| API | Vercel project `countonce-api` (`api.<domain>`) | repo root | FastAPI at `countonce.api.asgi:app` (`[tool.vercel]` in `pyproject.toml`) |
| Worker | The GPU laptop (M19) | n/a | Runs the pipeline; reads queued jobs from the shared store |

**Why a separate API project, not `/api` in the site:** the site is Node, the API is Python with its
own dependency set, and a path inside the site would need rewrites plus a second runtime in one
project. The browser only ever calls the site; the site proxy forwards to `COUNTONCE_API_URL`, so
no CORS is needed.

## Long work never runs in a function

The API only stores jobs, receives upload chunks, signs file URLs and reports status. It imports no
torch, transformers, pycolmap or OpenCV (`tests/test_api_vercel.py` checks this). Reconstruction,
detection and counting run in the worker, which claims jobs from the shared Postgres store and
writes results to object storage. Without a worker, jobs stay `queued` and the UI shows the
paused-worker banner.

The API function needs persistent state, so set `COUNTONCE_PG_URL` and S3/R2 storage for real use.
The SQLite and temp-dir fallback is for smoke checks only.

## Live progress: SSE, then polling

`GET /jobs/{id}/events` is Server-Sent Events. A function has a maximum duration, so the stream
closes after `COUNTONCE_SSE_MAX_S` seconds (set about 50 on a 60 s function). The web client
(`web/src/lib/job-events.ts`) treats a closed stream as an error and falls back to polling
`GET /jobs/{id}` every 2 s until the job is finished. The proxy route sets `maxDuration = 60`.
Chunk uploads are 4 MiB, under the 4.5 MB Vercel request body limit, so they pass through both
projects.

## Environment variables

API project (Production and Preview):

| Variable | Required | Purpose |
|---|---|---|
| `COUNTONCE_PG_URL` | yes | Postgres job store shared with the worker |
| `COUNTONCE_AUTH_SECRET` | yes | Shared secret for `POST /auth/exchange`; same value as the site |
| `COUNTONCE_STORAGE`, `COUNTONCE_S3_*` | yes | `s3` object storage (R2) for uploads and results |
| `COUNTONCE_STORAGE_SECRET` | if `local` | Signs local file URLs |
| `COUNTONCE_SSE_MAX_S` | recommended | SSE cap in seconds (about 50) |
| `COUNTONCE_WEB_URL` | recommended | Where `/` redirects (the site origin); `/docs` if empty |
| `COUNTONCE_DATA_DIR` | no | Scratch root (default: system temp dir) |

Site project (`web/`):

| Variable | Required | Purpose |
|---|---|---|
| `COUNTONCE_API_URL` | yes | Origin of the API project, e.g. `https://api.<domain>` |
| `COUNTONCE_AUTH_SECRET` | yes | Same value as the API |
| `COUNTONCE_SESSION_SECRET` | yes | 16+ random characters; encrypts the session cookie |
| `COUNTONCE_PUBLIC_URL` | yes | Public origin for OAuth redirects and share previews |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | yes | Google sign-in (redirect URI `<origin>/api/auth/google/callback`) |
| `COUNTONCE_DEV_AUTH` | never in production | Test sign-in without Google |
| `COUNTONCE_RESULTS_DIR` | no | Benchmark CSV folder; default `content/benchmark` when `../results` is absent |

The benchmark page reads the committed `web/content/benchmark/benchmark_summary.csv`; refresh it
with `npm run sync:results` after `python -m bench.web_summary`.

## Steps (need your Vercel account; not done by the agent)

1. Create the API project from the repo root. Add the variables above. Deploy.
2. Check `https://api.<domain>/health` returns 200 and `/` redirects to the site.
3. Create the site project with root `web/`. Set `COUNTONCE_API_URL` to the API origin. Deploy.
4. Open `/status` on the site: it should show the API as reachable.
5. Start the worker against the same Postgres and storage, then run one job from `/new`:

   ```bash
   COUNTONCE_STORE=postgres COUNTONCE_PG_URL=... COUNTONCE_STORAGE=s3 scripts/start_worker.sh
   ```

   Local sqlite: `scripts/start_worker.sh` alone (`COUNTONCE_DB` picks the file). Reads `.env`.

### Worker service (M19)

- `scripts/start_worker.sh` runs `countonce worker --supervise`: a crash (CUDA fault, OOM kill)
  restarts the worker with 1 s to 60 s backoff; Ctrl-C/SIGTERM finishes the current job first.
- Boot-time start: `countonce worker --print-unit > ~/.config/systemd/user/countonce-worker.service`
  then `systemctl --user enable --now countonce-worker`.
- Heartbeat: running jobs beat every 15 s; an idle worker updates its `workers` row every 10 s, which
  is what `/health` and the site's paused banner read.
- Errors are typed (`ERR_*`); retryable ones back off, permanent ones go to `dead_letter`.
  `countonce jobs --db jobs.db dead` lists them, `... retry <job id>` requeues one.
- Retention: the worker purges finished jobs older than `--retention-days` (default 30, 0 disables)
  hourly. `/privacy` says so.

Local equivalent without Vercel: see `web/README.md` (fake worker + dev sign-in).
