# Serving: cloud worker cold start and cost

The cloud worker is the same image as the laptop worker (`docker/worker.Dockerfile`, PLAN 14.4). Weights are mounted at `/weights`, not baked in. The image has not been built yet.

Measured with `countonce coldstart`; rows append to `results/coldstart.csv`.

## Cold start

| Phase | Name | Seconds |
|---|---|---|
| Process start | interpreter | TODO: fill from results |
| Import | pipeline | TODO: fill from results |
| Model load | all models | TODO: fill from results |
| First warm inference | warm | TODO: fill from results |

## Warm latency

| p50 (s) | p95 (s) |
|---|---|
| TODO: fill from results | TODO: fill from results |

## Cost per 1,000 frames

One warm inference is counted as one frame; rate is the `cloud` `cost_per_hour` in `cost.yaml`.

| Rate (per GPU-hour) | Cost per 1,000 frames |
|---|---|
| TODO: set rate | TODO: fill from results |

## HTTP API load test

`countonce loadtest-api` (config `configs/loadtest_api.yaml`) starts the real FastAPI app with uvicorn in-process on a free localhost port, over a temporary sqlite store and `LocalBlobStore`, plus fake worker processes. Concurrent `httpx` clients each run create, chunked upload, submit, poll or SSE-follow, and fetch the result, with idempotency keys and deliberate duplicate create/submit pairs sent concurrently. A 429 is retried after `min(Retry-After, retry_after_cap_s)`. The clients share the process (and the GIL) with the server and the sqlite file with the workers, and the API key check is a 100k-iteration PBKDF2 per request, so absolute latencies are pessimistic and only comparable between runs on the same machine. Rows append to `results/api_load.csv` (gitignored; git sha and host recorded).

Host of the latest run: `DESKTOP-D9KBPSF`, 16 CPUs, Python 3.12.3, git `cc1bbe471332` (dirty tree).

| Run | Clients / users / workers | Request limit per user | Requests | req/s | 429s | p50 / p95 / p99 (ms) | Jobs done | Lost | Double-queued |
|---|---|---|---|---|---|---|---|---|---|
| `65de1a52` | 16 / 8 / 4 | 300 per 60 s | 865 | 17.2 | 0 | 583 / 2732 / 4437 | 80 of 80 | 0 | 0 |
| `a3e4d320` | 16 / 8 / 4 | 120 per 60 s | 860 | 16.9 | 0 | 565 / 3266 / 4545 | 80 of 80 | 0 | 0 |
| `fbfa96c6` | 16 / 8 / 4 | 40 per 60 s | 1686 | 16.7 | 824 | 119 / 1679 / 3340 | 80 of 80 | 0 | 0 |

Per route, run `fbfa96c6` (latency per request attempt, including 429s; the SSE row is time to the first event):

| Route | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Status counts |
|---|---|---|---|---|---|
| `POST /jobs` | 213 | 147 | 2613 | 4276 | `{"200": 26, "201": 80, "429": 107}` |
| `POST /uploads` | 158 | 123 | 1971 | 3998 | `{"201": 80, "429": 78}` |
| `PUT /uploads/{id}/chunks/{n}` | 597 | 121 | 1228 | 2492 | `{"200": 320, "429": 277}` |
| `POST /uploads/{id}/finalize` | 150 | 163 | 2262 | 3812 | `{"200": 80, "429": 70}` |
| `POST /jobs/{id}/submit` | 246 | 117 | 1599 | 3465 | `{"200": 106, "429": 140}` |
| `GET /jobs/{id}` | 85 | 91 | 1314 | 1904 | `{"200": 50, "429": 35}` |
| `GET /jobs/{id}/events (time to first event)` | 83 | 75 | 687 | 1044 | `{"200": 40, "429": 43}` |
| `GET /jobs/{id}/result` | 154 | 86 | 933 | 1739 | `{"200": 80, "429": 74}` |

Duplicate retries in run `fbfa96c6`: 26 jobs had their create and submit sent twice at once; 52 of those responses were flagged `idempotent_replay`, and no job was queued twice (extra rows, repeated completions and re-attempts all zero; double-queued = 0).

Finding: the first runs showed some concurrent duplicate submits both answering `idempotent_replay: false`. The database state was right (one `draft -> queued` transition), only the response flag was wrong; the submit route now derives the flag from `queue_draft`'s result.
