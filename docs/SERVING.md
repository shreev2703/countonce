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
