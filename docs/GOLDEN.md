# Golden benchmark gate

Implements PLAN §16B.2. A frozen 5-scene set is run on every push to `main`; the result is compared
with the last accepted result and a regression fails the job.

## Commands

```
python -m bench.golden run [--cpu-synthetic] [--data-root data] [--out-dir results/golden]
python -m bench.golden compare --baseline X.json [--current Y.json] [--cpu-synthetic] [--tol latency_s=off]
python -m bench.golden freeze        # pin sha256 of the real scenes in configs/golden.yaml
```

`run` writes `results/golden/<git-sha>.json` (`-cpu` suffix in CPU mode): MAPE, association F1, peak
VRAM and latency per scene and overall. `compare` prints a Markdown delta table and exits 1 when
a metric is worse than its tolerance (defaults in `configs/golden.yaml`: MAPE +0.02 abs, F1 -0.02
abs, VRAM +10%, latency +25%; `--tol metric=off` disables one). A scene missing from the current run
is a regression.

## Two modes

* **CPU synthetic (`--cpu-synthetic`)**: five seeded plane scenes, fake detector and plane
  reconstructor. Deterministic counts and F1, pinned by the sha256 of each scene spec. Runs on every
  PR in hosted `ci.yml` against `configs/golden_cpu_baseline.json` with latency ignored (hosted
  runners are noisy). Safe for untrusted code.
* **GPU (real scenes)**: five scenes under `--data-root`, pinned with `bench.splits.scene_hash`.
  `TODO: choose the 5 scenes` in `configs/golden.yaml`, run `freeze`, then record the first
  `main` result as `configs/golden_gpu_baseline.json` (`TODO: fill from results`). Until both exist
  the GPU job fails by design; no numbers are invented.

Updating a baseline is a deliberate commit that explains the accepted change.

## Runner hygiene (self-hosted GPU runner)

`golden-gpu.yml` triggers only on `push` to `main` and `workflow_dispatch`, never on
`pull_request` or `pull_request_target`, so fork code cannot reach the laptop. To check a PR branch,
read the diff first, then dispatch the workflow on it by hand. The workflow has
`permissions: contents: read`, a `timeout-minutes`, no secrets, and `persist-credentials: false`.
`tests/test_golden_gate.py` asserts these properties.

Set the runner up (not done by this repo; no runner is registered):

* Run it as a separate low-privilege user in its own WSL2 distro, not your daily distro.
* No personal files mounted: disable Windows drive automount in that distro
  (`/etc/wsl.conf`: `[automount] enabled=false`, `[interop] enabled=false`).
* No cloud credentials, SSH keys, tokens or `~/.config/gh` in that user's home.
* Restrict the runner to this one repository; do not register it at organization level.
* Keep the job timeout; use ephemeral work directories and clean them between runs.
* Model weights go in a read-only directory the job user cannot write outside of.
