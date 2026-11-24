# Gates and roadblocks

Log decision-gate failures (PLAN §21) and genuine roadblocks (missing credentials, gated weights,
broken downloads, two failed fix attempts) here: exact error, what was tried, what is needed.


## 2026-11-24: Human actions needed (logged by tech-lead audit, not overnight tasks)

- **Dev environment:** countonce has no venv. `ruff`/`pytest` currently run from the read-only
  `~/vggtProject/.venv`, and mypy has never been run. Needed: create `countonce/.venv` (Python 3.11) and
  `pip install -e .[dev,recon]`, then run `mypy` once.
- **Gates G0 / G1 (PLAN §21):** need the niche interviews and professor meeting (PLAN §20) before they
  can be decided. `docs/RELATED_WORK.md` has `TODO: verify` items that need a full-paper read.
- **Gate G2 (PLAN §21):** needs one foliage capture and one easy capture, then a VGGT run on the GPU.
- **Own captures + hand counts (PLAN §9.2):** 10–20 scenes, plus identity labels on 3–5 of them.
- **Fuji-SfM download and license (PLAN §9.1):** ~6.9 GB from Zenodo 3712808. The license is conflicting (see
  `docs/DATASETS.md`). Read `LICENSE.txt` and confirm the file layout so the loader's format TODOs can be closed.
- **Detector weights:** CountGD and GroundingDINO checkpoints need to be downloaded on a networked session
  before the real detector adapters can be checked end to end.

## Detector weights needed (2026-11-24, detector interface task)
- CountGD: clone niki-amini-naieni/CountGD, download its checkpoint (`checkpoint_fsc147_best.pth`) and the GroundingDINO Swin-B base weights it builds on; then implement `CountGDDetector._load`/`_infer` (currently stubs) and verify the output-format `TODO` in `countonce/detect/countgd.py`.
- GroundingDINO: `pip install groundingdino` (IDEA-Research), config `GroundingDINO_SwinT_OGC.py` and `groundingdino_swint_ogc.pth`; run `GroundingDinoDetector` once on a real frame to confirm `predict` score/box conventions.

## 2026-11-24: Second tech-lead audit (human actions, not queued as overnight tasks)
- Still open from the first audit: the countonce venv plus a first mypy run, G0/G1 interviews and the professor meeting, the G2 captures, own captures and hand counts, the Fuji-SfM download and license check, and the detector weights.
- **Baselines 5-6 (C-DOG, CountVid) and 7 (DG-Net):** these need their repos and weights cloned and downloaded on a networked session, plus the RGBD-VideoCount/VideoCount data. They are not queued until that is done. Once it is, log failures to run in DEVIATIONS (PLAN §9.3).
- **Labeling tool for identity labels (PLAN Phase 2):** this is useful only once real captures exist. It will be queued after the first captures land.
- **CI:** the workflow file will be added locally. Pushing it and enabling Actions on GitHub is a human step.

## 2026-11-24: Third tech-lead audit (human actions, not queued as overnight tasks)
- All 22 earlier overnight tasks are checked. `pytest` passes (157 tests) and `ruff check .` is clean, both
  still run from `~/vggtProject/.venv`. mypy is still not installed anywhere, so the countonce venv item above stays open.
- **Frontier VLM baseline (PLAN §9.3 item 8):** needs API keys (Anthropic, Google, OpenAI), a spending cap
  and your approval of the prompt and the scene subset before any paid call. The harness can be built with a fake
  client later, but no real run happens without you.
- **Kubric / BlenderProc rendering (PLAN §10.2, Phase 3):** needs Docker in WSL2 and a GPU-render test on the
  RTX 5070. This has to be done interactively. Until it is, the matcher trains on numpy geometry episodes only.
- **Golden GPU regression runner (PLAN §16B.2):** setting up the self-hosted runner (a separate low-privilege
  WSL distro and a GitHub registration token) is a human step.
- **Web product (PLAN §14, §17):** mockups need your approval before any frontend code. The API/DB work waits
  until the Core benchmark has real numbers.
- **Real-data numbers:** every queued task is tested only on synthetic or tmp fixtures. Results tables and figures stay
  `TODO: fill from results` until the Fuji-SfM download, the detector weights and your own captures exist.

## 2026-11-24: Fourth tech-lead audit (human actions, not queued as overnight tasks)
- Audit state: all 26 earlier overnight tasks are checked. `pytest` passes 229 tests and `ruff check .` is clean, both still run from
  `~/vggtProject/.venv`. mypy is still not installed in any environment, so the countonce venv item stays open.
- **Track J dependencies (PLAN §16B.1):** `pyarrow`, `duckdb`, `webdataset` and `dvc` are not installed anywhere, so the
  Parquet feature store, DuckDB catalog, WebDataset shards and DVC versioning are not queued yet (their tests would only skip).
  Needed: add them to the countonce venv, `dvc init`, and choose a remote (local disk or R2 credentials).
- **v3 checkpoint:** no matcher has been trained. A real training run on the RTX 5070 (a 1,000+ episode synth set) is needed before
  `co_v3_vggt` produces benchmark rows. Run it interactively so the GPU, time budget and results can be watched.
- **VLM baseline:** the harness is queued with paid calls disabled by default. Real runs still need API keys, a spending cap, and
  your approval of the prompts and the scene subset (see the third audit).
- **Kubric/BlenderProc rendering:** still human-gated. The renderer-agnostic recipe and validation gates are queued. The renderers
  are not.

## 2026-11-24: Fifth tech-lead audit (human actions, not queued as overnight tasks)
- Audit state: all 35 earlier overnight tasks are checked. `pytest` passes 345 tests and `ruff check .` is clean, both still run from
  `~/vggtProject/.venv`. mypy is still not installed anywhere (`~/vggtProject/.venv/bin/mypy` is missing), so the pre-commit mypy hook
  has never run. The countonce venv item stays open.
- **Privacy region detector weights:** the installed `cv2` (5.1.0) has no bundled Haar cascades. For face/plate blurring to run for real,
  download a face model (e.g. OpenCV YuNet ONNX or `haarcascade_frontalface_default.xml`) and a plate cascade, and set their paths
  in config. The queued task builds the blur/mask plumbing with a fake detector only.
- **Unimplemented backbones/detector in `run()`:** `countgdpp`, `vggt_omega` and `vggt_t3` still raise `NotImplementedError` in
  `countonce/pipeline.py`. They need gated or unreleased weights (CountGD++, VGGT-Ω non-commercial gated weights, VGG-T³ code/weights), so
  they are not queued.
- **Track F GPU measurements:** the queued optimization switches and `docs/PROFILE.md` generator are CPU-tested only. Filling the
  profile table needs an interactive RTX 5070 run with VGGT and detector weights present.
- **Web API/DB/queue (PLAN §14.2-14.4):** still deferred until the Core benchmark has real numbers, and the frontend still waits for mockup approval.

## 2026-11-24: Sixth tech-lead audit (human actions, not queued as overnight tasks)
- Audit state: all 47 earlier overnight tasks are checked. `pytest` passes 402 tests (1 skipped) and `ruff check .` is clean, both
  still run from `~/vggtProject/.venv`. mypy is still not installed, so the countonce venv item stays open.
- **Web product deps (PLAN §14.2-14.4, §14.8 idempotency/resumable uploads):** `fastapi`, `httpx`, `sqlalchemy` and a Postgres instance
  are not available, and the frontend still waits for mockup approval. The API, idempotency keys and load test stay unqueued until you
  approve mockups and add those deps. The correction tools are queued as a backend only.
- **MCP server:** the `mcp` package is not installed. The queued task builds the tool function and a lazily imported server. To run it
  for real, install the `mcp` extra and register the server in a Claude client yourself.
- **Golden gate runner:** the queued task adds the workflow and comparison script only. Registering the laptop as a self-hosted runner
  (a separate low-privilege user in its own WSL2 distro, restricted to this repo) is yours to do, and so is choosing the 5 real golden
  scenes after captures exist.
- **Track J (Parquet/DuckDB/WebDataset/DVC):** `pyarrow` and `duckdb` are still missing, so it stays unqueued (see the fourth audit).

## 2026-11-24: Seventh tech-lead audit (human actions, not queued as overnight tasks)
- Audit state: all 54 earlier overnight tasks are checked. `pytest` passes 477 tests (2 skipped) and `ruff check .` is clean in
  `~/vggtProject/.venv`. A countonce `.venv` now exists (uv, **Python 3.12.3**, dev extra only). In it, 5 test modules fail to collect
  (matplotlib/torch missing) and `mypy countonce` reports 65 errors. Both are queued as fixes. Human items: decide whether the venv should be
  3.11 to match CI (`setup-python 3.11`), then `uv pip install -e .[dev,recon,mcp]` (torch for `sm_120` needs the CUDA 12.8+ wheel index).
- **SAM/SAM2 weights** for the queued mask producer: download a checkpoint (e.g. `sam2_hiera_small.pt`) on a networked session and set its path
  in config. The queued task builds and tests only the box-fill/fake producer.
- **Human baseline study (PLAN §9.3 item 10):** the queued task builds the count sheets and the scoring harness only. Recruiting 3–5
  participants, running the timed sessions, and collecting the CSV need you, and they need real captures with hand counts first.
- **Model registry promotion on real data:** the promotion gate will be tested on synthetic/fake checkpoints. A real promotion needs a trained v3
  checkpoint (see the fourth audit) and the frozen real test split.
- **Still unqueued (dependencies or approvals missing):** the web API/DB/queue/worker and load test (fastapi/sqlalchemy/Postgres plus mockup
  approval); Track J Parquet/DuckDB/WebDataset/DVC (pyarrow/duckdb); OpenTelemetry (needs the API and an exporter account); Triton kernel and
  FP8/NVFP4 (interactive GPU); VGGT-Ω/VGG-T³/CountGD++ backends (gated weights); OneScene-Bench, RGBD-VideoCount and VideoCount loaders plus the
  C-DOG/CountVid/DG-Net baselines (downloads); HF Space/model/dataset releases (needs results and your accounts).

## 2026-11-24: SAM weights for the mask producer (human action)
- `Config.masks: sam` needs the `sam2` package and a SAM 2 checkpoint, e.g. `sam2_hiera_small.pt` (config `sam2_hiera_s.yaml`), downloaded
  by you on a networked session and set as `mask_checkpoint`. Nothing was downloaded; `sam` mode raises `MaskModelUnavailable` when missing.
  The queued task built and tested only the `box_fill` producer and a fake. The bench ablation row uses `box_fill` for that reason.

## 2026-11-24: Eighth tech-lead audit (human actions, not queued as overnight tasks)
- Audit state: all 61 earlier overnight tasks are checked. In the countonce `.venv` (Python 3.12, dev extra only), `pytest` gives 480 passed and
  22 skipped, `ruff check .` is clean and `mypy countonce` reports no issues (70 files). The 22 skips are missing torch, matplotlib, cv2, mcp and
  pycolmap. The 69 skipped-module tests pass in `~/vggtProject/.venv`. CI has the same blind spot, so a `full-cpu` CI job is queued.
- **License choice (PLAN §16, §24):** the repo has no LICENSE. Pick the code license yourself (e.g. MIT or Apache-2.0 for the code).
  Several dependencies and weights are non-commercial (VGGT, VGGT-Ω, possibly Fuji-SfM; see `docs/DATASETS.md`, `docs/MODEL_SURVEY.md`),
  so the README needs a note on that whatever you choose. The queued hygiene task deliberately skips the LICENSE file.
- **Local job queue vs. "API/DB deferred":** the queued queue/worker task uses stdlib sqlite only and has no web API, Postgres or frontend, so it
  needs neither new deps nor mockup approval. FastAPI, Postgres, auth, SSE and the frontend stay unqueued until mockups are approved and
  `fastapi`/`sqlalchemy`/a Postgres instance are added.
- **Identity labelling itself:** the queued labeling tool only builds the tool. Labelling 3–5 real scenes needs your captures first.
- **Still unqueued (unchanged from the seventh audit):** Track J Parquet/DuckDB/WebDataset/DVC and the OLTP→OLAP export (pyarrow/duckdb missing);
  OpenTelemetry; Triton kernel and FP8/NVFP4 (interactive GPU); VGGT-Ω/VGG-T³/CountGD++ backends (gated weights); the real CountGD/GroundingDINO
  forward passes (weights); dataset loaders and the C-DOG/CountVid/DG-Net baselines (downloads); HF releases; cloud GPU worker (account and billing).
- **Remaining autonomous work after this batch is thin:** almost every remaining PLAN item needs data, weights, GPU time, new deps or your approval.
  The best next unblockers are installing `pyarrow`+`duckdb` (unlocks Track J), approving mockups and adding `fastapi`/`sqlalchemy` (unlocks the
  web API), and downloading Fuji-SfM plus the detector weights (unlocks real Core numbers).

## 2026-11-24: Ninth tech-lead audit (human actions, not queued as overnight tasks)
- Audit state: all 69 earlier overnight tasks are checked. In the countonce `.venv` (now with CPU torch), `pytest` gives 573 passed and 3 skipped
  (mcp, pycolmap x2), and `ruff check .` is clean. **`mypy countonce` regressed to 3 errors** (privacy.py, appearance.py, vggt.py). The fix is queued first.
- **Uncommitted `docs/PLAN.md` edit:** the working tree has a formatter-only diff in the §7 code blocks (comment spacing, blank lines). The audit
  did not commit it. Keep it or `git checkout docs/PLAN.md`, your choice.
- **Cost rates:** the queued cost-per-job task leaves `cost_per_hour` null. Set the laptop electricity rate and a cloud GPU price yourself.
- **Pre-flight browser port and web API:** the queued pre-flight task is the Python reference only. The browser version, FastAPI endpoints, SSE,
  Google sign-in and Postgres still wait for mockup approval plus `fastapi`/`sqlalchemy`/Postgres (unchanged from the eighth audit).
- **OpenTelemetry export:** the queued tracing task writes local JSONL traces. Sending them to Grafana Cloud/Honeycomb needs an account and
  an OTLP endpoint/token from you.
- **Still unqueued (unchanged):** Track J (pyarrow/duckdb), Triton kernel and FP8/NVFP4 (interactive GPU), VGGT-Ω/VGG-T³/CountGD++ (gated weights),
  real CountGD/GroundingDINO/SAM forward passes (weights), the Fuji-SfM download and license, the C-DOG/CountVid/DG-Net baselines and video datasets,
  Kubric rendering (Docker), a v3 training run on the GPU, the cloud worker and cold-start measurements (account and billing), HF releases, the LICENSE choice,
  captures, interviews and the professor meeting.
- **Autonomous work after this batch is nearly exhausted.** After these 6 tasks, the remaining PLAN items all need data, weights, GPU time,
  new dependencies or your approval. Biggest unblockers: Fuji-SfM plus the detector weights (real Core numbers), `pyarrow`+`duckdb` (Track J), and mockup approval (web product).

## 2026-11-24: Tenth tech-lead audit (human actions, not queued as overnight tasks)
- Audit state: all 75 earlier overnight tasks are checked. In the countonce `.venv`, the full `pytest` run gives 609 passed and 3 skipped, `ruff check .`
  is clean and `mypy countonce` reports no issues (79 files). **`tests/test_job_retention.py::test_cli` is flaky**: it failed once under `-x`, then passed
  on reruns, because share tokens can start with `-` and break argparse. The fix is queued first. The system PATH has no `ruff` or `rtk pytest`. Use `.venv/bin/…`.
- **Uncommitted working-tree diffs:** `synth/dag.py` and `examples/tiny_scene/generate.py` hold `ruff format` fixes, and the queued first task commits them.
  The formatter-only `docs/PLAN.md` diff is still yours to keep or revert (unchanged from the ninth audit). Untracked `logs/`, `scripts/*.sh` and `uv.lock`
  are the orchestrator's own files and were left alone. Decide whether `uv.lock` should be committed.
- **Pilot and interviews:** the queued pilot-report task builds the generator and an empty `docs/INTERVIEWS.md` template. Running the pilot, recruiting
  3–10 users and doing the interviews are yours.
- **Load-test numbers:** the queued harness measures the sqlite queue on your laptop only. A real API load test (k6/Locust) waits for the FastAPI layer.
- **Still unqueued (unchanged):** Track J (install `pyarrow`+`duckdb`, both still missing from `.venv`), the web API/frontend/auth/SSE/Postgres and R2 multipart (mockup approval
  plus `fastapi`/`sqlalchemy`/Postgres), Triton kernel and FP8/NVFP4 (interactive GPU), VGGT-Ω/VGG-T³/CountGD++ and the real CountGD/GroundingDINO/SAM forward
  passes (weights), the Fuji-SfM download and license, the C-DOG/CountVid/DG-Net baselines plus RGBD-VideoCount/VideoCount, Kubric (Docker), the v3 GPU training run and
  sim-to-real ablations (data and GPU), the cloud worker and cold start (account), the OTLP endpoint, HF releases, the LICENSE choice, the demo video and project page, captures,
  interviews and the professor meeting.
- **After this batch, the autonomous backlog really is empty.** Every remaining PLAN item then needs data, weights, GPU time, new dependencies,
  accounts or your approval. Biggest unblockers, in order: Fuji-SfM plus the detector weights (real Core numbers), mockup approval plus `fastapi`/`sqlalchemy`
  (web product), and `pyarrow`+`duckdb` (Track J).
