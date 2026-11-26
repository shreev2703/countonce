# Gates and roadblocks

Log decision-gate failures (PLAN §21) and genuine roadblocks (missing credentials, gated weights,
broken downloads, two failed fix attempts) here: exact error, what was tried, what is needed.

Every `##` heading ends in its status: **open** (with an owner) or **resolved**. Text under a resolved
heading is a historical record and may describe things as they stood then.


## 2026-11-24: Human actions needed (logged by tech-lead audit, not overnight tasks) — resolved (audit snapshot; open items carried to the open sections at the end)

- **Dev environment:** countonce has no venv. `ruff`/`pytest` currently run from the read-only
  the sibling VGGT benchmark repo venv, and mypy has never been run. Needed: create `countonce/.venv` (Python 3.11) and
  `pip install -e .[dev,recon]`, then run `mypy` once.
- **Gates G0 / G1 (PLAN §21):** need the niche interviews and professor meeting (PLAN §20) before they
  can be decided. `docs/RELATED_WORK.md` has `TODO: verify` items that need a full-paper read.
- **Gate G2 (PLAN §21):** needs one foliage capture and one easy capture, then a VGGT run on the GPU.
- **Own captures + hand counts (PLAN §9.2):** 10–20 scenes, plus identity labels on 3–5 of them.
- **Fuji-SfM download and license (PLAN §9.1):** ~6.9 GB from Zenodo 3712808. The license is conflicting (see
  `docs/DATASETS.md`). Read `LICENSE.txt` and confirm the file layout so the loader's format TODOs can be closed.
- **Detector weights:** CountGD and GroundingDINO checkpoints need to be downloaded on a networked session
  before the real detector adapters can be checked end to end.

## Detector weights needed (2026-11-24, detector interface task) — open (owner: Shreev; CountGD only, GroundingDINO HF resolved)
- CountGD: clone niki-amini-naieni/CountGD, download its checkpoint (`checkpoint_fsc147_best.pth`) and the GroundingDINO Swin-B base weights it builds on; then implement `CountGDDetector._load`/`_infer` (currently stubs) and verify the output-format `TODO` in `countonce/detect/countgd.py`.
- GroundingDINO (**resolved 2026-11-24 for the HF backend**: `grounding-dino-tiny` runs for real, see `docs/DEVIATIONS.md`; only the optional original-package path below is still unverified): `pip install groundingdino` (IDEA-Research), config `GroundingDINO_SwinT_OGC.py` and `groundingdino_swint_ogc.pth`; run `GroundingDinoDetector` once on a real frame to confirm `predict` score/box conventions.

## 2026-11-24: Second tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Still open from the first audit: the countonce venv plus a first mypy run, G0/G1 interviews and the professor meeting, the G2 captures, own captures and hand counts, the Fuji-SfM download and license check, and the detector weights.
- **Baselines 5-6 (C-DOG, CountVid) and 7 (DG-Net):** these need their repos and weights cloned and downloaded on a networked session, plus the RGBD-VideoCount/VideoCount data. They are not queued until that is done. Once it is, log failures to run in DEVIATIONS (PLAN §9.3).
- **Labeling tool for identity labels (PLAN Phase 2):** this is useful only once real captures exist. It will be queued after the first captures land.
- **CI:** the workflow file will be added locally. Pushing it and enabling Actions on GitHub is a human step.

## 2026-11-24: Third tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- All 22 earlier overnight tasks are checked. `pytest` passes (157 tests) and `ruff check .` is clean, both
  still run from the sibling VGGT benchmark repo venv. mypy is still not installed anywhere, so the countonce venv item above stays open.
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

## 2026-11-24: Fourth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 26 earlier overnight tasks are checked. `pytest` passes 229 tests and `ruff check .` is clean, both still run from
  the sibling VGGT benchmark repo venv. mypy is still not installed in any environment, so the countonce venv item stays open.
- **Track J dependencies (PLAN §16B.1):** `pyarrow`, `duckdb`, `webdataset` and `dvc` are not installed anywhere, so the
  Parquet feature store, DuckDB catalog, WebDataset shards and DVC versioning are not queued yet (their tests would only skip).
  Needed: add them to the countonce venv, `dvc init`, and choose a remote (local disk or R2 credentials).
- **v3 checkpoint:** no matcher has been trained. A real training run on the RTX 5070 (a 1,000+ episode synth set) is needed before
  `co_v3_vggt` produces benchmark rows. Run it interactively so the GPU, time budget and results can be watched.
- **VLM baseline:** the harness is queued with paid calls disabled by default. Real runs still need API keys, a spending cap, and
  your approval of the prompts and the scene subset (see the third audit).
- **Kubric/BlenderProc rendering:** still human-gated. The renderer-agnostic recipe and validation gates are queued. The renderers
  are not.

## 2026-11-24: Fifth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 35 earlier overnight tasks are checked. `pytest` passes 345 tests and `ruff check .` is clean, both still run from
  the sibling VGGT benchmark repo venv. mypy is still not installed anywhere (its `bin/mypy` is missing), so the pre-commit mypy hook
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

## 2026-11-24: Sixth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 47 earlier overnight tasks are checked. `pytest` passes 402 tests (1 skipped) and `ruff check .` is clean, both
  still run from the sibling VGGT benchmark repo venv. mypy is still not installed, so the countonce venv item stays open.
- **Web product deps (PLAN §14.2-14.4, §14.8 idempotency/resumable uploads):** `fastapi`, `httpx`, `sqlalchemy` and a Postgres instance
  are not available, and the frontend still waits for mockup approval. The API, idempotency keys and load test stay unqueued until you
  approve mockups and add those deps. The correction tools are queued as a backend only.
- **MCP server:** the `mcp` package is not installed. The queued task builds the tool function and a lazily imported server. To run it
  for real, install the `mcp` extra and register the server in a Claude client yourself.
- **Golden gate runner:** the queued task adds the workflow and comparison script only. Registering the laptop as a self-hosted runner
  (a separate low-privilege user in its own WSL2 distro, restricted to this repo) is yours to do, and so is choosing the 5 real golden
  scenes after captures exist.
- **Track J (Parquet/DuckDB/WebDataset/DVC):** `pyarrow` and `duckdb` are still missing, so it stays unqueued (see the fourth audit).

## 2026-11-24: Seventh tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 54 earlier overnight tasks are checked. `pytest` passes 477 tests (2 skipped) and `ruff check .` is clean in
  the sibling VGGT benchmark repo venv. A countonce `.venv` now exists (uv, **Python 3.12.3**, dev extra only). In it, 5 test modules fail to collect
  (matplotlib/torch missing) and `mypy countonce` reports 65 errors. Both are queued as fixes. Human items: decide whether the venv should be
  3.11 to match CI (`setup-python 3.11`), then `uv pip install -e .[dev,recon,mcp]` (torch for `sm_120` needs the CUDA 12.8+ wheel index).
- **SAM/SAM2 weights** (RESOLVED 2026-11-24 through Hugging Face: `facebook/sam2.1-hiera-small`, public, ungated, Apache-2.0, used by `Sam2HfMaskProducer`; only the `sam2`-package path still needs a manual checkpoint): download a checkpoint (e.g. `sam2_hiera_small.pt`) on a networked session and set its path
  in config. The queued task builds and tests only the box-fill/fake producer.
- **Human baseline study (PLAN §9.3 item 10):** the queued task builds the count sheets and the scoring harness only. Recruiting 3–5
  participants, running the timed sessions, and collecting the CSV need you, and they need real captures with hand counts first.
- **Model registry promotion on real data:** the promotion gate will be tested on synthetic/fake checkpoints. A real promotion needs a trained v3
  checkpoint (see the fourth audit) and the frozen real test split.
- **Still unqueued (dependencies or approvals missing):** the web API/DB/queue/worker and load test (fastapi/sqlalchemy/Postgres plus mockup
  approval); Track J Parquet/DuckDB/WebDataset/DVC (pyarrow/duckdb); OpenTelemetry (needs the API and an exporter account); Triton kernel and
  FP8/NVFP4 (interactive GPU); VGGT-Ω/VGG-T³/CountGD++ backends (gated weights); OneScene-Bench, RGBD-VideoCount and VideoCount loaders plus the
  C-DOG/CountVid/DG-Net baselines (downloads); HF Space/model/dataset releases (needs results and your accounts).

## 2026-11-24: SAM weights for the mask producer — resolved (Hugging Face SAM 2.1, see below)
- `Config.masks: sam` needs the `sam2` package and a SAM 2 checkpoint, e.g. `sam2_hiera_small.pt` (config `sam2_hiera_s.yaml`), downloaded
  by you on a networked session and set as `mask_checkpoint`. Nothing was downloaded; `sam` mode raises `MaskModelUnavailable` when missing.
  The queued task built and tested only the `box_fill` producer and a fake. The bench ablation row uses `box_fill` for that reason.

## 2026-11-24: Eighth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 61 earlier overnight tasks are checked. In the countonce `.venv` (Python 3.12, dev extra only), `pytest` gives 480 passed and
  22 skipped, `ruff check .` is clean and `mypy countonce` reports no issues (70 files). The 22 skips are missing torch, matplotlib, cv2, mcp and
  pycolmap. The 69 skipped-module tests pass in the sibling VGGT benchmark repo venv. CI has the same blind spot, so a `full-cpu` CI job is queued.
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

## 2026-11-24: Ninth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
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

## 2026-11-24: Tenth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
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

## 2026-11-24: Eleventh tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 82 earlier overnight tasks are checked. In the countonce `.venv`, `pytest` gives 653 passed and 3 skipped (mcp, pycolmap x2),
  `ruff check .` and `ruff format --check .` are clean, and `mypy countonce` reports no issues (84 files). The audit found no new bugs or stubs beyond the weight-gated
  `NotImplementedError` paths already logged (`countgdpp`, `vggt_omega`, `vggt_t3`, the CountGD forward pass).
- **Policy change on new PyPI deps:** earlier audits treated `pyarrow`/`duckdb` and `fastapi`/`httpx` as human-gated. They are public, non-credentialed
  installs, so this batch queues Track J, the OLAP export and the HTTP API, and each task installs its own extra into `.venv`. If an install fails (no network),
  the task logs it here and exits. The **frontend** still waits for your mockup approval. Postgres, R2, Google sign-in and deploys still need your accounts.
- **Still yours (unchanged):** the formatter-only `docs/PLAN.md` diff (keep or revert), whether to commit `uv.lock`, the LICENSE choice, Docker in WSL2 (to build the
  queued worker image and run Kubric), HF account and uploads for the queued release scaffolding, GPU runs (v3 training, profiling, cold start, Triton, FP8/NVFP4),
  gated or large weights and datasets (Fuji-SfM, CountGD/GroundingDINO/SAM, VGGT-Ω/VGG-T³/CountGD++, video datasets, C-DOG/CountVid/DG-Net), the OTLP endpoint,
  cost rates, captures, hand counts, identity labels, interviews, the professor meeting, the pilot, the human baseline study, the demo video and the paper venue.
- **After this batch the autonomous backlog is empty.** Everything left in PLAN.md needs data, weights, GPU or Docker time, accounts or your approval.

## 2026-11-24: Twelfth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 89 earlier overnight tasks are checked. In the countonce `.venv`, `pytest` gives 708 passed and 4 skipped (safetensors, mcp, pycolmap x2),
  and `mypy countonce` reports no issues (91 files). **`ruff check .` fails (9 errors) and `ruff format --check .` flags 1 file, all in the untracked
  orchestrator script `scripts/hud.py`.** That breaks every task's lint gate, so the fix (ruff `extend-exclude`, not edits to your file) is queued first.
  The system PATH still has no `ruff` or `pytest`, so use `.venv/bin/…`.
- **Still yours (unchanged):** the formatter-only `docs/PLAN.md` diff (keep or `git checkout docs/PLAN.md`), whether to commit `uv.lock`, and whether to
  commit your orchestrator scripts (`scripts/hud.py`, `dashboard.sh`, `morning_review.sh`, `overnight_orchestrator.sh`). The queued task only excludes them from ruff.
- **New queued tasks that end at a human gate:** the Postgres store needs a real database URL (`COUNTONCE_PG_URL`, local Docker or Neon/Supabase) before its
  contract tests run for real. The S3/R2 storage backend needs R2 credentials and a bucket. The DVC scaffold needs you to choose and configure a remote.
  The FSC-147, WGISD, MinneApple and AppleGrowthVision download scripts are written but not run, so download them and check their licenses yourself.
  The detector fine-tuning harness needs CountGD weights plus those datasets and GPU time.
- **Still unqueued (unchanged):** frontend, 3D viewer and Google sign-in (mockup approval), deploys (Vercel/Render/Neon accounts), Gaussian-splat view toggle
  (gsplat GPU pipeline and mockup), Triton kernel and FP8/NVFP4 (interactive GPU), VGGT-Ω/VGG-T³/CountGD++ and real detector/SAM forward passes (weights), Kubric and
  the worker image build (Docker), the v3 GPU training run, the cloud worker, the OTLP endpoint, HF uploads, the LICENSE choice, captures, hand counts, identity labels,
  interviews, the professor meeting, the pilot, the human baseline study, the frontier-VLM baseline run (API keys and spend), the demo video and the paper venue.
- **After this batch the autonomous backlog is truly exhausted.** The next audit should only queue bugs or regressions.

## Object storage against real R2 — open (owner: Shreev)
- `S3BlobStore` and the cache/upload/worker routing pass the contract suite on `moto` only. To verify on Cloudflare R2: create a bucket and an API token, set `COUNTONCE_STORAGE=s3`, `COUNTONCE_S3_BUCKET`, `COUNTONCE_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com`, `COUNTONCE_S3_ACCESS_KEY_ID`, `COUNTONCE_S3_SECRET_ACCESS_KEY` (`COUNTONCE_S3_REGION=auto`), install `.[storage]`, then run a worker with `--storage s3` and fetch a presigned URL by hand. Also set a bucket CORS rule before browser uploads use `presign_put`.

## DVC remote (R2) — open (owner: Shreev)
`dvc.yaml`/`params.yaml` are committed but no DVC remote is configured. Needs a human: create the
R2 bucket, then `dvc remote add -d r2 s3://<bucket>/dvc`, set the endpoint URL and credentials
(local, not committed), `dvc push`. Until then `dvc.lock`/data stay local.

## Real CountGD fine-tune (Track D) — open (owner: Shreev)
- **Gate:** CountGD weights (and the real MinneApple/WGISD/Fuji-SfM downloads). The harness (`countonce finetune-detector`, `scripts/make_detector_adaptation_report.py`) is tested with a tiny fake torch detector only. Needs a human to fetch the weights and supply a module factory; layer names in `configs/finetune.yaml` are `TODO: verify`.

## 2026-11-24: Thirteenth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: all 96 earlier overnight tasks are checked. In the countonce `.venv`, `pytest` gives 870 passed and 22 skipped (19 Postgres contract cases with no
  `COUNTONCE_PG_URL`, plus mcp and pycolmap x2). `ruff check .`, `ruff format --check .` and `mypy` (109 files) are clean, and `git status` is clean.
- **Bugs found and queued:** (1) the CI `full-cpu` job should fail, because `dvc` isn't installed there, so the DVC test skips outside the
  `check_skips.py` allow-list. (2) `synth/shards.pack_shards` leaks an open tar handle and a partial shard on error, and 18 tests leak file handles.
  (3) `bench/vlm/parse` reads "Confidence: 80%" as a count of 80 and "No apples are visible." as a parse failure instead of 0.
- **The GPU is not human-gated:** this machine has an RTX 5070 Laptop GPU (8 GB) and the public `facebook/VGGT-1B` weights cached under `~/.cache/huggingface`.
  So this batch queues the first real-VGGT run and baseline profile (color-blob detector, tiny scene, capped at 20 minutes, its own `.venv-gpu`, leaving
  the VGGT benchmark repo untouched). Earlier audits listed "GPU runs" as yours. If you'd rather watch GPU runs yourself, delete that task before the orchestrator reaches it.
- **Mockups:** the queued task only *produces* static mockups in `docs/mockups/`. **Your approval** is still required before any `frontend/` code, and the
  frontend, 3D viewer and Google sign-in stay unqueued until you approve.
- **Still yours (unchanged):** a real Postgres (Neon/Supabase) and R2 bucket and credentials, the DVC remote, deploys (Vercel/Render), the OTLP endpoint, HF uploads,
  the LICENSE choice, Docker in WSL2 (Kubric, the worker image build), gated or large weights and datasets (CountGD/GroundingDINO/SAM, VGGT-Ω/VGG-T³/CountGD++,
  Fuji-SfM, the video datasets, the crop datasets and their license checks), the v3 GPU training run at full scale, Triton and FP8/NVFP4 experiments, frontier-VLM
  API spend, captures, hand counts, identity labels, interviews, the professor meeting, the pilot, the human baseline study, the demo video and the paper venue.
  Also still yours from earlier audits: whether to commit `uv.lock` and the orchestrator scripts.
- **After this batch:** the remaining PLAN.md work needs your approval (frontend), accounts (deploys, Postgres/R2 hosting), gated weights or data, or Docker.

## Mockup approval (PLAN §17) — resolved (superseded by PLAN §0.1 rule 5: review after the build, H2)

Static mockups are in `docs/mockups/` (upload, progress, results, history, states) with an approval checklist in `docs/mockups/README.md`.
**Shreev must approve (or request changes) before any `frontend/` code starts.** No screenshots yet: `playwright` is not installed,
so `scripts/render_mockups.py` skips; open the HTML files directly or install playwright and rerun.

## 2026-11-24: Fourteenth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state: at HEAD the code is healthy (`ruff check .`, `ruff format --check .` and `mypy countonce` (112 files) are clean), but **task 102 (API load test) was
  marked `[x]` without being committed**. Its untracked and modified files are still in the working tree, and with them `pytest` gives 938 passed, 26 skipped and
  **2 failed** (`test_coldstart.py::test_report_with_and_without_rows` and `test_loadtest_api.py::test_rate_limit_produces_429_and_still_reconciles`). Finishing
  and committing it is queued first. The audit left those files untouched. `runs/` (generated outputs) is untracked and not gitignored, and the same task fixes that.
- **Fuji-SfM is resolved:** it is downloaded and unpacked in `data/fuji_sfm/`, and `LICENSE.txt` says **CC BY-NC-SA 4.0** (research and education only). No images or derived GT
  may go into git, an HF dataset or a commercial product. The loader verification and the first real Core benchmark are queued.
- **Detector and mask weights partly resolved:** `IDEA-Research/grounding-dino-tiny` and `facebook/sam2.1-hiera-small` are public, ungated and Apache-2.0 on the HF Hub, and they
  load through `transformers`. The real forward passes are queued. **CountGD is still yours:** its checkpoint and the GroundingDINO Swin-B base aren't on an ungated HF repo
  this audit could confirm, so `CountGDDetector` stays a stub.
- **GPU tasks queued:** the Fuji-SfM benchmark (90 min cap) and a capped v3 training run (45 min cap). If you'd rather watch GPU runs, delete those tasks before the orchestrator reaches them.
- **Still yours (unchanged):** **mockup approval** (`docs/mockups/`) before any `frontend/` code, plus the frontend, 3D viewer and Google sign-in after it; own captures, hand counts and
  identity labels (the own-capture half of Core); a real Postgres and R2 bucket with credentials, the DVC remote, deploys, the OTLP endpoint, HF uploads, the LICENSE choice (note that
  Fuji-SfM is NC-SA); Docker (Kubric, the worker image); CountGD/CountGD++/VGGT-Ω/VGG-T³ weights; the video datasets and C-DOG/CountVid/DG-Net baselines (CountVid is a Core baseline,
  so it is the biggest remaining Core gap); frontier-VLM API spend; interviews, the professor meeting, the pilot, the human baseline study, the demo video and the paper venue; and
  whether to commit `uv.lock` and the orchestrator scripts.

## 2026-11-24: SAM weights resolved through Hugging Face — resolved
- `Config.masks: sam` without `mask_checkpoint` now uses `Sam2HfMaskProducer` (`facebook/sam2.1-hiera-small`, public, ungated, Apache-2.0), downloaded from the HF Hub on first use. No manual download is needed; the earlier "SAM weights" human action above is closed. Setting `mask_checkpoint` still selects the `sam2`-package adapter.

## 2026-11-24: Fifteenth tech-lead audit (the two `[!]` tasks were not real gates) — resolved (non-gates; re-queued work done)
- Audit state at `af6dba0`: `pytest` 957 passed and 29 skipped, and `ruff check .`, `ruff format --check .` and `mypy countonce` (112 files) are all clean.
- **"First real Core benchmark on Fuji-SfM" (`[!]`):** the session hit the orchestrator's 40-turn cap (`error_max_turns`, log `logs/overnight_20261124_1502/181914_*.log`), so this was not a credential or weights gate. It had written `bench/fuji_core.py` and `scripts/run_fuji_core.py` (WIP commit `7e23718`, untested) and got one east-side overview run through (peak VRAM 5.5 GB, no OOM). It was then stuck registering VGGT windows to the GT frame. Its last diagnostic, which loaded the 542k-point GT cloud, was killed with **exit 137** (host RAM, 15 GB total). `.venv-gpu` has no `pandas`. The task is re-queued as a cleanup/tests task plus three bounded steps (train-split thresholds, resumable test-split runs, CPU-only report).
- **"First capped v3 matcher training run" (`[!]`):** also stopped at the 40-turn cap (log `184420_*.log`). The training is **Python-bound: 703 s for epoch 1** of 800 episodes at about 11% GPU utilisation (epoch-1 val_f1 0.368; this is a partial run, not a result). It is re-queued as a training speed-up task, then a fresh capped run.
- **Orchestrator note (human):** GPU tasks that wait on long runs with background monitors burn turns. Either raise the per-task turn cap for GPU tasks or keep them split as above. The new tasks tell agents to run GPU jobs in the foreground under `timeout`.
- **Still yours (unchanged from the fourteenth audit):** mockup approval before any `frontend/` code; own captures, hand counts and identity labels; CountGD/CountGD++/VGGT-Ω/VGG-T³ weights; CountVid/DG-Net/C-DOG and the video datasets; real Postgres/R2/DVC remote/OTLP/deploys; HF uploads; frontier-VLM spend; the LICENSE choice (Fuji-SfM is CC BY-NC-SA, VGGT-1B is non-commercial); interviews, the professor meeting, the pilot, the human baseline study and the demo video.

## 2026-11-24: Fuji Core step 2 and report re-run blocked — resolved (superseded by M8)
- Step 2 (test-split runs) and the report re-run were blocked because step 1 had not frozen `configs/fuji_core.yaml`,
  so `results/bench.csv` had no Fuji rows. No test-split run was started without frozen thresholds and no cell was filled.
- The report re-run was retried 23 times while blocked. The third to twenty-third retries are the commit range
  `3737ddc`..`2afa977` (history left as is).
- M8 superseded all of these: it froze the config, ran the test trees into `results/bench.csv` and filled the report.

## 2026-11-25: Sixteenth tech-lead audit — resolved (audit snapshot; open items carried to the open sections at the end)
- Audit state at `c3e5655`: all 119 earlier tasks are checked. `ruff check .`, `ruff format --check .` and `mypy countonce` (112 files) are clean. `pytest -m "not slow and not gpu"` gives 988 passed and 26 skipped (21 Postgres with no `COUNTONCE_PG_URL`, plus mcp and pycolmap x2 in `.venv`).
- **Bug found and queued:** the Fuji-SfM test-split numbers now in the README and `docs/RESULTS_FUJI.md` come from an evaluation protocol with visible defects. About half of the k=3/5 windows have GT = 0, miss rate is about 0.95 where it is defined, assoc F1 is set on only 54 of 320 rows, Tree06/west k=5 returns 0 for every method, and no row records `low`/`high`. **Until the queued protocol fix and re-run land, treat those numbers as provisional and do not quote them.** The queued order is: protocol fix (CPU), re-run, the sync and COLMAP baselines, the sweep axes, v3 on Fuji (G4 evidence), conformal and slicing on real rows, then docs.
- **Orchestrator note (human):** the "Fuji Core report re-run" task was retried 23 times while its dependency was unmet. Each retry made a no-op commit (`3737ddc`..`2afa977`) and added GATES/DEVIATIONS entries. Consider having the orchestrator skip a task after 2 consecutive gate logs, or mark it `[!]` so it is not retried. The last queued task cleans up the entries; rewriting history is your call (never done automatically).
- **Gate G4 (v3 vs. v2 on real data):** the v3-on-Fuji task only produces the evidence. **The decision is yours.** The same applies to G3 once the re-run report exists.
- **Still yours (unchanged from the fifteenth audit):** mockup approval (`docs/mockups/`) before any `frontend/` code, plus the frontend, 3D viewer and Google sign-in after it; own captures, hand counts and identity labels (Phase 1-2 own-scene half of Core, and the 5-real-scene "Done when" of Phase 1); CountGD/CountGD++/VGGT-Ω/VGG-T³ weights; CountVid/DG-Net/C-DOG and the video datasets (CountVid is still the largest Core baseline gap); a real Postgres (the 21 skipped contract tests need `COUNTONCE_PG_URL`), R2, DVC remote, OTLP endpoint and deploys; HF uploads; frontier-VLM API spend; Docker (Kubric, the worker image); the LICENSE choice (Fuji-SfM is CC BY-NC-SA, VGGT-1B is non-commercial); interviews, the professor meeting, the pilot, the human baseline study, the demo video and the paper venue.
- **After this batch:** the remaining autonomous PLAN.md work is the Fuji fine-tune ablation with at least 3 seeds (Phase 3), which waits on G4. Everything else needs the human items above.

## 2026-11-25: Gate G3 (geometry helps) on Fuji-SfM, M8 evidence — open (owner: Shreev, decision)
- **Recommendation: G3 not met (mixed).** On the 8 frozen test trees (320 rows, protocol `fuji_core_v2`), v1/v2 beat Sum in most slices at k=10 and 20 but not at k=3/5, and beat Max in most slices only at k=20 (v2 vs Max paired MAE diff -51.9, CI [-81.7, -20.3]); Max has the lowest MAE at k=10 (31.1 vs v2 40.1). Table and per-slice counts: `docs/RESULTS_FUJI_ANALYSIS.md`.
- Failure sources (k=10, no oracle geometry available on Fuji): the detector (about 121 false positives and 52 misses per window) is the largest error, then duplicates from association (34-37 per window even with perfect detections).
- Fallback per PLAN §21: publish the honest mixed result; the final decision and any redirect (detector first, then association) is the owner's. Does not decide G4.

## 2026-11-25: OneScene-Bench images not published (found during M10; does not block M10) — open (owner: Shreev)
- `github.com/swathisuhas/OneScene-Bench` holds parquet annotations whose image fields are relative paths with no bytes (checked first rows of three files), and the repo tree has no image files.
- Needed before the Strong-tier OneScene evaluation: get the images from the authors, or rebuild them from DL3DV-10K under its terms (yours: it needs accepting terms or contacting the authors). `bench/datasets/onescene.py` also needs rewriting for the parquet layout.

## 2026-11-25: Google OAuth client needed (does not block M16) — open (owner: Shreev)
- The sign-in code is in place but has never run against Google. Yours: create an OAuth web client, add `<origin>/api/auth/google/callback` as a redirect URI, set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `COUNTONCE_AUTH_SECRET`, `COUNTONCE_SESSION_SECRET` and `COUNTONCE_PUBLIC_URL`, and leave `COUNTONCE_DEV_AUTH` unset in public.

## M18: hosted deployment — open (owner: Shreev)
Create the two Vercel projects, Neon Postgres and the R2 bucket, then follow `docs/DEPLOY.md`. Until then "frontend talks to the deployed API" is verified only locally.

## M19: GPU end-to-end — open (owner: Shreev)
Run once on the laptop: start the API and `web/` dev server (see `web/README.md`), run `scripts/start_worker.sh`, upload photos at `/new` and confirm the result page. Not run overnight (needs a GPU session and a browser).

## M20: live demo URL — open (owner: Shreev)
No hosted frontend exists (see M18 above). After deploying, replace the "Live demo: not hosted yet" line in `README.md` with the URL.

## M20: five-view VGGT run failed through the CLI on WSL2 8 GB — resolved
The CLI's `expandable_segments:True` gave `CUDA driver error: device not ready`. `configure_allocator()`
now leaves the allocator alone on WSL (`countonce/oom.py`); 3 of 3 five-view runs succeed without an override.

## M21: one step left for a human (2026-11-25) — open (owner: Shreev)
`CLAUDE.md` is still tracked and still contains personal paths, so M21 is not done. The user-level
`guard_commit.py` PreToolUse hook rejects any agent commit whose staged paths include `CLAUDE.md`,
including its removal from the index. The hook was not bypassed. Everything else in M21 is committed.

To finish, from your own shell in the repo: `git rm --cached CLAUDE.md`, then
`git commit -m "chore(repo): untrack agent rules file (M21)"`, then tick M21 in the local task file.
The file stays on disk and is already in `.gitignore`. Then remove the `xfail` marker and the
`CLAUDE.md` exemption in `tests/test_repo_hygiene.py`.

Before merging into `main` (which still tracks the agent files): back up the local copies. Checking out
`main` and then merging this branch deletes them from the working tree. Git history also still holds
them and their personal paths; history was not rewritten.

## M22: §0.6 rows left unticked (2026-11-25) — open (owner: Shreev)
The consistency pass is committed, but its "done when" is not met: only the reliable-path, tiny-scene and
README rows could honestly be ticked. M22 is left unchecked. What each other no-photo row needs:
- **Gate G3 (yours):** confirm or reject the "not met" recommendation above.
- **Repo hygiene (yours):** the M21 `CLAUDE.md` step, then push and check hosted CI.
- **Lighthouse (code):** resolved 2026-11-25. `/demo/tiny` performance went from a median 0.54 to 0.93
  (3 local runs); see DEVIATIONS.
- **Failure sources:** separating geometry from association needs data with oracle geometry.
- **Related work:** run experiments E1-E5 and read the remaining works (`docs/RELATED_WORK.md`).

Also found (fixed later via the `worker` extra): `docker/worker.Dockerfile` installs `.[recon,data]` without `transformers` or the
`vggt` package, so that image cannot run the default pipeline (the image has never been built). The
README setup was verified with `uv pip`, not plain `pip`, and resolved torch 2.15.0+cu132.

## 2026-11-25: Seventeenth tech-lead audit — open (owner: Shreev)
- Audit state at `f6392e3`: `ruff check .` clean; `pytest -m "not slow and not gpu"` gives 1113 passed, 27 skipped
  (Postgres without `COUNTONCE_PG_URL`, `mcp`, `pycolmap`), 1 xfail (the M21 `CLAUDE.md` hygiene test).
- **M5 was not a real gate:** its run stopped at the orchestrator's 40-turn cap (`error_max_turns`), leaving
  WIP in `bd60d83`. It is re-queued as M5a (harness) and M5b (tuning + v3 re-run), sized to fit one session.
- **Queued (autonomous, in order):** M5a, M5b, worker Docker image deps, order-invariance on Fuji (real data),
  sync and v3 on Fuji held-out trees (G4 evidence only), `/demo/tiny` Lighthouse performance, GATES cleanup,
  §28.5 quarantine of unproven Moon-tier modules into `experimental/`.
- **`TASKS_OVERNIGHT.md` is gitignored (M21, §0.1 rule 6).** This audit updated it locally and did not
  force-add it; only this file is committed.
- **Still yours (human only):**
  - Gate G3: confirm or reject the "not met (mixed)" recommendation above. Gate G4 when the Fuji v3 rows exist.
  - M21: `git rm --cached CLAUDE.md` and commit from your own shell, then drop the xfail in
    `tests/test_repo_hygiene.py`; push and check hosted CI.
  - H1 capture day (20-30 scenes, ≥3 zero-target), then P1-P5 run as agent tasks.
  - H5 accounts (Vercel x2, Google OAuth client, Neon, R2) and the M18 deploy; M19 GPU end-to-end from a browser;
    put the live URL in the README.
  - H2 frontend review, H3 interviews, H4 professor meeting, H6 ownership study, the demo video.
  - Docker (worker image build, Kubric), gated weights (CountGD/CountGD++, VGGT-Ω, VGG-T³), CountVid/DG-Net and
    the video datasets (E1-E3), OneScene-Bench images (E5), frontier-VLM API spend.
- **After this batch:** remaining PLAN work needs the photos (P1-P6) or the human items above. Oracle-geometry
  failure-source runs need a dataset with cameras (own captures with COLMAP, or rendered data in P6).

## 2026-11-25: Eighteenth tech-lead audit — open (owner: Shreev)
- Audit state at `426f112`: `ruff check .` clean; `pytest -m "not slow and not gpu"` gives 1091 passed, 26 skipped
  (Postgres without `COUNTONCE_PG_URL`, `pycolmap`), 1 xfail (the M21 `CLAUDE.md` test); `pytest experimental`
  gives 36 passed, 1 skipped (`mcp`).
- **Neither `[!]` task is a real gate.** M5b and "sync and v3 on Fuji held-out trees" both stopped at the
  orchestrator's 25-minute session timeout (exit 124). The Fuji test-split sync/v3 run kept going detached
  (PID 1692818, 56 of 128 rows per method at 20:45). Both are re-queued: the Fuji one as a task that finishes
  that run and never starts a second copy, and M5b as M5b-1 (resumable, parallel tuning launched detached) plus
  M5b-2 (test run and report).
- **Bug found:** a synthetic-component test wrote toy rows into `results/synth_component_tuning.csv` on every
  `pytest` run. The audit deleted the toy file; the fix and a guard against tests writing into `results/` and
  `docs/` are queued first.
- **Gap found:** `Config.assoc` defaults to `v1`, but PLAN §0.1 rule 1 names v2 (or sync if it wins on
  validation) and the README quickstart passes `--assoc v2`. Queued as a decision task that applies the rule to
  the Fuji validation numbers.
- **Also queued:** order-invariance at a gate where v2 and sync actually merge (the existing Fuji result is
  vacuous for them), and a 3.11 + 3.12 CI matrix (closes the environment item below).
- **Checked and not queued:** oracle-geometry failure-source runs on Fuji. Fuji-SfM has no dataset cameras or
  depth (`scripts/run_failure_sources_fuji.py`), so they need own captures with COLMAP or rendered data (P2/P6).
  No Strong-tier scaffolding is queued: PLAN §4 rules 9-11 and §0.1 rule 2 forbid it until the Core rows are
  done, and those rows need the photos.
- **`TASKS_OVERNIGHT.md` stays gitignored** (M21, §0.1 rule 6). The audit updated it locally and did not
  force-add it; only this file is committed.
- **Still yours (human only), unchanged from the seventeenth audit:** G3 decision (and G4 once the Fuji v3
  rows are complete); the M21 `CLAUDE.md` untrack and a push to check hosted CI; H1 capture day, then P1-P5;
  H5 accounts and the M18/M19 deploy; H2, H3, H4, H6 and the demo video; Docker builds, gated weights,
  external datasets and VLM API spend.
- **After this batch:** the PLAN has no autonomous work left that doesn't need the photos or the human items.

## 2026-11-25: Gate G4 (learning helps), v3 vs v2 on Fuji-SfM — open (owner: Shreev, decision)
- **Recommendation: G4 not met on Fuji-SfM. Keep v3 out of the default path.** G4 asks for v3 to beat v2 on
  association F1 on Fuji-SfM and on own captures. v3 does not on Fuji; there are no own captures.
- Association F1, v3 minus v2, on slices where both are defined (per-tree mean, 95% bootstrap CI over trees):
  k=3 -0.36 [-0.68, -0.08] (5 slices); k=5 0.00 (2 slices, uninformative); k=10 -0.32 [-0.61, -0.06] (7 slices,
  6 trees); k=20 -0.61 [-0.71, -0.50] (13 slices, 7 trees). These are small subsets of the 16 slices per k.
- Count MAE, v3 minus v2, all 16 slices per k (8 trees): k=3 -1.7 [-3.7, -0.3]; k=5 +2.4 [0.1, 4.7];
  k=10 +25.5 [9.8, 40.2]; k=20 +47.9 [-1.5, 107.6]. Slices where v3's error is lower / equal / higher: 5/8/3,
  2/9/5, 3/2/11, 5/0/11. No multiple-comparison correction.
- The one gain (k=3 MAE) comes with lower F1; v3 merges little and the pipeline under-counts there on average,
  so it is not evidence of better matching.
- v3 is the `v001` checkpoint trained on synthetic data only. A sim-to-real gap is a likely cause but was not
  tested. PLAN §21 fallback: improve randomization or fine-tune (at least 3 seeds), and if it still fails,
  report it and keep v2 as default. Whether to spend on the fine-tune ablation is the owner's call.
- Evidence: `docs/RESULTS_FUJI.md`, `results/bench.csv` (rows from `ec31df2`, dirty tree; see DEVIATIONS).

## Open items carried from earlier audit snapshots — open (owner: Shreev)
Items from the resolved snapshots that no other open section above lists.
- **Phase 0 and captures:** gates G0/G1; the G2 foliage and easy captures; identity labels on 3-5 scenes; the
  `TODO: verify` full-paper reads in `docs/RELATED_WORK.md`.
- **Studies and release:** the pilot, the human baseline study, HF uploads and releases, the paper venue.
- **Accounts and services:** a real Postgres (`COUNTONCE_PG_URL`), the OTLP endpoint and token, the cloud GPU
  worker and cold-start measurements, cost rates (`cost_per_hour`), API keys, spend cap and prompt approval for the
  frontier-VLM baseline.
- **Weights and data:** face and plate detector weights for privacy blurring; the FSC-147, WGISD, MinneApple,
  AppleGrowthVision and video dataset downloads with license checks; the C-DOG, CountVid and DG-Net baselines.
- **Interactive GPU / Docker:** Kubric/BlenderProc rendering, the worker image build, Triton and FP8/NVFP4, the
  Track F profile measurements, a full-scale v3 training run and model-registry promotion on real data.
- **Runners and tools:** register the self-hosted golden GPU runner and pick 5 golden scenes; install the `mcp`
  extra and register the server in a client.
- **Environment:** resolved 2026-11-25. CI now runs both Python jobs on a 3.11 + 3.12 matrix, so the 3.12 `.venv` is covered; `uv lock --check` passes (lockfile unchanged).

## Overnight tasks left `[!]` — open (owner: orchestrator / Shreev, to re-queue)
- **M5b** (tune v1/v2/sync on synthetic validation, re-run v3) has a WIP commit only (`7427ba3`); re-queued as
  M5b-1 and M5b-2. Sync and v3 on Fuji held-out trees is finished (see G4 below).
- Orchestrator settings suggested by earlier audits: skip a task after 2 consecutive gate logs, and raise the
  turn cap for GPU tasks or split them.

## 2026-11-25: Nineteenth tech-lead audit — open (owner: Shreev)
- Audit state at `5d08ec8`: `ruff check .` clean; `pytest -m "not slow and not gpu"` gives 1097 passed, 26 skipped
  (Postgres without `COUNTONCE_PG_URL`, `pycolmap` not in `.venv`), 1 xfail (the M21 `CLAUDE.md` test), 16 warnings.
- **Bug found:** the 16 warnings are `DeprecationWarning: ... multi-threaded, use of fork() may lead to deadlocks`
  from `bench/synth_component.py:406`, which builds its pool with `mp.get_context("fork")`. Queued first.
- **Correction to the eighteenth audit's "no autonomous work left":** several PLAN items run on Fuji-SfM and public
  weights without photos or accounts. Queued in order: COLMAP-lifting baseline (§9.3 #4; `pycolmap` 4.2.1 is in
  `.venv-gpu`), a re-implemented C-DOG baseline and experiment E4 (§9.3 #5; no repo licence, so from the paper),
  real-data ablations on Fuji (§9.6 list; only synthetic ablations exist), order-invariance on the test trees and
  for v3 plus a re-run-VGGT-per-permutation variant (§0.6 row note), full reads of the six related-work papers
  still marked "not read in full" (§0.6 row), experiment E1 tracker side with SAM 2.1 and ByteTrack on Fuji, and
  Lighthouse on `/jobs/[id]` and `/s/[token]` (§0.6 product row note).
- **Checked and not queued:** oracle-geometry failure sources on Fuji (no dataset cameras or depth; a COLMAP
  reference would be another estimate, not an oracle); Strong-tier scaffolding (PLAN §4 rules 9-11 and §0.1 rule 2
  until the Core rows are done); the v3 fine-tune ablation from the G4 fallback (owner's call, below).
- **GPU tasks and the 25-minute session cap:** every queued GPU run must launch detached and resumable and leave a
  GATES note with PID and row count if still running, so a cap hit is not mistaken for a gate.
- **`TASKS_OVERNIGHT.md` stays gitignored** (M21, §0.1 rule 6). The audit updated it locally and did not
  force-add it; only this file is committed.
- **Still yours (human only), unchanged:** G3 decision; G4 decision and whether to fund the v3 fine-tune ablation
  (≥ 3 seeds); the M21 `CLAUDE.md` untrack, then push and check hosted CI; H1 capture day, then P1-P5; H5 accounts
  and the M18/M19 deploy plus the live URL in the README; H2, H3, H4, H6 and the demo video; Docker builds, gated
  weights (CountGD/CountGD++, VGGT-Ω, VGG-T³), CountVid/DG-Net and the video datasets (E2, E3), OneScene-Bench
  images (E5), frontier-VLM API spend.

## Fuji window alignment leaves rotation unconstrained; COLMAP test run not launched (CODE FIXED 2026-11-26, ROWS STALE)

**Update 2026-11-26:** the alignment code is fixed (rotation from camera orientations, 30 degree
median rotation gate in `check_alignment`, xfail removed). Cached VGGT windows replayed on CPU
(median rotation error, degrees): Tree04 east k=3/5/10 = 1.0 / 6.9 / 10.2 (was 78 / 152 / 26).
Tree02 west k=5 is now rejected (`ChunkAlignmentError`, negative scale from the camera centres):
once the orientations are respected, the window's centres run against the pseudo-pose path, so
that side's pseudo-poses (or the window reconstruction) still need checking.

**Still open:** every existing Fuji row and the frozen thresholds predate the fix. They were not
rerun and `fc.PROTOCOL` is still `fuji_core_v2`. Before writing any new Fuji row: bump
`fc.PROTOCOL`, check the pseudo-poses, make the VGGT path in `scripts/run_fuji_core.py` record
alignment rejections as `status=failed`, re-tune on the training trees and rerun. The text below
describes the state before the fix.

**Affects:** every Fuji row in `results/bench.csv` (VGGT v1/v2/sync/v3 and baselines), the frozen
thresholds in `configs/fuji_core.yaml` (tuned under the same alignment), `docs/RESULTS_FUJI*.md`.

**Cause:** `align_chunk` uses only camera centres for three or more views. Overview pseudo-poses
interpolate centres linearly (stride 3), so short windows are exactly collinear and Umeyama's
rotation about that line is arbitrary.

**Measured** (cached VGGT windows, median camera rotation error against the pseudo-poses, degrees):

| window | current alignment | centres + axis tips | centre singular values |
|---|---|---|---|
| Tree04 east k=3 | 78.2 | 10.4 | 0.11, 0, 0 |
| Tree04 east k=5 | 152.0 | 15.1 | 0.179, 0.042, 0 |
| Tree04 east k=10 | 26.1 | 14.9 | 0.243, 0.218, 0.02 |
| Tree02 west k=5 | 156.3 | 81.3 | 0.267, 0.023, 0 |

A COLMAP window on Tree04 east k=3 registered 3/3 views and scored pred 0 against gt 100 (smoke
run on a temporary CSV). Its error was 103 degrees. Sparse lifting also rejects many detections
(no triangulated point in the box centre), so the alignment is not the only cause of low counts.

**Done:** `--backbone colmap`, the report section and tests are committed. The 128-slice run was
**not** launched (no PID, 0 COLMAP rows in `results/bench.csv`): COLMAP would be scored against
GT built in a differently rotated frame, so the comparison would be meaningless.

**Next (owner or a new task):**
1. Fix the window alignment using camera orientations. Axis tips alone are not enough (81 degrees
   remains on Tree02 west), so also check the pseudo-poses themselves, and gate on rotation error
   in `check_alignment`. The strict xfail in `tests/test_fuji_colmap.py` must then pass.
2. Bump `fc.PROTOCOL`, re-tune on the training trees, rerun all VGGT test rows.
3. Then run (detached, resumable; roughly 5-15 s per window, not measured at k=20):
   `PYTHONPATH=. setsid nohup .venv-gpu/bin/python -u scripts/run_fuji_core.py bench --split test --backbone colmap --budget-min 600 > runs/fuji_core/test_colmap.log 2>&1 &`
   and `python -m bench.fuji_report`.

**Re-check (2026-11-25, retry of the same task after a rate limit): still OPEN.** No alignment fix
has landed: the strict xfail still fails as expected, `results/bench.csv` has 0 `_colmap` rows and no
`run_fuji_core.py bench` process is running. The COLMAP task is **not done** and was not launched.
`ruff check .` is clean; `pytest -m "not slow and not gpu"` (CUDA hidden): 1104 passed, 26 skipped,
2 xfailed. The measurements in the table above were not repeated.

- **Hold until the alignment is fixed:** the queued C-DOG/E4, real-data ablation, test-tree
  order-invariance and E1 tracker tasks use the same Fuji windows. Any of them that scores against
  window GT would add rows that need re-running.
- **README is missing the warning:** its "Measured results on real trees" section still quotes the
  affected numbers. `docs/RESULTS_FUJI.md` carries the "Known problem" section; the README does not.

**Re-check #2 (2026-11-25, third attempt of the COLMAP task): still OPEN.** Alignment fix has not landed,
`results/bench.csv` has 0 `_colmap` rows, and no `run_fuji_core.py bench` process is running (only the
orchestrator's own session matched `pgrep`). No code changed; the 128-slice run was not launched. Task left unticked.

**Re-check #3 (2026-11-25, fourth attempt of the COLMAP task): still OPEN.** No commit since
re-check #2 touches the alignment, the strict xfail still fails as expected, `results/bench.csv` has 0
`_colmap` rows and no bench process is running. Not launched; task left unticked. Re-queuing this task
alone cannot succeed: queue the alignment fix (steps 1 and 2 above) ahead of it.
`ruff check .` clean; `pytest -m "not slow and not gpu"`: 1104 passed, 26 skipped, 2 xfailed.

**Re-check #4 (2026-11-25, fifth attempt of the COLMAP task): still OPEN.** No change to the alignment
code since f2d08ac, `results/bench.csv` has 0 `_colmap` rows, no bench process is running. Not launched;
task left unticked. The 128-slice run would be scored in a wrongly rotated GT frame; fix alignment first
(steps 1-2 above).

## Fuji COLMAP test-split run in progress (OPEN, 2026-11-25)

**Do not start a second copy.** Launched detached (setsid) from the main checkout; PID 1969296, 39 of 64
`countonce_v2_colmap` rows in `results/bench.csv` when this was written (0 with status `error`). The
split has 64 windows, not 128. Log: `runs/fuji_core/test_colmap.log`; `runs/fuji_core/test_colmap.done`
holds the exit code when it ends. k=10 windows took about 50-80 s each; k=20 is not yet timed.

- This supersedes the "not launched" status above: COLMAP is now placed in the VGGT window's frame
  (DEVIATIONS). The VGGT alignment bug itself is **still open**, so all rows still need a rerun after it.
- So far: k=3 and k=5 each have 14/16 registration failures (`no_model`).
- The rows were produced from a dirty tree (the code of the commit that adds this note).
- While it runs, a full pytest in this checkout errors in the `results/` write guard; test in a copy.

**When it has ended:** if it died, rerun the same command (it skips finished slices):
`PYTHONPATH=. setsid nohup .venv-gpu/bin/python -u scripts/run_fuji_core.py bench --split test --backbone colmap --budget-min 600 > runs/fuji_core/test_colmap.log 2>&1 &`
Then run `python -m bench.fuji_report`, commit the report and summary CSV, and tick the task once
there are 64 ok-or-failed rows.

**Status check (2026-11-25 23:31, sixth session on this task):** run still going, PID 1969296, 42 of 64
`countonce_v2_colmap` rows, no `.done` file. No second copy started, no code changed. Task left unticked;
when it ends, follow "When it has ended" above (report, commit, tick).

**Status check (2026-11-25 23:51, seventh session on this task):** run still going, PID 1969296, 51 of 64
`countonce_v2_colmap` rows, no `.done` file; no second copy started. All 16 k=10 windows are finished (0
registration failures), so the report now has the k=10 row; k=20 stays TODO. k=20 windows take about
2.5 min each, so expect the run to end around 00:25. Task left unticked.

**Re-queue note:** a session after the run ends only needs `python -m bench.fuji_report`, a commit and the
tick. "128 test slices" in the task is 64 windows under the current protocol (8 trees x 2 sides x 4 k);
the 128 `countonce_v2` test rows include 64 from the superseded v1 protocol.

**Status check (2026-11-25 23:55, eighth session on this task):** run still going, PID 1969296, 51 of 64
`countonce_v2_colmap` rows, no `.done` file; no second copy started, no code changed. Task left unticked;
when it ends, follow "When it has ended" above (report, commit, tick).

**Resume (2026-11-26 00:58, ninth session): run in progress again, do not start a second copy.** The
first run was stopped by SIGTERM at 51 of 64 rows (`test_colmap.done` held `exit=143`; log kept as
`runs/fuji_core/test_colmap.run1.log`). The remaining k=20 windows were relaunched detached,
**PID 2086810**, 52 of 64 rows at the time of writing, at a few minutes per window. It runs from the
worktree `~/countonce_prefix` (pre-fix commit `df5a7c8`, see DEVIATIONS) and appends
to the main checkout's `results/bench.csv`; do not remove that worktree while the PID is alive. A
copy of the CSV from before the relaunch is `runs/fuji_core/bench.before_colmap_resume.csv`.
While it runs, test in a clone (the `results/` write guard).

When `runs/fuji_core/test_colmap.done` says `exit=0` and there are 64 ok-or-failed rows:
`.venv/bin/python -m bench.fuji_report`, commit, tick the task, then
`git worktree remove --force ~/countonce_prefix` (its `data`, `runs` and `weights`
are symlinks; only the links go). If it died, rerun from inside that worktree (finished slices are
skipped):
`PYTHONPATH=. setsid nohup ~/countonce/.venv-gpu/bin/python -u scripts/run_fuji_core.py bench --split test --backbone colmap --budget-min 600 --csv ~/countonce/results/bench.csv > ~/countonce/runs/fuji_core/test_colmap.log 2>&1 &`

**Status check (2026-11-26, tenth session on this task):** run still going, PID 2086810, 53 of 64
`countonce_v2_colmap` rows (k=20 windows remain), no `.done` file; no second copy started, no code
changed. Task left unticked; when `test_colmap.done` says `exit=0`, follow "Resume" above
(report, commit, tick, remove the worktree).

**Status check (2026-11-26 01:06, eleventh session on this task):** run still going, PID 2086810, 55 of 64
`countonce_v2_colmap` rows, no `.done` file; no second copy started. The rows so far have unique
windows that all match a VGGT v2 window, with equal GT on every ok row. Remaining windows are all
k=20 at about 3 min each. Task left unticked; **do not re-queue it before the `.done` file exists**,
then follow "Resume" above. Also replaced personal home paths in this file with `~` (the hygiene
test failed on them).

**Status check (2026-11-26 01:07, twelfth session on this task):** run still going, PID 2086810, 55 of 64
`countonce_v2_colmap` rows, no `.done` file; no second copy started, no code changed. Task left unticked;
when `test_colmap.done` says `exit=0`, follow "Resume" above (report, commit, tick, remove the worktree).

**Status check (2026-11-26 01:14, thirteenth session on this task):** run still going, PID 2086810, 57 of 64
`countonce_v2_colmap` rows (all k=3/5/10 done, only k=20 windows left), no `.done` file; no second copy
started. ruff clean; fast tests in a clone: 1101 passed, 31 skipped, 1 xfailed. All 28 failed rows so far
are `ReconAdaptError: ... no_model`. Task left unticked; follow "Resume" above once `test_colmap.done`
says `exit=0`.

**Status check (2026-11-26 01:26, fourteenth session on this task):** run still going, PID 2086810, 60 of 64
`countonce_v2_colmap` rows (only k=20 windows left), no `.done` file; no second copy started, no code
changed. Task left unticked; when `test_colmap.done` says `exit=0`, follow "Resume" above
(report, commit, tick, remove the worktree).

**Resolved (2026-11-26):** the run finished (`exit=0`), 64 of 64 rows (36 ok, 28 failed `no_model`).
The report and docs are updated, the task is ticked and the `countonce_prefix` worktree is removed.
The "Resume" steps above are done. Still open: the joint rerun of all Fuji rows after the alignment fix.

## Real-data Fuji ablations: BLOCKED on the joint rerun (2026-11-26)

Not started; no code changed, no rows added. Task left unticked.

- **Why:** every `countonce_v2` row in `results/bench.csv` (protocol `fuji_core_v2`) and the frozen
  thresholds in `configs/fuji_core.yaml` predate the window-alignment rotation fix (see the
  "Next" note in the handoff and DEVIATIONS). The task needs *paired* differences against the
  frozen v2. Ablation rows scored with the fixed alignment against stale v2 rows would pair two
  different GT frames. The cached chunked reconstructions are also pre-fix, so "reuse cached
  reconstructions" would reuse misaligned chunks.
- **Also:** `scripts/tune_fuji_cdog.py` (PID 2130274) and its `run_fuji_core.py` child were using the
  GPU and the same `runs/fuji_core/cache` when this session started; a second writer is unsafe.
- **Resume (in order):** (1) bump `fc.PROTOCOL`, record VGGT alignment rejections as `failed`;
  (2) invalidate/recompute the chunked recon cache; (3) re-tune on val and refreeze
  `configs/fuji_core.yaml`; (4) rerun v2 on the test trees; (5) then add `--ablation` to
  `scripts/run_fuji_core.py` (flip one `Config` switch from the frozen v2, method
  `countonce_v2__<ablation>`, k=10 only, masks last) and run the 64 rows.

### Retry (2026-11-26): still BLOCKED; `--ablation` option added, 0 of 64 rows

- Block unchanged: `fc.PROTOCOL` is `fuji_core_v2`, v2 not rerun, and the C-DOG tuning job
  (PID 2130274) was still running on the shared cache. No benchmark was launched.
- Added: `scripts/run_fuji_core.py bench --split test --ks 10 --ablation
  {no_cannot_link,no_hallucination_filter,sam_masks}`. It exits while the protocol is stale
  (`fc.STALE_PROTOCOLS`). Unit-tested for arguments and overrides only; the run path has never
  executed.
- **Appearance ablation needs code first.** No caller passes `DinoV2Extractor` to
  `V2Associator` and GroundingDINO sets no `Detection.feature`, so frozen v2 uses no appearance
  and `appearance=False` is a no-op. Wire the extractor into the pipeline, then compare
  "with DINOv2" against frozen v2 (this adds the feature rather than removing it). DINOv2 weights
  were not in the local HF cache. So the full task is 64 rows only after that; 48 are ready now.
- **Correction to resume step (2):** k=10 windows are a single chunk (`chunked.py`,
  `len(frames) <= chunk_size`), so their cached reconstructions do not depend on the alignment
  fix. The overview reconstructions are stitched, so `overview --force` is needed to rebuild the
  pseudo-poses. k=20 windows are also single-chunk.
