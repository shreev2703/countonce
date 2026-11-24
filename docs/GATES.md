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
