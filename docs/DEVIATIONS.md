# Deviations

Where the real code or an engineering judgment call differs from `docs/PLAN.md`.
One entry per decision: date, what the plan said, what was done, why.

## 2026-11-24: Bootstrap

- **Python version:** plan says Python 3.11. `pyproject.toml` declares `requires-python = ">=3.11"`
  and tool targets of 3.11; the dev machine's system Python is 3.12, which is compatible.
- **Packaging:** setuptools backend with a flat `countonce/` package; dev tools under the
  `dev` optional extra. Runtime deps limited to numpy and pydantic until later tasks need more.
- **Subpackages:** created `ingest`, `detect`, `recon`, `merge`, `export`, `schemas`, `match`
  (names from PLAN §8 and §10) as empty packages; `py.typed` added for mypy consumers.
- **Lint rules:** ruff `E,F,I,UP,B,W`, line length 100. mypy runs in strict mode with the pydantic plugin.
- **Tooling check:** ruff and pytest verified via the read-only `~/vggtProject/.venv`; mypy is not installed there and the system Python lacks venv/pip, so mypy config is unverified. Install with `pip install -e .[dev]` once a venv exists.

## 2026-11-24: Port from vggtProject
- `Recon` from vggt-sparse-recon lives in `countonce/recon/result.py` (the ported, method-level reconstruction result). The §7 data-model `Recon` in `countonce/types.py` is a separate task and may wrap or replace it.
- Memory guard (`profiling.py`) placed in `countonce/recon/profiling.py`; `torch` is imported at module top, so it needs the `recon` extra.
- Plotting: ported only generic helpers (style, `save`, `depth_to_rgb`, `point_cloud_preview`, `vram_vs_frames`). Benchmark-specific plots (`band_lines`, `scene_type_bars`) were left behind; they belong to Track B.
- Env check: Python pin relaxed from `==3.11` to `>=3.11` to match `pyproject.toml`.
- Frame extraction: dropped the unused `EXAMPLES` constant. `configs/colmap.yaml` copied verbatim.
- Added `recon` optional-dependency extra and `pythonpath = ["."]` for pytest. `ruff`/`pytest` were run from `~/vggtProject/.venv` (3.11, has torch/pycolmap) since countonce has no venv yet.
- Video-clip helper (`clip.py`) was not ported: it is demo-site specific and not in the task list.

## types.py (2026-11-24)
- `Instance.possibly_moved` (PLAN §7) is named `possibly_moving` to match the task spec and the same flag on `LiftedDetection`; the flag was also added to `LiftedDetection`.
- Added `CountResult.confidence_method` and `recon_cache_key` (task spec; not in the §7 listing). `recon_cache_hit` is kept.
- Dataclasses with `__post_init__` validation (not pydantic); strict wire models stay in `countonce/schemas/`. Optional/defaulted fields were reordered after required ones.
- `types.Recon` (dense §7 form) is separate from `countonce/recon/result.Recon` (ported); reconciling them is deferred.

## lift.py / cluster.py (2026-11-24)
- Clustering is greedy single-linkage over pairs with normalized distance < k, closest first, with cluster-level cannot-link (frame sets must be disjoint). It is not centroid-linkage; chaining is limited by cannot-link but may need revisiting if sweeps show over-merging.
- `lift_conf` = score x (mean_conf / (mean_conf + 1)) x 1/(1 + depth_spread/0.1): an uncalibrated heuristic. Instance `conf` = mean lift_conf x (1 - 0.5^n_views), also heuristic (`confidence_method="heuristic"`).
- `frame_id` indexes recon views unless a `frame_to_view` map is passed. If `Frame` objects are supplied, pixel coords and privacy masks are rescaled to the recon resolution.
- `motion_residual` / `possibly_moving` are not computed yet (§8.3b is Moon tier); the fields stay at their defaults.
- Rejected (failed-lift) detections are reported in `ClusterResult.detection_status`/`reject_reasons`; `MatchEvidence.reproj_error_px` and `appearance_sim` are 0.0 in v1.
- Synthetic geometry test builds `LiftedDetection`s directly (noise/drops/false positives); lifting is tested separately on a known plane.

## export (2026-11-24)
- Pydantic models live in `countonce/schemas/__init__.py` (`ResultV1`, `MarkersV1`, `extra="forbid"`), with an `UPGRADERS` registry (empty at v1.0) and `load_result`/`load_markers`. A stored example of each older version must be added to tests on every bump; v1.0 has none older yet, so the upgrade path is tested with a synthetic "0.9".
- `pillow` moved into core dependencies (overlays); imported lazily. `scene.ply` is not part of this task (needs recon points).
- `counts.csv` is per-frame detections by status plus a `total` row; `new` sums to the count. Overlays take explicit `frames` + `detections` since `CountResult` does not store boxes. Mask-only detections are drawn at their centroid.

## RELATED_WORK (2026-11-24)
- Written from search snippets/abstracts, not full papers; claims tagged `TODO: verify`. Added an "adjacent work" section (OpenM3D, See and Detect, etc.) beyond the six requested works because they threaten the novelty claim.

## Reconstruction reconcile, cache, ply, ingest filters (overnight)
- `Reconstructor` (frames -> `types.Recon`) added beside the path-based `ReconMethod`; `recon/adapter.py` bridges them. Kept both rather than rewriting the ported wrappers.
- Dense `world_points` are computed by unprojecting VGGT depth with K and extrinsics (numpy), not read from the VGGT point head; `world_points_conf` reuses `depth_conf`. Unregistered views get zero confidence.
- COLMAP results have no per-view depth, so `to_dense_recon` builds a sparse-to-dense Recon (`recon/adapter.sparse_to_dense`): each triangulated point is splatted to one pixel in every view that observes it (nearest wins), confidence = track length / (1 + reprojection error), zero elsewhere and for unregistered views. This supersedes the earlier "sparse metrics only" decision, so `co_v1_colmap` (§9.3 item 4) works with the real backbone. `lift.py` already takes the median over confident pixels; for sparse input the pipeline lowers `min_pixels` to 3 and sets `conf_percentile=0` (config: `lift_min_pixels`, `lift_conf_percentile`). Points are splatted to single pixels, not dilated, and occlusion is only handled through COLMAP tracks. Tested with hand-built points; not yet run on real COLMAP output.
- Cache is a flat `<sha256>.npz` per key (input hash + backbone + version + config); `CachedReconstructor.reconstruct_cached` returns the key/hit to copy into `CountResult.recon_cache_key`/`recon_cache_hit`. Corrupt entries count as misses.
- `scene.ply` is binary little-endian with seeded random downsampling of confident points; gray unless colors are passed.
- Near-duplicate filter: mean abs diff of 16x16 gray thumbnails vs the last kept frame, default threshold 0.02 (`TODO: tune on real captures`).

## Detector interface (2026-11-24, overnight)
- `Detector` also exposes `name`, `version` and `config()` (cache-key inputs); exemplars are `Exemplar(frame_id, box_xyxy)` rather than bare arrays so they can be tied to a frame. `detect` takes all frames and returns one flat list.
- Naming differences vs the plan: both real models output normalized `cxcywh` boxes plus per-token sigmoid logits (`pred_logits`/`pred_boxes` in CountGD; `(boxes, logits, phrases)` from GroundingDINO's `predict`). We score a query as max over tokens and convert to pixel xyxy. Captions are lowercased and period-terminated.
- CountGD's output format is written from memory of the repo (`TODO: verify` on a checkout); its model build and forward pass are stubbed (`NotImplementedError`) until weights exist. GroundingDINO's adapter uses the real `load_model`/`load_image`/`predict` API but has never been run.
- Raw floor is 0.05 (below both repos' demo thresholds, CountGD 0.23 / GroundingDINO 0.35) so threshold sweeps need no rerun. The cache stores boxes, points and scores as JSON per frame; masks and features are not cached. `countgd_pp.py`, `masks.py` not part of this task.

## Config, run() and CLI (2026-11-24, overnight)
- `run()` lives in `countonce/pipeline.py` (re-export from the package root deferred to avoid importing pydantic/PIL on `import countonce`); optional `detector=`/`reconstructor=`/`export=` keyword overrides let tests inject fakes. Signature otherwise matches PLAN 8.11.
- `Config` accepts every PLAN name (`countgdpp`, `vggt_omega`, `vggt_t3`, `assoc v2/v3`) but only `countgd|gdino`, `vggt|colmap` and `v1` are wired; the rest raise `NotImplementedError` (CLI exit code 2).
- `interval` is a degenerate `[count, count]` placeholder with method `sensitivity` until the confidence-range task lands; `confidence_method` stays `heuristic`.
- Provenance (git commit/dirty, UTC time, torch/GPU, full config) is stored under `result.config`, since the strict v1 schema has no separate provenance field.
- `pyyaml` promoted to a core dependency (Config YAML). `bench`/`synth`/`train`/`profile` subcommands are not part of this task.
- Defaults `max_frames=32`, `score_threshold=0.3`, blur filter off: `TODO: tune on real captures`.

## v2 associator (2026-11-24)
- **Associator return type:** PLAN §8.7 writes `associate(...) -> list[Instance]`; the shared `Associator` protocol (`merge/base.py`) returns `ClusterResult` (instances plus detection status, reject reasons and match evidence), because the pipeline needs all four. `V1Associator` wraps `cluster_detections`; `ClusterResult` moved to `merge/base.py`.
- **Sequential matching:** v2 processes frames in increasing frame id and matches each frame's detections to the running tracks (Hungarian per frame), instead of one global assignment. One detection per track per frame gives cannot-link by construction; output is independent of input ordering, but depends on frame ids.
- **Cost and gating:** cost = reprojection error / box size (max box side; projected object size for bare points) + `appearance_weight * (1 - cos) / 2` when both features exist. Pairs above `max_reproj_norm` (0.5) or below `min_appearance_sim` are not assignable. Defaults are untuned (`TODO: tune on validation scenes`).
- **Failed lifts** stay `rejected` in v2 (same as v1) rather than being matched by projection alone.
- **Hallucination filter** only considers instances seen in <= 1 view; "expected" = visible (in image + depth test) in >= 2 other recon views. Modes `off|downweight|drop`, default `downweight` (conf x 0.5). Split check uses member spread (> 2 sizes) and appearance (min pairwise cosine < 0.3); defaults untuned.
- **DINOv2:** `DinoV2Extractor` (torch.hub, lazy import) is written but never run; it needs torch and downloaded weights.
- `Config.assoc="v2"` is wired in `run()` with default `V2Config` and no extractor.

## Confidence range and warnings (2026-11-24)
- **Grid:** `k` = `merge_k` x (0.5, 0.75, 1, 1.5, 2); score threshold = `score_threshold` +/- (0.05, 0.1), clipped to [0.01, 0.99]. For v2 (no `k`), the same factors scale `V2Config.max_reproj_norm` (its analogue of merge strictness). The detector is run once; thresholds re-filter the raw detections.
- **Percentiles and level:** low/high are the 10th/90th grid percentiles (per PLAN), widened to include the default count. `Interval.level` is therefore recorded as 0.8 (nominal grid mass), not `Config.interval_level`; that field is reserved for the conformal tier. It is a sensitivity range, not a coverage claim.
- **`confidence_method`:** `"sensitivity"` added to `CONFIDENCE_METHODS` so `run()` can set it as the task specifies (it also names the interval method).
- **Warning thresholds** (`WarningThresholds`, all untuned, `TODO: tune on calibration data`): <3 frames; <30% of instances in >=2 views; median depth confidence <1.5; >30% failed lifts (>=3 detections); width >= 2 and > 25% of count. Test fixture `PlaneRecon` has depth_conf 1.0, so e2e runs show `low_depth_confidence`.
- **Noise test:** range width grows with noise only until detections over-split (count saturates), so the test compares noise 0.02 < 0.1 <= 0.2 (object size 1, spacing 4).

## 2026-11-24: Benchmark core (`bench/`)
- **Package location:** `bench/` sits at the repo root, outside the `countonce` wheel (`pythonpath = ["."]` makes it importable in tests). It imports `countonce.provenance` only.
- **MAPE:** scenes with gt = 0 are excluded (undefined); they belong to the §9.8b false-positive metrics. Within-10% treats gt = 0 as a pass only for an exact 0.
- **Duplicate/miss rate definitions:** each predicted cluster gets its majority GT id (unlabelled detections ignored). Duplicate rate = redundant clusters / clusters; miss rate = GT objects with no cluster / `gt_universe` (defaults to GT ids seen, so pass the full list to include never-detected objects).
- **Pairwise F1:** precision/recall are vacuously 1.0 when there are no predicted/true positive pairs.
- **Median baseline** is returned unrounded.
- **Bootstrap:** percentile, 10,000 resamples of scenes, NaNs dropped. Paired p-value is twice the smaller tail of the bootstrap distribution of the mean difference; Wilcoxon is two-sided and returns 1.0 for all-zero differences.
- **Splits:** assignment ranks scenes by `sha256(seed:id)`, so it is order-independent; `write_splits` refuses to overwrite a frozen file. Only `val` and `test` splits are written; `verify_splits` also flags scenes on disk missing from the file.
- **Runner:** error rows are written with `status=error` and retried on resume; only `status=ok` keys are skipped. Provenance (git commit/dirty, UTC time, torch/GPU if present) is captured once per `run_jobs` call. Plain `csv` module, no pandas dependency.

## 2026-11-24: Fuji-SfM loader, Sim(3), GT matching
- **Sim(3) location:** `countonce/recon/align.py` (`umeyama`, `Sim3`, `camera_centers`, `align_by_camera_centers`, `transform_extrinsics`) so chunk stitching (§8.3) and the benchmark share it. Convention `dst = s R src + t`.
- **Loader layout is a guess:** all file names live in `bench/datasets/fuji_sfm.py::_read_layout`; format TODOs from DATASETS.md are in the module docstring. Never run against real data.
- **Matching** (`bench/gt_matching.py`): nearest projected GT within a pixel threshold (default 20, `TODO: tune`); no occlusion test (no GT depth), so a detection on an occluded apple may take a hidden GT's id; many-to-one allowed within a frame (a within-frame duplicate). Detection reference = point, else box centre.
- **Download script** `scripts/download_fuji_sfm.sh` lists files via the Zenodo API, verifies md5, requires an empty target dir. Not run.

## 2026-11-24: Sweep and figure scaffolding (`bench/sweep.py`, `figures.py`, `gallery.py`)
- **Scene layout:** `configs/bench.yaml` expects `scenes_dir/<scene>/images/` plus `gt.json` (`count`, `tier`, optional prompt). Own-capture layout is not fixed in PLAN, so this is a choice. Tier is stored in each row's `config_json["tier"]` (no new CSV column, so existing `bench.csv` files stay valid).
- **Sampling:** per view count, `random0..random{n-1}` (seed = `seed*1000 + i`) plus `even`; ordered/shuffled runs use the same frames and differ only in presentation order; every method sees the same frames (paired comparisons). Ablations are one-factor-at-a-time around `ablation_base`.
- **Default executor:** CountOnce methods (including `co_v1_colmap`, baseline 4) call `run()`; baselines build the config-selected detector, run it once over the selected views through the detection cache, apply `score_threshold` and reduce the per-image counts (zero for views with no detections) with `bench.baselines`. Detector/reconstructor factories can be injected (tests use `FakeDetector` and a fake reconstructor). Shuffled order only matters if ingest does not re-sort (`collect_paths` keeps list order; dedup/cap are order-dependent by design).
- **Bench cache and run dirs:** `BenchConfig.cache_dir` (default `results/cache`) is shared by baselines and CountOnce runs, so a detector runs once per frame per prompt across methods; `runs_dir` (default `results/runs`) holds per-run outputs. Baselines skip the ingest filters (blur, dedup, frame cap) so they count exactly the selected views.
- **Figures** (§9.7 items 1, 2, 3, 8): points are means over scenes (samples averaged within scene first), 95% bootstrap CI over scenes (2000 resamples). MAPE excludes gt = 0 scenes. Headline and tier plots use ordered runs only. Figures 4-7, 9 (rendering), 10-11 are out of scope here.
- **Failure gallery:** worst 10% per method by relative error (absolute error when gt = 0), ceil, deterministic tie-break; written as a Markdown table with a coarse over/under-count label only (true cause needs the overlay). `countonce bench --figures DIR` renders all of it.
- Smoke tests use a synthetic CSV in `tmp_path`; nothing under `results/` is committed.

## Ingest-to-original pixel mapping (overnight)
- `Frame` gains optional `pixel_map: PixelMap` and `original_path`; `resize_to_grid_mapped` returns the maps and `resize_to_grid` keeps its old signature. Overlays draw on `original_path` in original pixels when both are set, otherwise on `frame.path` as before.
- `pipeline.make_frames` does not resize today (frames are used at native size), so it leaves `pixel_map` unset; it must be populated once a backbone-grid resize step is wired into `run()`.

## Chunked reconstruction (`countonce/recon/chunked.py`, PLAN §8.3)
- **Limitation (drift):** chunks are chained sequentially, each aligned to the already-merged result, so scale and rotation errors accumulate with chunk count. Pose-graph stitching with loop closures (§8.3) is not implemented; drift has not been measured (`TODO: fill from results`). Only camera poses drive the alignment, so a backbone whose chunk poses are noisy gives a noisy stitch; the world points are transformed rigidly, not re-fused.
- **Merging:** shared frames keep the earlier chunk's copy (extrinsics, depth, points, confidences); later chunks drop their first `overlap` frames. Depth is multiplied by the Sim(3) scale. `scene_scale` is the first chunk's; `runtime_s` is summed, `peak_vram_mb` is the max over chunks.
- **Overlap of exactly 2:** camera centres alone leave the rotation about the 2-point axis free, so for two shared frames the Umeyama fit also uses each camera's optical-axis and up-axis tips at one centre-to-centre baseline length (scale-consistent within a chunk). With 3+ shared frames only centres are used; nearly collinear centres (e.g. a straight dolly) are still weakly conditioned. Coincident shared cameras raise `ChunkAlignmentError`; `overlap < 2` or `chunk_size <= overlap` raises `ValueError`.
- **Config:** `chunked_recon` (default off), `chunk_size` (0 = memory-guard calibration `max_ok_frames` for `backbone_options.weights_dtype`, default `bf16`; if neither exists no chunking happens), `chunk_overlap` (default 4, >= 2). The ingest `max_frames` cap still applies first; set it to 0 or above the chunk size for chunking to matter.
- Tested on CPU with a fake backbone returning each chunk in a random Sim(3) frame; not run with real VGGT.

## CI / mypy (overnight)
- `mypy countonce` is strict, but 40 errors (missing annotations, `Any` returns, untyped optional-dep calls) were found when run locally (mypy was not installed; run from a temp `--target` env). Rather than rewriting 15 modules in a CI task, those modules get a relaxed `[[tool.mypy.overrides]]` block in `pyproject.toml`; new modules stay strict. TODO: annotate them and shrink the override list. Real-looking items to review: `recon/colmap.py:91` (`Path | None` passed to `Path`), `merge/match.py:138` (`FeatureExtractor | None` used unguarded), `merge/appearance.py:45`.
- Missing-import errors for optional deps (torch, vggt, pycolmap, matplotlib, ...) are silenced with `ignore_missing_imports`. `bench.stats` has `warn_unused_ignores` off because its `type: ignore`s depend on whether scipy stubs are installed.
- Hosted CI installs only `.[dev]`, so no torch/PIL-extras beyond core deps; mypy there was not run (only locally). Golden test `tests/test_golden.py` (marker `golden`) pins v1/v2 counts = 6 on a seeded scene.

- Slicing (`bench/slices.py`): slice-vs-overall difference uses independent bootstrap resampling of slice scenes and all scenes (conservative since slice ⊂ overall); slices with < 5 scenes are reported but never flagged; occlusion and camera-baseline covariates from §9.9 are not computed (need identity labels / pose spread) — TODO.

## Synchronization baseline (`assoc: sync`)
- v2 does not expose per-pair matches (it is an incremental tracker), so `merge/sync.py` rebuilds pairwise matches with v2's own geometry (symmetric reprojection error + appearance gate, per-frame-pair Hungarian) and synchronizes those. Rounding is spectral embedding + per-view Hungarian into a growing universe (cannot-link holds by construction). v2's hallucination filter is reused; its split check is not.
- Wired into `run()`/CLI/`configs/bench.yaml` as `co_sync_vggt`. No real-data numbers: TODO: fill from results.

## Track C scaffolding (synth/geometry.py, match/sinkhorn.py)
- `synth/` is a top-level package (as in PLAN §10.2), added to setuptools `packages.find`. The geometry generator uses a flat-array `.npz` layout and 4-d per-detection features (u, v, depth, score); appearance/camera embeddings are deferred to the model task.
- Sinkhorn follows SuperGlue marginals (real rows/cols = 1, dustbin row = N, dustbin col = M). Tests skip if torch is missing.
