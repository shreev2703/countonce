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
- `mypy countonce` is strict. The relaxed `[[tool.mypy.overrides]]` block that covered 16 modules has been removed (2026-11-24 paydown): all of `countonce/` passes `mypy` with no per-module relaxations, run from a temporary `--target` install (mypy 2.4.1, numpy, scipy, scipy-stubs, pydantic, types-PyYAML; no torch/pycolmap, matching hosted CI). Only the `ignore_missing_imports` override for optional heavy deps (torch, vggt, pycolmap, matplotlib, groundingdino, transformers) remains.
- The three suspected bugs were typing-only, not runtime bugs: `recon/colmap.py` `from_yaml` (restructured so `Path | None` narrows; behaviour unchanged), `merge/match.py` extractor use (`missing` is only filled when an extractor exists; now an explicit local + assert), `merge/appearance.py:45` (`_model` was inferred as `None`; now `Any`). Pinned by `tests/test_typing_regressions.py`. One latent edge fixed: `DinoV2Extractor.extract` now raises a clear `ValueError` for a detection with no box/point/mask instead of mis-narrowing (cannot happen for validated `Detection`s).
- Remaining: `bench/` is not in mypy's `files` (CI checks `countonce` only); `bench/metrics.py:23,50` has two `no-any-return` errors if it is checked. `lift_detections` takes a `LiftKwargs` TypedDict instead of `**kwargs: Any`; `vggt.forward` uses `with torch.no_grad()` rather than the decorator so it type-checks whether or not torch is installed. Not verified against an environment with torch installed.
- Missing-import errors for optional deps (torch, vggt, pycolmap, matplotlib, ...) are silenced with `ignore_missing_imports`. `bench.stats` has `warn_unused_ignores` off because its `type: ignore`s depend on whether scipy stubs are installed.
- Hosted CI installs only `.[dev]`, so no torch/PIL-extras beyond core deps; mypy there was not run (only locally). Golden test `tests/test_golden.py` (marker `golden`) pins v1/v2 counts = 6 on a seeded scene.

- Slicing (`bench/slices.py`): slice-vs-overall difference uses independent bootstrap resampling of slice scenes and all scenes (conservative since slice ⊂ overall); slices with < 5 scenes are reported but never flagged; camera-baseline spread (RMS camera-centre spread / `scene_scale`) and mean occlusion (synthetic GT visibility, or an identity-span proxy; NaN when absent) are now computed in `SceneMeta`.
- §9.8b `no_objects_found`: fires when count is 0, or count <= 2 with interval low 0. The below-threshold summary is in the warning message (and structured in `result.config["below_threshold"]`) so `WarningModel` is unchanged and no schema bump or upgrader was needed. `Detection` gained optional `label` (default None; summary falls back to the prompt). Thresholds are my defaults.

## Synchronization baseline (`assoc: sync`)
- v2 does not expose per-pair matches (it is an incremental tracker), so `merge/sync.py` rebuilds pairwise matches with v2's own geometry (symmetric reprojection error + appearance gate, per-frame-pair Hungarian) and synchronizes those. Rounding is spectral embedding + per-view Hungarian into a growing universe (cannot-link holds by construction). v2's hallucination filter is reused; its split check is not.
- Wired into `run()`/CLI/`configs/bench.yaml` as `co_sync_vggt`. No real-data numbers: TODO: fill from results.

## Track C scaffolding (synth/geometry.py, match/sinkhorn.py)
- `synth/` is a top-level package (as in PLAN §10.2), added to setuptools `packages.find`. The geometry generator uses a flat-array `.npz` layout and 4-d per-detection features (u, v, depth, score); appearance/camera embeddings are deferred to the model task.
- Sinkhorn follows SuperGlue marginals (real rows/cols = 1, dustbin row = N, dustbin col = M). Tests skip if torch is missing.

## Backbone-grid pixel mapping (ingest/frames.py, lift.py, merge/match.py)
- The task said to fill `Frame.pixel_map`, but that field already means grid -> original for overlays (and `Frame` is native-size in `run()`, so the two directions would collide). Added a separate `Frame.grid_map: GridMap` (original -> backbone grid, exact per-axis scale + crop/pad offset, with `to_grid`/`to_original`/`at_size`). `pixel_map` is unchanged; `grid_pixel_map` is now derived from the same exact map (the old one assumed a uniform scale, ignoring the rounding of height to the 14 px patch grid).
- `backbone_grid_map(w, h, mode)` mirrors `vggt.utils.load_fn.load_and_preprocess_images` (facebookresearch/vggt) for `crop` and `pad`. Not verified against the installed vggt package (not importable here); only the arithmetic is mirrored.
- Backbone spec: `preprocess_mode` attribute on the method (`VGGTMethod = "crop"`), exposed by `MethodReconstructor`; `run()` reads it before wrapping with chunked/cache. Backbones without it (COLMAP, test fakes) get no `grid_map` and keep the plain rescale.
- `lift` clips boxes to the real-pixel region of the grid (not padding), and returns `fail_reason="outside_recon"` for boxes/points wholly outside it. v2/sync reproject through `_Views.to_frame` (inverse map). Partially cropped boxes are clipped, not dropped.
- Frame sizes in a recon that differ from the grid (e.g. downscaled) are handled by `GridMap.at_size`.

## 2026-11-24: Core figure set and failure gallery
- **CSV schema is append-only:** `peak_vram_mb`, `timings_json`, `assoc_f1` and `run_dir` were appended to `bench.runner.COLUMNS`. `read_rows` accepts the original 18-column header (or any prefix of the current one) and fills missing columns with `""`; `run_jobs` rewrites an old file once with the new header (atomic replace) before appending. Test: `tests/test_bench_core_figures.py`.
- **`assoc_f1` is not filled by the default executor yet:** pairwise F1 needs identity labels (GT projection matching, `bench/gt_matching.py`) that the own-capture `gt.json` scenes do not carry. The column and the figure exist; rows stay blank (and the figure is skipped) until the Fuji-SfM path fills it. TODO: wire after the real loader is verified.
- **Figures 4, 6, 7** (`fig_assoc_f1`, `fig_runtime_vs_views`, `fig_ablation`) skip silently in `make_all` when their data is absent. Ablation deltas pair on per-scene mean percent error (ordered runs) against the `ablation_base` method; the base is inferred from the `base+field=value` names unless passed (`countonce bench --figures` passes `ablation_base`).
- **Failure gallery:** each sampled run gets a contact sheet (max 12 evenly subsampled overlay frames) in `<figures>/gallery/`, embedded in its Markdown row; runs are located by the CSV `run_dir`, else `runs_dir/<run_key>`. Rows without overlays say "no overlay". Baselines have no overlays or VRAM.

## 2026-11-24: Split conformal and Jackknife+ (countonce/conformal.py, bench/calibration.py)
- `Config.interval_method` is `sensitivity|conformal|jackknife_plus`; `conformal` is recorded in `result.json` as `split_conformal` (the existing `INTERVAL_METHODS` name). `confidence_method` stays `sensitivity` (the sigma source). `Config.calibration_path` points at the JSON `CalibrationArtifact`; it is level-agnostic, the level comes from `interval_level`.
- Jackknife+ uses unnormalized residuals so the stated 1-2 alpha worst-case guarantee holds; only split conformal is sigma-normalized. Tuned parameters built in: `k` (factor of `merge_k`, from stored per-scene k-grid counts) and `scale` (global count multiplier). At run time `k` re-counts once per distinct LOO factor.
- >= 15 real-scene rule: `conformal` with fewer real calibration scenes falls back to Jackknife+ (if the artifact has LOO fits) else to the sensitivity range, with a `conformal_fallback` warning. Missing artifact or too few scenes for the level (e.g. 95% with n < 19) is a hard `CalibrationError`.
- Builder refuses `split="test"`, any scene recorded in `test`, and real scenes not in `val`. Synthetic scenes are not in `splits.json` and are accepted as `domain="synthetic"`.
- No calibration artifact is produced from real data yet (no real runs): TODO: fill from results. Figure 11 (`bench.figures.fig_conformal_coverage`) takes coverage curves and is not in `make_all`.

## CountOnce-Match v3 (model, instances, `assoc: v3`)
- Synthetic episodes carry no box or object size, so `episode_inputs` uses a nominal 0.1-unit object (size and a pinhole box derived from depth). Real runs use the detection box and lifted size.
- Added a geometric pair term to the score (even MLP of the 3D offset, added to the descriptor dot product). Descriptor dot products alone did not overfit 10 episodes in the time budget. The term is even in the offset so view order cannot change scores.
- No learned view-index embedding: the view enters only through camera centre and ray direction, so permuting views leaves the instances unchanged. `match_probs` averages both Sinkhorn orientations for the same reason.
- Features are centred on the detection centroid and divided by `scene_scale` (episodes use the median camera-to-detection distance).
- Cannot-link splitting uses an s-t min-cut (scipy max-flow, probabilities scaled to ints) between two same-view detections, recursively, on above-threshold weights only. Not a global Stoer-Wagner cut.
- Under `assoc: v3` the sensitivity grid's merge strictness `k` maps to the match threshold `1 - 0.5*k/merge_k` (0.5 at the configured `merge_k`), clipped to [0.05, 0.95].
- The CLI gained `--v3-checkpoint`; the missing-checkpoint error is raised at the start of `run()`. No checkpoint is trained or shipped: `train.py` is a later task, so v3 is untested on real data.

## Matcher training, cycle loss, calibration (overnight)
- Cycle loss is the mean squared gap between `P_ij @ P_jk` and `P_ik` on the real-detection blocks (dustbin row/column dropped), over up to `max_triplets` sampled view triplets per episode. Default `cycle_weight: 0` (the "pairwise only" ablation); the "+ cycle loss" arm sets it above 0.
- Count-consistency term: differentiable count `sum_i 1/(1+sum_j P_ij)` built from the forward Sinkhorn blocks, L1 against the true count, normalized by the true count.
- "Association F1" for checkpoint selection is pairwise F1 over cross-view detection pairs at `match_threshold` (0.5) on the val split, not the instance-level F1 of the benchmark.
- Resume is exact because shuffling and triplet sampling derive from `(seed, epoch)`; `last.pt` holds model, optimizer, scheduler and history. `stop_after_epoch` pauses a run without changing the cosine horizon.
- Calibration is fitted on the val split only after the final epoch and written to `<out_dir>/calibration.json`; test-split ECE reporting is left to the evaluation step.
- Temperature scaling recalibrates the symmetric match probability `match_probs`, not Sinkhorn log-scores.
- `countonce synth` writes train/val/test geometry episodes with disjoint seed ranges. No training was run.

## Profiling harness and OOM ladder (overnight)
- Rungs are numbered so `oom_fallback_rung` 0 means no fallback: 0 initial, 1 free cache and retry, 2 chunked reconstruction at half the frame count (min 3), 3 lower resolution (tiers 0.75 then 0.5, each its own attempt, both rung 3). Then `GpuOomError` (`ERR_GPU_OOM`, retryable, with a user tip).
- The `recon` and `detect` stages are both wrapped. Detector rungs: free cache, then smaller detector batches (`chunk_size` frames per `detect` call, cache freed between batches), then `ERR_GPU_OOM`. Detectors have no resolution control, so the detector ladder is built with no resolution tiers (rung 3 does not exist for it).
- Resolution rung (closed): `VGGTMethod.with_resolution_scale(scale)` returns a copy sharing the loaded model with `input_size = scaled_target(scale)` (`round(518 * scale / 14) * 14`, so 0.75 -> 392, 0.5 -> 252 or 266 by rounding, never below 14). `MethodReconstructor`, `ChunkedReconstructor` and `CachedReconstructor` pass it through; a method without it (COLMAP) raises `RungUnavailable` and the rung is skipped. The pipeline rebuilds each `Frame.grid_map` with `backbone_grid_map(w, h, mode, target)` for the lowered tier (`regrid_frames`), so lifting stays exact. `input_size` is in `VGGTMethod.config()`, so it is part of the recon cache key. Non-native widths use `load_images_at`, which mirrors VGGT's crop-mode loader; it is not exercised against real weights yet.
- OOM re-raise (closed): `VGGTMethod.raise_on_oom` (default False) makes `reconstruct` re-raise `torch.cuda.OutOfMemoryError` after freeing memory. The pipeline switches it on via `with_oom_raising()` for every run, so the direct, unladdered call still returns `Recon.failure("cuda_oom")`. The WSL "CUDA ... error" `RuntimeError` path still returns a failure (the CUDA context may be unusable).
- Bug fixed on the way: a ladder chunk (min 3) could be smaller than `chunk_overlap` (default 4), so `ChunkedReconstructor` raised `ValueError` instead of retrying. The pipeline now clamps the overlap to `max(2, min(chunk_overlap, chunk - 1))` for ladder attempts.
- `recon_stage` now also returns the frames it used, and `pipeline.run` continues with those, so a lowered tier's `grid_map` reaches lifting.
- Not verified on a GPU: no real VGGT run, no memory numbers.
- OOM is detected by class name `OutOfMemoryError` or a "out of memory" RuntimeError, so no torch import is needed.
- `configure_allocator()` runs at `countonce.cli` import and in `main()`. It keeps other `PYTORCH_CUDA_ALLOC_CONF` options and respects an explicit `expandable_segments` setting. Python entry points that skip the CLI must call it themselves.
- Profile CSV: one row per (repeat, stage) from the pipeline's `timings`. VRAM is filled only for `recon` (run peak) and `total`, and empty without CUDA. `docs/PROFILE.md` is a skeleton; `torch.profiler`/Nsight traces are not wired.
- Tests ran with `~/vggtProject/.venv` (countonce still has no venv, see GATES.md).

## §9.6 ablation grid
- `cannot_link: false` for v2 only lets *unassigned* same-frame detections join their cheapest admissible existing track (no tracks exist in the first frame, so same-frame pairs there stay separate). v1 skips both cannot-link checks.
- `appearance: false` makes `V2Associator._features` return nothing (cost, gate and split check all ignore features); `sync` inherits it.
- `lift_region: mask` is the old behaviour (mask when present, else box/point). `box` strips masks; mask-only detections keep their mask. Fallback counts are in `result.config["lift_fallbacks"]`, not a warning, to avoid noise on box-only detectors.
- `assoc=v3` is not a one-factor ablation (it would need a checkpoint in the base); it is the `co_v3_vggt` method. Unchecked-in checkpoint path `runs/match/best.pt` is a guess matching `configs/match.yaml` `out_dir`.
- `SkipRun` rows count as "skipped" in `run_jobs`' return tuple and are retried on resume. Ablation lists hold only non-default values. YAML `off` is quoted (bare off parses as False).

## Synthetic association bench and ablation runner
- `bench.csv` schema gained append-only columns `domain`, `duplicate_rate`, `miss_rate`, `count_error` (older CSVs are upgraded on resume). `bench --synth-assoc [CONFIG]` writes to its own `results/bench_synth_assoc.csv` by default so synthetic rows do not enter the real-scene MAPE figures; `fig_assoc_f1` reads any such CSV and `domain` tells the rows apart.
- Geometry-only episodes have no images, so v1/v2/sync/v3 run on a stand-in `Recon` (64 px grid, NaN depth so the occlusion test never fires, `scene_scale` equal to the one `episode_inputs` uses). The box half-extent is half of `episode_inputs`'s so that v3's feature builder reproduces the trainer's inputs (tested). Detection scoring ignores false positives in pairwise F1 (no GT id), but they count in the instance count.
- v3 rows are `skipped_no_checkpoint` until a checkpoint exists. `V3Associator` gained `synchronize` (spectral sync of the match matrix instead of threshold + components) and `feature_group`.
- Feature groups: `3d` zeroes box, score, lift confidence, rays and camera centre (log-depth then degrades to distance from the scene centre); `3d_rays` keeps rays and camera centre; `full` keeps everything. The appearance slot is zero-masked when the model has one and the episodes carry none. Appearance-only and VGGT-token groups are not run (synthetic episodes have neither).
- Ablation: each distinct (cycle loss, group, seed) is trained once and reused (`arm:pairwise` and `group:full` share a model, `arm:sync` reuses it with synchronization). Summary: mean ± std over seeds; CI is a percentile bootstrap over test episodes of the per-episode seed mean. At least 3 distinct seeds are required. `countonce ablate --config configs/synth_ablation.yaml`.
- Smoke-tested on tiny episodes in tmp with 1 epoch; no real run, no numbers committed.

## Media sanitization + typed errors
- `GpuOomError` now lives in `countonce/errors.py` (subclass of `CountOnceError`); `countonce.oom` re-exports it plus `ERR_GPU_OOM`/`USER_TIP`, so old imports work.
- Added `ERR_UNSUPPORTED_CODEC`, `ERR_WORKER_TIMEOUT`, `ERR_STORAGE_TRANSIENT`, `ERR_NO_OVERLAP` (PLAN 14.4) beyond the task's list. Retry policy defaults: 4 total attempts, 2 s base doubling, 60 s cap (not specified in PLAN).
- Limits in `MediaLimits` (50 MB image, 100 MP, 4 GB video, 8K, 120 fps, 600 s, 36k frames, codec allow-list) are my defaults; tune later.
- Images are re-saved to `<out_dir>/sanitized/<n>/` (JPEG q95, re-encoded) and the original is never modified. `collect_paths` now raises typed `MediaError` for unsupported files (was `FileNotFoundError`); missing paths still raise `FileNotFoundError`. Video frames are extracted with `-map_metadata -1`. Provenance: `config["exif_stripped"]` and `config["sanitize"]`.
- Face/plate blurring and resumable uploads (rest of PLAN 14.8) are not part of this task.

## Frontier VLM baseline harness (no paid calls)
- `kind: vlm` methods (`config: {provider, model, prompt: count|enumerate, version}`) live in `bench/vlm/`. They run only with `--allow-paid` AND `--spend-cap-usd N` (>0) AND the provider key env var (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`); otherwise `status=skipped_paid_disabled` and no client is built. The cap is per bench process on *estimated* cost: a call is skipped (`skipped_cost_cap`, latched) if spent + projected worst case would exceed it.
- Prices and per-request image caps in `bench/vlm/client.py:PROVIDERS` are my conservative planning numbers, not quotes (verify before a paid run; override per method with `price_in`/`price_out` USD per Mtok). Image caps: anthropic 100, openai 500, google 3000.
- The Anthropic/OpenAI/Google adapters are untested against live APIs (no keys, by design). Model ids are not hard-coded; the bench config has commented examples with `MODEL_ID`. The open-weight (Qwen-VL) baseline is not part of this task.
- Refusals (no integer plus refusal wording) are logged in `results/vlm_calls.csv` and recorded as `status=refused`; non-`ok` rows are retried on resume, so a rerun would pay again for refused jobs. Unparseable replies are `status=error`.
- Per-call log is a separate CSV (`results/vlm_calls.csv`) since its columns (tokens, cost, prompt version) do not fit `bench.csv`. `order: [ordered, shuffled]` doubles VLM calls; set `order: [ordered]` to halve cost.
- Figure 10 (`python -m bench.vlm.fig10`) groups by the scene's `overlap` key in `gt.json` (copied into the job config only when present, so existing job keys are unchanged) and falls back to number of views.

## Schema fixtures and pre-commit (overnight)
- Fixtures are stored per version as `tests/fixtures/schemas/v1.0/{result,markers}.json`. Provenance (git commit, timestamp, torch, gpu) is scrubbed to constants and `out_dir`/`cache_dir` made relative.
- Result documents carry no pixel boxes, so the render test synthesizes a box per stored detection id on tiny generated frames; it checks overlays draw, not geometry.
- `ruff format` was applied to 10 pre-existing files (formatting only) so the `ruff-format --check` hook passes repo-wide.
- The mypy hook is `language: system`; mypy is not installed in the CPU venv used overnight, so it was NOT run here (TODO: run `pre-commit run --all-files` with dev extras).
- `pre-commit install` was not run.

## Dynamic-object residual filter (overnight)
- `motion_filter` defaults to **off** so shipped behaviour and stored results are unchanged; the `motion_filter=True` ablation row is in `configs/bench.yaml`.
- "Downweight in v1/v2/v3 association" is implemented as `MotionAwareAssociator`, a wrapper over any associator, not edits inside each one. Instances made only of flagged detections get `conf * motion_conf_factor`; flagged instances whose disagreeing-view nearest detection is a flagged detection of another instance are merged as one moving object (cannot-link on frames respected). Without that merge, downweighting alone could not fix the ghost count.
- Residual = pixel distance from the projected `xyz` to the nearest detection in the other frame, over box size (floored at 2% of `scene_scale` projected). Views with no detections at all are skipped. A static object the detector missed in a view looks like disagreement; `motion_min_agree=0.5` and `motion_min_views=2` tolerate that. All thresholds are untuned defaults; no benchmark numbers exist.
- `bench.metrics.ghost_duplicate_rate` takes the moving GT ids explicitly; it is not yet wired into `bench.csv` (no motion captures exist).

## Data-backed warning thresholds (bench/tune_warnings.py)
- Tunable from `bench.csv` alone: `min_frames` (n_views), `max_range_width_fraction` ((high-low)/pred), and `min_multiview_fraction` (only when run_dir result.json is readable). `min_depth_conf`, `max_failed_lift_fraction`, `small_count` are not stored per run and keep defaults; `configs/warnings.yaml` is not committed because no real bench data exists yet.
- Cut choice: among cuts whose slice is flagged by `slice_report` (min 5 scenes in slice, 15 total), the one with the highest lower CI bound on (slice MAE - overall MAE). Rows come from one method (`--tune-method`, default `co_v2_vggt`), `val` split or synthetic domain; test-split scenes are dropped and `split="test"` raises.

## Privacy-mask producer (overnight)
- `privacy_blur` defaults to **off** so bench pixels and stored results are unchanged. When on, `privacy_model_path` (an OpenCV `.xml` cascade or `.onnx` YuNet-style face model) is required; the installed `cv2` has no bundled models and nothing is downloaded. Missing path raises `ValueError`/`FileNotFoundError`. No model file is committed, so the real adapter is untested beyond its missing-file behaviour (tests use a fake detector).
- Blur runs after sanitization and after blurry/dedup/cap filtering (fewer images to process) and writes PNGs to `out_dir/privacy/`; detection and recon then see blurred pixels. The mask is stored at native frame size; `lift.py` already maps it through `Frame.grid_map`. `privacy.mask_to_grid` gives the same mask on the backbone grid (crop/pad aware) for viewers and tests.
- Blur sigma (`privacy_blur_sigma`, 12 px) is an untuned default. Plates are only covered if the supplied model detects them.

## Drift monitoring (countonce/monitor/drift.py)
- `result.json` does not record resolution or mean depth confidence, so they are read from optional `mean_depth_conf` / `config.image_width|image_height` keys, with depth confidence otherwise parsed from the `low_depth_confidence` warning text (only present when that warning fired). Features absent on either side are listed under `skipped`, never scored.
- `lift_failure_rate` = `config.lift_fallbacks` / detections; `detection_density` = detections per frame (the bench CSV `density` column is per megapixel; the two are not mixed unless both sides come from the same source).
- Categorical drift is flagged on PSI; the chi-square statistic is reported but not used for flagging (no p-value; scipy not assumed). Min 2 rows per side, else a ValueError / CLI exit 2.
- Ran with the shared venv `~/vggtProject/.venv` since the repo has none.

## "Explain this match" and schema 1.1
- `result.json` 1.0 had no per-detection geometry, so 1.1 appends `detections` (id, frame, recon view, xyz, size, box-centre uv in recon pixels, optional feature) and `cameras` (K, world->camera extrinsics). `UPGRADERS["1.0"]` bumps the version and adds empty lists (markers only change version). `ResultV11` subclasses `ResultV1`; `tests/fixtures/schemas/v1.1/` is new and `v1.0/` is untouched. Existing tests that asserted `"1.0"` for freshly written files now assert `"1.1"`.
- Reprojection error each way is recomputed from the stored cameras (a's 3D point into b's view vs b's box centre), so any pair can be explained, not only logged ones. For 1.0 documents it is reported as unavailable and the 3D distance falls back to logged evidence if any.
- Feature vectors are stored only with the new `Config.store_features` (default off, embeddings are large); otherwise the appearance cosine is reported as unavailable. Detection ids carry the frame, which equals the recon view in the pipeline.
- `match_prob` is the associator's logged probability; v1 logs a heuristic, only `assoc: v3` logs a learned one (the associator name is in the output).

## One-click report
- "Top-N overlays" are ranked by file size (proxy for annotation density, ties by name), default N=6; no per-overlay score is stored. PDF output is skipped with a message when `weasyprint` is not installed (not added as a dependency).

## Zero-count / distractor suite (PLAN 9.8b)
- `kind: zero_suite` is a `MethodSpec` kind skipped by `build_jobs`; `run_bench` runs it afterwards and writes `zero_suite.md` beside the CSV. Its `config` holds `options` (countonce overrides), `n_synthetic`, `seed`, `n_views`, `scenes_dir`.
- Synthetic scenes are written to `results/zero_suite/scenes`; a distractor's scores go up to 0.45, so at the default 0.3 threshold some are intentional false positives. Negative prompts (CountGD++) are not covered here (no detector support yet).
- `ghost_duplicate_rate` is an append-only column; the default executor leaves it blank (no moving GT ids on real scenes). NaN is written as blank.
- Tests ran via `~/vggtProject/.venv` again (no repo venv).
