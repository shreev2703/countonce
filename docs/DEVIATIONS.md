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
- **Tooling check:** ruff and pytest verified via the read-only the sibling VGGT benchmark repo venv; mypy is not installed there and the system Python lacks venv/pip, so mypy config is unverified. Install with `pip install -e .[dev]` once a venv exists.

## 2026-11-24: Port from vggtProject
- `Recon` from vggt-sparse-recon lives in `countonce/recon/result.py` (the ported, method-level reconstruction result). The §7 data-model `Recon` in `countonce/types.py` is a separate task and may wrap or replace it.
- Memory guard (`profiling.py`) placed in `countonce/recon/profiling.py`; `torch` is imported at module top, so it needs the `recon` extra.
- Plotting: ported only generic helpers (style, `save`, `depth_to_rgb`, `point_cloud_preview`, `vram_vs_frames`). Benchmark-specific plots (`band_lines`, `scene_type_bars`) were left behind; they belong to Track B.
- Env check: Python pin relaxed from `==3.11` to `>=3.11` to match `pyproject.toml`.
- Frame extraction: dropped the unused `EXAMPLES` constant. `countonce/configs/colmap.yaml` copied verbatim.
- Added `recon` optional-dependency extra and `pythonpath = ["."]` for pytest. `ruff`/`pytest` were run from the sibling VGGT benchmark repo venv (3.11, has torch/pycolmap) since countonce has no venv yet.
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
- **Loader layout (superseded, see "Fuji-SfM loader verified" below):** all file names live in `bench/datasets/fuji_sfm.py::_read_layout`.
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
- **Limitation (drift):** chunks are chained sequentially, each aligned to the already-merged result, so scale and rotation errors accumulate with chunk count. Pose-graph stitching is available as an opt-in (`chunk_stitch: pose_graph`, below); the default stays chain; drift has not been measured on real data (`TODO: fill from results`). Only camera poses drive the alignment, so a backbone whose chunk poses are noisy gives a noisy stitch; the world points are transformed rigidly, not re-fused.
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
- **`assoc_f1` is filled for the Fuji Core cells that have complete rows** (the miss-rate cells at k=20 only; see `docs/RESULTS_FUJI.md`). It stays blank for the own-capture `gt.json` scenes, which carry no identity labels, and for Fuji cells where not all 16 tree-sides have a defined F1 (`TODO: fill from results`).
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
- Tests ran with the sibling VGGT benchmark repo venv (countonce still has no venv, see GATES.md).

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
- Ran with the shared venv the sibling VGGT benchmark repo venv since the repo has none.

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
- Tests ran via the sibling VGGT benchmark repo venv again (no repo venv).

## Golden benchmark gate
- The real-scene (GPU) half of `configs/golden.yaml` is empty: no datasets are on disk, so no scenes or hashes were invented. `bench.golden run` without `--cpu-synthetic` refuses unpinned scenes; `freeze` pins them. `configs/golden_gpu_baseline.json` does not exist yet (`TODO: fill from results`), so `golden-gpu.yml` fails until the owner adds it.
- CPU mode writes `results/golden/<sha>-cpu.json` (suffix added to keep it apart from GPU results). Hosted CI compares to `configs/golden_cpu_baseline.json` with latency ignored (`--tol latency_s=off`). Association F1 is only defined for synthetic scenes (real scenes have counts only).
- pytest on this box lacks matplotlib/torch/cv2 so 6 unrelated tests fail and 5 modules fail collection; not caused by this task.

## Correction tools
- `apply_corrections` takes and returns the persisted `ResultV11` (not the in-memory `CountResult` dataclass): corrections arrive after the job finished and only `result.json` exists. `corrections_applied` is stored in `result.config` (the provenance dict). Schema version is unchanged.
- Idempotence uses a content key (job id + action + sorted detection ids); already-recorded keys are skipped. Interval low/high shift by the count change (clamped at 0), method unchanged: not re-calibrated.
- Split with several source instances puts all moved detections in one new instance; splitting an instance's entire membership is refused (`ERR_INVALID_CORRECTION`). Merge of detections already in one instance is a no-op. Cannot-link violations raise `CorrectionError` with `ERR_CANNOT_LINK`.
- Merge hard positives are all cross pairs between the two instances' members; Split hard negatives are moved-vs-remaining members. Pass the ORIGINAL results to `export_training_pairs`. Corrections whose job id has no result are skipped. CLI maps every job id in the file to the one `out_dir` result.
- Tests ran with the sibling VGGT benchmark repo venv (project `.venv` lacks torch/matplotlib).

## Negative prompts (natural-language prompts)
- Suppression runs the negative phrases as a second detector call and matches boxes by IoU >= 0.5 (same frame) rather than adding tokens to one caption, because `Detection` carries only the max-token score. Works for any detector whose scores depend on the caption; opt-in via `supports_negatives` (GroundingDINO only). `Config.negative_margin` default 0.0, untuned. Detectors without the flag keep the original prompt and get a `negative_prompts_ignored` warning. `Config.negative_prompts=False` sends the raw prompt untouched (ablation row).

## Bench evaluator / leaderboard
- GT lives in `bench/gt/<split>.json` (`{"split", "scenes": {id: {"count", "instances"?}}}`) or any path via `--gt`; no GT files are committed. "Frozen" means `bench/splits.json` exists, names the split, and predictions cover exactly its scenes; with no splits file, the leaderboard refuses every row.
- Assignments are `detection id -> label`; F1 is pairwise over detections shared with GT, per-scene mean with bootstrap CI.

## Pose-graph chunk stitching (`countonce/recon/posegraph.py`, PLAN §8.3)
- **Option:** `Config.chunk_stitch: chain | pose_graph`, default `chain` so stored results are unchanged. Passed to `ChunkedReconstructor(stitch=...)`; recorded in `config()` only when not `chain`.
- **Edges:** one pairwise Sim(3) `Z_ij` (chunk j frame -> chunk i frame) for every chunk pair sharing >= 2 frame ids, estimated with the same `align_chunk` (Umeyama on shared camera centres, 2-camera axis-tip trick) as the chain. Contiguous chunk ranges only share frames across non-adjacent chunks when `overlap` exceeds half of `chunk_size`; the solver itself takes arbitrary frame-id sets, so loop closures (revisited frames) are supported there, but the pipeline's contiguous ranges do not produce them from a single pass.
- **Solve:** unknowns `T_i` (chunk -> global), `T_0 = I` as the fixed anchor. Residual per edge is `log(T_j^-1 T_i Z_ij)` split into rotation vector (rad), log scale, and translation divided by the shared cameras' spread in chunk j (dimensionless), weighted by sqrt(shared frame count). Each `T_i` is parameterised as (rotvec, log s, t); `scipy.optimize.least_squares` (TRF, numerical Jacobian) starting from the strongest-edge spanning-tree chain. This is a Euclidean-parameterised Gauss-Newton on the Sim(3) log coordinates, not a Lie-group manifold solver or robust (outlier-rejecting) loss: a bad edge is averaged in, not rejected.
- **Tree fallback:** if edges == chunks - 1 (no redundant edge) the chain result is used; `ChunkedReconstructor` then takes its original sequential path, so output is bit-identical to `chain`. A disconnected graph raises `ChunkAlignmentError`.
- **Merge:** shared frames still keep the earlier chunk's copy; only the Sim(3)s differ. All chunks are reconstructed before stitching in this mode (chain streams them), so peak host memory is higher.
- **Evidence:** CPU synthetic tests only (per-chunk camera-centre noise plus a loop-closing chunk: pose-graph RMS camera error below chain). No real VGGT run and no drift numbers: `TODO: fill from results`.

## MCP tool error contract (overnight)
- `count_objects` returns `{"ok": false, "error": {code, message, tip, retryable}}` instead of raising, since an agent needs a readable result. Added `ERR_INVALID_INPUT` (non-retryable) for empty prompts, missing paths and bad config keys. Typed `CountOnceError`s from the pipeline keep their own code.
- `out_dir` defaults to a fresh temp dir so `result_path` is always valid. The server is FastMCP over stdio; the `mcp>=1.0` pin is unverified because `mcp` is not installed here, so the registration test skips.

## Lazy plotting and torch imports for `.[dev]`-only CI (overnight)
- `countonce.plotting` no longer imports matplotlib at import time: `plt` is a small lazy proxy that imports matplotlib (and selects `Agg`) on first attribute access; `Figure` is a `TYPE_CHECKING` import. `bench/figures.py` and `bench/vlm/fig10.py` import that proxy, so modules that only compute numbers load without matplotlib. This keeps every call site unchanged, versus rewriting each to a local import.
- `countonce/recon/profiling.py` imports torch inside each GPU helper, so the pure `frame_cap` works without torch.
- Tests that really plot, use torch or need cv2 call `pytest.importorskip` inside the test; non-plotting tests in the same files still run. `types-PyYAML` added to the `dev` extra.

## mypy to zero errors (real run)
- `.venv/bin/mypy countonce` now exits 0 (65 source files). Stubs: `scipy-stubs` added to the `dev` extra (`types-PyYAML` was already there). `ignore_missing_imports` overrides extended to `cv2`, `mcp`, `anthropic`, `openai`, `google` (optional runtime deps, not installed in dev); the last three go beyond the task's list because `bench/vlm/client.py` imports them lazily.
- `LadderResult` made `Generic[T]` so `run_with_oom_ladder` returns a typed value; this removed the `has-type` and `object` errors in `pipeline.py` and two `type: ignore`s. No behaviour change, so no new regression test (none of the errors was a real bug).
- Remaining targeted ignores: `nn.Module` base classes in `match/model.py` (torch imported lazily), Pillow `ImageFile` vs `Image` in `ingest/sanitize.py`, scipy `maximum_flow` taking `csr_matrix` in `match/instances.py`.

## Human baseline harness (bench/human_baseline.py)
- Pooled human MAE/within-10% average per-participant errors per scene (not the error of the mean count), then bootstrap over scenes. Per-participant CIs bootstrap that participant's scenes.
- Time ratio = mean human seconds / median (p50) CountOnce runtime over the shared scenes; the CI resamples scenes jointly for both. CountOnce runtime is per-scene mean of `runtime_s` in `bench.csv`.
- Scene manifest JSON (images + GT count) is the harness input; GT is never written into sheets. No real study data exists; tests use synthetic CSVs only.

## Model registry, promotion gate, active learning (overnight)
- **Evaluators injected.** `Registry.promote` takes `frozen_eval`/`golden_eval` callables (checkpoint path -> `bench.evaluate` / `bench.golden` result). The CLI builds the frozen one from `--frozen-gt` plus a dir of `<version>.json` Predictions (produced by running each checkpoint on the frozen split); the golden one runs the CPU synthetic set with the checkpoint as `assoc=v3` (`run_synthetic(cfg, checkpoint=...)`). The default golden path was not exercised with a real checkpoint; tests use fake evaluators.
- **No evaluator is a refusal, not a skip** (`GateConfig.require_frozen=True`), except the very first model, which has no incumbent. `--allow-no-frozen` overrides.
- **Tolerances** (untuned): frozen MAE +5% relative, frozen assoc F1 -0.01 absolute, golden MAPE +0.02 / F1 -0.02; golden VRAM/latency never gate (hardware noise).
- **Active score** = 0.5 x mean(1 - 2|p-0.5|) + 0.5 x min(interval width / count, 1); weights are arbitrary, jobs with `corrected/` are skipped.
- **Corrections in training.** Persisted results hold no rays/camera centres, so correction groups use xyz/size/frame only (other inputs neutral) and a BCE loss on match probability; one group per optimiser step, round-robin, weight `corrections_weight`. `false_positive` pairs are ignored by the matcher. The data version gets a `+corr-<hash>` suffix. Not run on real corrections.

## Mask producer (masks.py)
- Masks are attached to all raw detections (not only those above `score_threshold`) so the sensitivity sweep over thresholds also lifts masks; cost is extra SAM prompts on low-score boxes.
- Mask cache is separate from the detection cache (`<cache_dir>/masks/*.npz`, packed bits), keyed by frame hash, detector@version, mask model, mask version and the exact boxes.
- Ablation row `masks: box_fill` only; `sam` is not in `configs/bench.yaml` because weights are gated.

## Synthetic-data DAG (`synth/dag.py`)
- Stage stamps key on the upstream stages' *output hashes*, the stage's config slice, recipe version and seed. The git sha is recorded in each stamp but excluded from the key, otherwise every commit would rerun everything.
- No renderer exists, so `validate` runs on `proxy_scene` (tiny constant frames, real recipe camera centres); only a degenerate rig can quarantine a proxy scene. A real renderer plugs in via `Dag(scene_loader=...)`.
- `episodes` converts each clean recipe into a geometry episode (seed = recipe seed) with view and instance counts pinned to the recipe; recipe randomization knobs are not used by the geometry generator.
- Dataset version for the DAG is a sha256 content hash of the episode set (train, val), passed to `TrainConfig.data_version`; checkpoints now carry it in `meta` (`checkpoint_meta`). `evaluate` scores `best.pt` on the test episodes and runs the ablation grid only when `ablate:` is non-empty.
- Fixed 11 latent mypy errors in `synth/` and `bench/` that became visible once the CLI imported `synth.dag`.

## Shadow-mode model routing (`countonce/routing.py`)
- Shadow runs synchronously after the live result is computed (before `run_with_shadow` returns) instead of in a background thread: the live result is snapshotted first, so it cannot be affected, and tests stay deterministic. Callers wanting zero added latency can call it from their own worker.
- Bucket = sha256(`shadow name` + job id) mod 10000, so shadows sample independently; the first shadow whose percentage covers the bucket wins. Jobs without a shadow are not logged.
- Stage sharing: both runs use the same `Config.cache_dir` (recon/detection caches are keyed by model identity, so matching stages hit); injected live detector/reconstructor objects are reused when the shadow's stage matches. Shadow writes to `<out_dir>/shadow/<name>` with `export=False`.
- Report: per-shadow mean |delta count|, interval overlap rate, shadow-minus-live wall latency, percentile bootstrap 95% CIs over jobs; failed shadow runs are counted, not averaged.

## CLI end-to-end on `examples/tiny_scene`
- `detector: color_blob` (`countonce/detect/color_blob.py`, HSV threshold + connected components, colour word taken from the prompt) and `backbone: fixture` (`countonce/recon/fixture.py`, serves cameras and depth from `scene.json`) are **test backends only**. They are registered in `Config`, the CLI choices and `build_detector`/`build_reconstructor` so the real command line runs on CPU with no weights, but they are not detection or reconstruction methods and must not appear in benchmark results.
- The task's command form `countonce run examples/tiny_scene ...` needed a positional input, so `run` now accepts an optional positional `source` (directory, image or video) in the same mutually exclusive group as `--images`/`--video`. Flags `--detector color_blob --backbone fixture` are still explicit; the CLI only defaults `backbone_options.scene` to `<input dir>/scene.json` for `fixture`.
- Scene is 128x96, 5 views, 5 red GT spheres plus 1 blue and 1 green distractor; `scene.json` is about 250 KB, PNGs about 4 KB each. The render is deterministic numpy ray casting (`examples/tiny_scene/generate.py`).

## Identity labeling tool (`bench/label/`)
- `result.json` stores detection centres (`uv`, recon-grid pixels), not boxes, so `label.html` draws circles at the centre, not boxes. Grid size is inferred as `2*cx, 2*cy` from the camera principal point and rescaled to the image (about 1 px error). Views come from `sanitized/**` in sorted order (index = frame id); no image on disk gives a blank canvas.
- Instance ids in labels are strings (`i<id>` when seeded from a run, `n<k>` when new). Rejected/unlabelled detections map to `None` in the GT format.
- The embedded JavaScript was not executed in a browser or Node here (neither available); only the Python side is tested. A human should smoke-test `label.html` before labelling.

## Local job queue and worker (`countonce/jobs/`)
- SQLite stand-in for Postgres: the schema and every statement use standard SQL only (`ON CONFLICT`, `UPDATE ... RETURNING`, scalar sub-select, `COALESCE`, `?` placeholders). A Postgres store reuses them by translating `?` to `%s` and setting `store.CLAIM_LOCK = "FOR UPDATE SKIP LOCKED"`; the single-statement claim is atomic on SQLite because writers are serialised. Times are epoch-seconds floats (`DOUBLE PRECISION`), not timestamps; booleans are 0/1 integers. Added columns beyond §14.3: `jobs.heartbeat_at`, `jobs.next_attempt_at` (backoff gate). `users` and `feedback` tables are not created yet (no user system); `user_id` is a plain string.
- One short-lived connection per call (WAL, `BEGIN IMMEDIATE`) rather than a shared connection, so threads need no locking.
- A job whose worker goes silent is requeued once (max 2 attempts in total) and then `failed` with `ERR_WORKER_TIMEOUT` as §14.4 says; reported errors use `retry_decision` (4 attempts, 2 s doubling, 60 s cap) and end in `dead_letter`.
- Unknown exceptions are classed `ERR_NO_RECON` (permanent) so a code bug never burns GPU retries; `ValueError`/`FileNotFoundError`/`NotImplementedError` -> `ERR_INVALID_INPUT`, `TimeoutError` -> `ERR_WORKER_TIMEOUT`, other `OSError` -> `ERR_STORAGE_TRANSIENT`.
- `countonce worker --once` drains every currently claimable job and exits (jobs still in backoff are left queued). Heartbeats run from a background thread every 15 s during a job. No download/upload step: inputs are local paths stored in `job_files.key`; outputs go to `<db dir>/outputs/<job id>`.
- The 'killed worker' in the load test is simulated with a `BaseException` that the worker deliberately does not catch.

## Activation checkpointing (matcher)
- `activation_checkpointing` is a `make_model` argument and `TrainConfig` field but not in the model `hparams`: it changes memory only, so old checkpoints and the registry hash stay valid. The setting is recorded in checkpoint `meta` and the `run` event of `log.jsonl`.
- `attention` wraps each block's attention sub-layer; `all` also wraps the MLP. Both skip checkpointing under `torch.no_grad()` as well as eval mode.
- `countonce train --profile-memory` measures bytes saved for backward with `saved_tensors_hooks` (works on CPU) rather than RSS; GPU numbers stay `TODO: fill from results`.

## Viewpoint-spread sweep axis (bench/sweep.py)
- Camera centres come from GT (`camera_centers`, optional `scene_scale` in `gt.json`), not from a recon, so the axis needs no GPU. Scenes without them record `skipped_no_cameras` for narrow/wide; `all` rows still record `baseline_spread` when centres exist. `scene_scale` falls back to the RMS spread of all the scene's centres.
- Views are ordered by angle in the best-fit plane of the centres; candidate sub-arcs are drawn from that order (pool of 64 plus one even full-arc set), but the terciles are of the measured spread (`bench.slices.centers_spread`), so `spread` and `baseline_spread` always agree. The even sample picks the tercile member with the most even angular spacing.
- `Job.spread` joins the run key only when not `all`, so existing CSV resume keys are unchanged. The repo config now sweeps 3 spreads (3x the grid).

## Pre-flight capture check (countonce/ingest/preflight.py)
- Python reference only. The browser port (Canvas/WASM, PLAN 14.8) waits for mockup approval and is not started; it should reproduce the same three measures (Laplacian variance, 16x16 grayscale thumbnail mean-abs-difference, pairwise grouping) and the same tip wording.
- All `PreflightThresholds` are PROVISIONAL, tuned on nothing. `min_photos` follows tuned `min_frames` from `configs/warnings.yaml` (bench/tune_warnings.py output) when that file exists; it is not committed yet (no real bench data). Others (`min_sharpness_ratio` 0.3 of set median, `max_neighbor_distance` 0.25, `max_isolated_fraction` 0.5, `min_diversity` 0.03) are placeholders; `dup_distance` copies the `dedup_threshold` default (0.02); `min_sharpness` is 0 (off) because Laplacian variance depends on resolution, so blur is judged relative to the set median by default.
- Overlap is estimated from thumbnail distance only (no reconstruction): a photo is isolated when every other photo is further than `max_neighbor_distance`. This is a rough proxy for the post-hoc `low_overlap` warning, which uses instance view counts.
- Overrides live under `preflight:` in the config (`Config.preflight`) or the `--config` YAML of `countonce preflight`. Exit code is 1 when any tip is raised, 2 on unreadable input.

## Retention, share tokens, API keys, rate limits (job store)
- Stdlib only: API keys are `co_<id>_<secret>` with a salted PBKDF2-SHA256 hash (100k iterations); lookup by the embedded id, constant-time compare. Plaintext is returned once by `create_api_key`.
- Rate limit is a persisted token bucket (`rate_buckets`), off unless `rate_limit=N` per `rate_window_s` is passed to `SqliteJobStore`; checked inside the `submit` transaction, so a rejected submit leaves no row and an idempotent duplicate costs no token. Raises `RateLimitError` (not in the `ErrorCode` taxonomy yet).
- `delete_job` refuses running jobs (`ValueError`); it deletes inputs and outputs (`<output_root>/<job_id>`), result rows and revokes share tokens, keeping the job row as `deleted`. `purge_expired` ages by `finished_at` (fallback `created_at`) and skips queued/running jobs.
- `training_opt_in` lives in a new `users` table; `export_training_pairs(..., opted_in_users=)` is an extra gate on top of per-correction `consent` (None keeps the old behaviour).

## Cost per job (admin metrics)
- Rates live in a new `countonce/configs/cost.yaml` (`cost_per_hour.laptop|cloud`, both null; `admin-stats --cost`), not in the pipeline `Config`, since they are ops settings. No default rate; null rate gives null cost, shown as `TODO: set rate` in HTML.
- GPU-seconds per job = sum of all stored stage `timings` (the store does not mark which stages are GPU). Worker kind comes from `workers.kind` via `jobs.worker_id`; jobs from unknown workers group as `unknown` with no rate. Frames = `n_frames` or `len(per_frame_counts)` in `result_json`; cost per 1,000 frames covers only jobs with a frame count.
- `cost_per_job` alert: mean cost of done jobs finished in `window_s` > `max_cost`; disabled while `max_cost` is null (shipped state).

## Stage tracing (overnight)
- Tracing is off by default. A worker traces via `countonce worker --tracing jsonl|otlp [--trace-dir]`; `pipeline.run` also self-traces when `Config.tracing: {exporter, dir}` is set and no trace is active (trace id = out_dir name).
- Stage spans reuse the pipeline's `timings` readings via `tracing.record_span` (one clock). A failing stage gets an error span from the end of the last finished stage.
- With `exporter: otlp` and no `opentelemetry` installed, a warning is logged and nothing is exported (no jsonl fallback). The OTLP path is untested against the real package; it uses `OTLPSpanExporter` (http/proto) with default env config. `opentelemetry.*` added to the mypy missing-imports override.
- OTLP spans get fresh OTel ids (our ids go in the JSONL only).

## Public API and packaged configs
- `colmap.yaml`, `alerts.yaml`, `cost.yaml` moved from `configs/` to `countonce/configs/` (package data). `countonce.resources.resolve_config` checks the CWD path first, then the packaged copy, then raises a clear error. Benchmark/train/synth/dag/profile/match configs stay in the repo `configs/` (their packages are not meant for the wheel); the CLI exits 2 with a message when they are missing outside a checkout.
- Top-level `countonce.load_result` is `explain.load_result_dir` (reads `<out_dir>/result.json`).
- The wheel test builds with `--no-build-isolation` and inspects the wheel contents rather than installing into a venv (dependencies cannot install offline).

## Retrain and promotion loop (`countonce retrain`)
- The gate is the existing `Registry.promote` (point estimates with tolerances). The CIs from `bench.evaluate` are recorded in `results/retrain_log.csv` (`*_lo`, `*_hi`) but do not drive the decision; golden metrics are scene means without CIs. A CI-aware gate is a later change.
- The gate runs on `gate_split` (default `val`); `test` is scored and logged only, so promotion never selects on test.
- Corrections count as used only with the per-correction `consent` flag AND the user in `jobs_db` `training_opt_in`; with no `jobs_db` nobody is opted in. Rejects are exported and counted (`n_false_positive`) but the pairwise matcher loss does not use them. `n_corrections` counts eligible corrections, not pairs.
- The default frozen evaluator reads `<predictions>/<version>.json`; nothing here generates those predictions for a new candidate, and `bench/splits.json` is not frozen yet. Until both exist, `countonce retrain` exits 2 with a clear message (`frozen:` missing or predictions file not found). Tests inject evaluators.
- `TrainConfig.init_checkpoint` fine-tunes from a checkpoint; its hparams (including `appearance_dim`) override `model:`.
- Log columns are results only; `configs/retrain.yaml` carries no numbers. Real before/after values: TODO: fill from results.

## Repeat-job cache profiling (PLAN 13 item 6b)
- `countonce profile --variant cache` is selected by a flag plus a `cache:` block in the profile YAML (`new_prompt`, scratch `dir`), not as an entry under `variants:` (those write the separate variants CSV and compare `Config` overrides). Rows go to the existing `out_csv` with three appended columns.
- Detection cache hit is inferred as "no new file written under `<cache>/detect`" because `CountResult` has no detection-hit field (adding one would change the result schema). A miss always writes, so the inference is exact for this harness. Recon hit comes from `CountResult.recon_cache_hit`.
- Each timed repeat clears the scratch cache first; `warmup` passes run full cold/new/repeat sequences in a separate cache dir that is deleted afterwards. "GPU-time savings" in the report is detect + recon wall time (the stages that use the GPU), not a device-timer measurement. No numbers exist until the GPU run: the table shows `TODO: fill from results`.

## Resumable chunked uploads (`countonce upload`, PLAN 14.8)
- R2 multipart and the TUS protocol wait for the web API. Chunks are stored on local disk under `<db dir>/uploads/sessions/<id>`; the `UploadTransport` protocol (`create_session`, `put_chunk`, `status`, `finalize`) is the seam an HTTP layer will implement, with R2 multipart or TUS replacing the chunk store there. `LocalTransport` is the only transport now.
- `put_chunk` takes an optional `sha256` argument (the spec signature had none, and a server cannot verify a chunk against its own hash). With it, a mismatch raises `ERR_CHUNK_CORRUPT`; without it only the length is checked, and the whole-file hash at finalize is the backstop. The client always sends it.
- Two new error codes: `ERR_CHUNK_CORRUPT` (retryable, resend the chunk) and `ERR_UPLOAD_INCOMPLETE`. A wrong whole-file hash reuses `ERR_CORRUPT_MEDIA`; size and type failures reuse `ERR_FILE_TOO_LARGE` / `ERR_UNSUPPORTED_MEDIA`; unknown, expired or non-open sessions and bad indices use `ERR_INVALID_INPUT`.
- Limits live in `countonce/configs/uploads.yaml` (packaged, CWD copy wins): `max_bytes`, chunk size bounds, `max_chunks`, `session_ttl_s`. The per-kind `MediaLimits` caps apply too (the smaller wins). They are not fields on the pipeline `Config`, to keep run provenance unchanged.
- Session expiry is idle time: each stored chunk renews `expires_at`. `purge_expired` also purges expired sessions (chunk files and rows) and creates the tables on demand, so an old database upgrades in place. A finalized session's sanitized file lives under `<root>/files/<id>` and goes with its job's retention, not with session purge.
- Finalize without `job_id` queues a new job and needs a prompt (`idempotency_key=upload:<session>`); with `job_id` it attaches to that user's queued job. A wrong whole-file hash or a media rejection marks the session `failed` and discards its chunks (client restarts). Images are re-written by `sanitize_image` (the stored file is the sanitized copy, so its sha256 differs from the upload's); videos are validated with ffprobe and stored unchanged.
- One file per session; a multi-photo job takes one session per photo via `--job-id`.

## 2026-11-24: Queue load-test harness

- **Plan said:** PLAN 16B.2 load test with k6/Locust-style tooling and a fake worker. **Done:** no HTTP API
  exists, so `countonce/loadtest.py` (`countonce loadtest`, `scripts/load_test.py`) drives `SqliteJobStore`
  directly with stdlib `multiprocessing` (spawn): N submitter and M fake-worker processes, sleep-based stages,
  injected reported failures and `os._exit` crashes recovered through heartbeat timeout. Crashed workers are
  respawned; crashes are capped by `max_crashes`.
- **Lock retries:** the stock store waits inside sqlite's busy timeout, which can't be counted. The harness
  uses a `CountingStore` subclass that connects with `timeout=0`, counts each `SQLITE_BUSY`, and retries with
  jitter. Its contention profile therefore differs a little from the production store's.
- **Reconciliation:** `reconciled` requires submitted = done + dead_letter + failed (`failed` is the
  stale-heartbeat exhaustion state), no duplicate completion events and one result row per done job.
- **Numbers:** throughput, latency percentiles and error rates are `TODO: fill from results` until a run on the
  target machine appends to `results/load_test.csv` (gitignored). Tests carry the `loadtest` marker.

## Pilot report (`countonce pilot-report`)
- The job store has no corrections table (corrections live in `corrections.json` files), so corrections metrics take an optional `--corrections` file; without it those cells stay `TODO: fill from results`.
- "Instances" for corrections-per-instance is `results.count` summed over finished jobs. A "visit" is a distinct `captured_at` (fallback `created_at`) per `jobs.site_id`; `sites.json` is not read.
- Bootstrap CIs (95% percentile, seeded, 1000 resamples) appear only for n >= 5; otherwise `n/a (n<5)`.

## Track J data stack (shards, Parquet features, DuckDB catalog)
- DAG stage order is now `recipe, validate, shard, episodes, features, catalog, train, evaluate`. `shard` sits after `validate` (packs clean scenes via the scene loader); `features` and `catalog` need the episodes, so they follow `episodes`. The DAG stages need the `data` extra and raise `DagError` without it; `tests/test_synth_dag.py` uses `pytest.importorskip` and CI's `full-cpu` installs `.[data]` (the skip checker does not allow pyarrow/duckdb skips).
- Shards are plain tar via `tarfile` (deterministic headers and npz zip timestamps). `max_bytes` caps the payload; tar pads the file up to a 10 KiB record.
- The generator has no per-object size or box, so feature rows use the nominal object size and a nominal pixel box; `embedding` is null until an embedder is supplied. `occlusion` in the catalog is a recipe proxy (occluders per object).
- The catalog is a Parquet file queried through in-memory DuckDB views (`scenes`, `detections`), not a `.duckdb` file, so DAG output hashing stays byte-stable.
- "Feature-store data version" = the episode-set content hash stored in the Parquet schema metadata. `TrainConfig.feature_store` makes `train` put it in checkpoint meta and the run log; DAG lineage and eval records carry it.
- `pyarrow.*`/`duckdb.*` added to the mypy ignore list (no stubs).

## OLTP -> OLAP export (`countonce etl export`)
- Export is sqlite -> local Parquet directory (`<out>/<table>/dt=YYYY-MM-DD/part-NNNNN.parquet`); R2 upload is left to the Postgres/web stage. `dt` is the UTC day of the event time (job finish; feedback/correction creation).
- Corrections are not in the store, so they come from `--corrections corrections.json` (as in `pilot-report`). Cost rows store GPU-seconds, frames and the rate; `cost` is null while the rate is unset (never 0).
- Watermark (`_watermark.json`) is per table: newest exported timestamp plus the ids at that instant. Rows finishing with a timestamp older than the watermark after an export are not picked up (job `finished_at` is set at completion, so this is not expected).
- The Parquet path is opt-in: `admin-stats --export DIR` and `drift` given an export dir (via `ParquetStore` / `load_drift_rows`); sqlite stays the default. Queued/running jobs are not exported, so queue depth and oldest-queued age read 0 / n/a there. `jobs.result_json` keeps only timings and frame count; drift features are precomputed into `drift_json`.

## HTTP API (`countonce serve-api`, `countonce/api/app.py`)
- **sqlite instead of Postgres:** the API wraps `SqliteJobStore`; the `JobStore` seam is unchanged, so a Postgres store can replace it later. Added a `draft` job status (not claimable) so `POST /jobs` can precede the upload, and `create_draft` / `queue_draft` / `add_feedback` on the store.
- **Local disk instead of R2 signed URLs:** `POST /jobs` returns the `/uploads` chunk endpoints, not signed URLs; chunks go through `UploadService` (the resumable-upload backend) and results are served from `GET /jobs/{id}/files/{file_id}`. R2 multipart/TUS is still open.
- **API keys instead of Google sign-in:** `Authorization: Bearer` or `X-API-Key`, keys from `countonce jobs apikey create`. Jobs of other users answer 404, not 403.
- **Extra routes beyond PLAN §14.2:** `POST /jobs/{id}/share`, `DELETE /share/{token}` (a token had no way to be created over HTTP), `GET /jobs/{id}/files/{file_id}`, and the four `/uploads` routes.
- **Idempotency:** `POST /jobs` uses the store's `(user, idempotency_key)` unique constraint (replay returns 200 with the original id). `POST /jobs/{id}/submit` is a `draft -> queued` state transition, so a retry never queues twice; the header is echoed but no response is replayed from a stored copy.
- **Rate limits:** the store's submit token bucket maps to 429 on `/submit`; an optional in-memory per-user request limiter (`--request-rate-limit`) covers all authenticated routes and resets on restart.
- **SSE** polls the store (0.5 s default) rather than being pushed; one `progress` event per state change, then a `done` event on any terminal status (done, failed, dead_letter).
- No load or latency numbers measured. `docs/API.md` is generated by `countonce serve-api --write-docs docs/API.md`.

## Video / multi-view dataset loaders
- Loaders return a new `BenchScene` (bench/datasets/common.py) rather than a Fuji-style per-dataset class, since the sweep needs one count per scene. `to_sweep_scenes` symlinks frames into `<csv dir>/datasets/<name>/<id>/images` plus `gt.json`; `bench.evaluate` takes `to_gt_doc` output. `bench/evaluate.py` itself is unchanged (it scores predictions, not datasets).
- RGBD-VideoCount: one scene per (clip, category). Only the top-level layout is documented (HF README); sub-folders and JSON schemas are assumed. VideoCount/OneScene layouts are fully assumed; their download scripts refuse to run until `DATA_URLS` is filled.

## 2026-11-24: Cloud-worker image and cold-start harness
- Dockerfile never built (no Docker overnight). Weights are mounted, not baked; baking means deriving an image. Base `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`, torch from the cu128 index.
- `safetensors` is not installed in `.venv`, so the mmap test skips here; it is imported lazily. TODO: add to a `serve` extra when the image is first built.
- Cost per 1,000 frames counts one warm inference as one frame; rates in `cost.yaml` are still null, so the report shows `TODO: set rate`.
- The default `countonce coldstart` warm call only touches loaded weights; real pipeline inference needs injected loaders (`run_coldstart`). No numbers recorded.

## HF release scaffolding (overnight)
- `countonce release synth-dataset` repacks a subset of existing DAG shards (`--shards`, default `data/dag/shard`, `--n-scenes`, `--split`) instead of rendering new scenes; run `countonce dag run` first.
- License is a flag, default `other` (the datasheet says `TODO: decide`); the card marks it as undecided. Model card likewise.
- Model-card metric rows fill only from `--results` JSON (or registry `metrics`); CI column stays `TODO: fill from results` unless a `<key>_ci` entry exists. Space `requirements.txt` leaves the countonce pin as TODO. Nothing uploads; `sdk_version` in the Space front-matter is unverified.

## Paper and project-page scaffolding
- `paper/refs.bib` entries carry titles/IDs only from `docs/RELATED_WORK.md` and `docs/MODEL_SURVEY.md`; authors are `TODO: verify authors` because no paper was read in full. Several titles are the short names used in those docs and are tagged `TODO: verify title`.
- `paper/tables/` and `site/` are generated and gitignored; `make paper` skips the PDF when `latexmk` is absent. Ablation rows are methods whose names contain `+` or `-` (the sweep's `<base>+<field>=<value>` naming).
- The site inlines images as base64 (one file, no network); the Links section is a placeholder until URLs exist.

## Lint gate and test-env hygiene (twelfth audit)
- Orchestrator files excluded via `[tool.ruff] extend-exclude`; `logs/` gitignored.
- `safetensors` added to the `test` extra (CI `full-cpu` installs `.[dev,test,...]`, so it gets it).
- `StarletteDeprecationWarning` fixed by adding `httpx2` to the `test` extra (Starlette's supported TestClient backend) instead of a `filterwarnings` entry.
- Another process committed these edits together with the orchestrator files in `5be3433`; history was not rewritten.

## FSC-147 detector sanity harness
- `countonce bench` gained an optional positional `detector-sanity` (flags `--dataset/--root/--split/--limit/--detector/--score-threshold/--sanity-out`) rather than a new top-level command.
- The CSV holds per-image rows plus one `summary` row per run in a single fixed schema; the bootstrap resamples images (each FSC-147 image is its own scene). Default score threshold 0.3 and no exemplars passed to the detector (`use_exemplars=False`); both are my choices.
- Whole FSC-147 layout is assumed (`TODO: verify`); download script uses `huggingface-cli` with `FSC147_REPO` default `isentropic/FSC147` and was not run. `docs/DATASETS.md` results stay `TODO: fill from results`.

## Data-quality gates for real-data loaders
- `quarantine.jsonl` (one `{scene_id, reasons, details, dataset}` object per line) is the task-specified file; synth writes `quarantine.json`. The record shape (`scene_id` -> sorted reasons) matches `synth.validate.QuarantineReport`; the two files were not unified.
- Header-only decode cannot see truncation, so `image_size` also checks the JPEG EOI / PNG IEND tail marker. Truncation that keeps the tail marker is not caught.
- "GT count vs annotation" and "image count vs camera/annotation" rules fire only when a loader supplies `extra["n_gt_items"]`, `n_cameras` or `n_annotated_frames`; no loader sets them yet because the real layouts are unverified (TODO: set after first download).
- Fuji-SfM is a single scene, so `load_fuji_sfm(validate=True)` raises `DataQualityError` (after writing the quarantine file) instead of skipping.
- Evaluator: quarantined ids come from `--quarantine <jsonl>` (repeatable) or a `quarantined` list in the GT doc; they are dropped from GT, predictions and reference, and reported as `n_quarantined_skipped`. `frozen_problems` still lists them as unscored, deliberately.
- Existing loader tests now write real tiny JPEGs, since fake `b"x"` frames correctly fail the gates.

## Postgres schema and job store
- **One SQL body, two stores.** `PostgresJobStore` subclasses `SqliteJobStore` and reuses its statements (`?` -> `%s`, claim with `FOR UPDATE SKIP LOCKED` via `build_claim_sql`). It was run against a real Postgres (the pgserver wheel, installed only locally, not in any extra): all 18 contract cases plus the CLI case pass, including 4-thread claim exclusivity and the 20-jobs/2-workers/one-killed load test. CI has no Postgres service, so those cases skip there (`COUNTONCE_PG_URL not set`, on the skip allow-list); set the variable to a scratch database to run them (tables are truncated).
- **Timestamps are `DOUBLE PRECISION` epoch seconds, not `timestamptz`**, so the injected `clock` and the shared SQL work unchanged. Plan §14.3 does not name a type.
- **Things §14.3 says that SQLite cannot express (or the store does not enforce):** real `boolean` columns (SQLite stores 0/1; the store converts through `_flag`); `bigint` byte counts (`job_files.bytes`, upload sizes; SQLite integers are already 64-bit); `json` columns for `corrections.detection_ids/instance_ids` and `shadow_results.timings_json` (SQLite stores text); `FOR UPDATE SKIP LOCKED` (SQLite serialises writers with `BEGIN IMMEDIATE` instead); foreign keys (SQLite leaves them off, so the sqlite store never enforces them); `ALTER TABLE` for later migrations (SQLite needs batch mode; the initial revision only creates and drops tables, so it runs on both).
- **No foreign keys from `jobs.user_id`, `feedback.user_id`, `jobs.model_routing_id`, `jobs.site_id`.** API-key users and routing/site ids reach the store before any `users`/`model_routing`/`sites` row exists, and the shared contract suite submits jobs for bare user ids. Revisit when sign-in lands.
- **`status` has no CHECK constraint** because the API also uses `draft` (not in §14.3's list); `JOB_STATUSES` in `models.py` lists the values.
- **Columns beyond §14.3**, declared in `EXTRA_COLUMNS`/`EXTRA_TABLES`: `jobs.heartbeat_at`, `jobs.next_attempt_at`, `api_keys.salt`, tables `rate_buckets`, `upload_sessions`, `upload_chunks`. The sqlite store creates only a subset of the Moon-tier columns (`users` has no `email`/`name`/`deleted_at`); a test checks it is a subset of the models.
- **Migration is autogenerated then hand-checked** (revision `0001`); a test asserts `compare_metadata` is empty and that `upgrade head` -> `downgrade base` works on SQLite. The Postgres store runs `upgrade head` on start (`migrate=True`).
- **`--store postgres`** reads `--pg-url` or `COUNTONCE_PG_URL`; `--db` is then only the anchor for local `outputs/` and `uploads/`. `submit`, `upload` and the admin commands stay sqlite-only. A new connection per call (no pool) matches the sqlite store; a pool is later work.
- `uploads.py` DDL widened `INTEGER` to `BIGINT` for byte columns (Postgres `integer` would overflow at 2 GiB). `uv.lock` was not regenerated.

## Object storage (`countonce/storage/`)
- **`list_prefix` returns `BlobInfo(key, size, modified)`, not bare keys**, because the cache TTL (§13 item 6b) and retention need the modified time and a separate `stat` call would be one more round trip per blob on S3.
- **`LocalBlobStore` also accepts an absolute path inside its root as a key**, and `job_files.key` / `upload_sessions.dir` keep storing absolute paths on the local backend, so existing databases and tests keep working. On S3 they store the blob key. A legacy path outside the root is deleted as a plain path.
- **Local signed URLs are `<base_url>/<key>?method&expires&sig` (HMAC-SHA256 over method, key, expiry).** `verify_url` is the check an HTTP route would call; no route serves them yet. With no `COUNTONCE_STORAGE_SECRET` the secret is random per process, so unsigned-then-restarted URLs fail closed.
- **`COUNTONCE_STORAGE=local` (default) leaves the worker and store on plain paths (`blobs=None`).** Only `UploadService` always goes through a `BlobStore` (a `LocalBlobStore` at its root by default). `--storage s3` or the env var switches the worker, the store (retention purge of inputs, outputs, `outputs/<id>/` and expired `sessions/`), the API and the caches to the bucket.
- **Config is env plus the `--storage` flag on `worker`/`serve-api`**; there is no `configs/storage.yaml` because every setting is deployment-specific. The caches share the bucket under `cache/recon/` and `cache/detect/` (`storage.factory.cache_kwargs`); `Config.cache_dir` still names the local cache directory and is unused under s3.
- **Worker uploads whatever the runner wrote under `<out>/<job id>` to `outputs/<job id>/`** and records each in `job_files` (`record_file`); with s3 `GET /jobs/{id}/files/{fid}` 307-redirects to a presigned URL instead of streaming. Inputs are fetched into a per-job temp dir that is removed afterwards.
- **`ReconCache`/`DetectionCache` TTL is an explicit `expire()` sweep** (`ttl_s=`), not a read-time check; nothing schedules it yet. `MaskCache` (`detect/masks.py`) is still disk-only.
- **Upload assembly still buffers the whole file in memory/temp disk** (chunks are fetched one by one into a temp file; the sanitized result is `put` in one call). S3 multipart/presigned-PUT upload straight from the browser (§14.8) is the later step; `presign_put` is ready for it.
- Verified only against `moto`. Real R2 behaviour (endpoint, `region=auto`, SigV4 presign) is untested: needs credentials and a bucket (human gate, see `docs/GATES.md`).

## DVC scaffold
- Added `--only STAGE` to `countonce dag run` / `Dag.run` so each DVC stage maps to one DAG stage.
- DVC stages run with `--config params.yaml` (generated from configs/dag.yaml + defaults) instead of configs/dag.yaml, so DVC param tracking matches what actually runs.
- `evaluate` tracks the whole `train` param section (DVC fails on missing dotted keys such as `train.match_threshold`).
- Stamp JSONs are DVC outs with `cache: false`; `episodes` is included in `dvc.yaml` as it is a real DAG stage.
- Version string is `dvc:<sha256[:12] of dvc.lock>`, looked up in the CWD; fallback is the episode content hash.

## Crop dataset loaders (WGISD, MinneApple, AppleGrowthVision)
- New `CropImage` type (`bench/datasets/crop_common.py`) instead of reusing `FscImage`: these sets have boxes/masks and no exemplar boxes or class file. It is duck-compatible, so `bench/detector_sanity.py` (now `load_images(dataset, ...)`, `--dataset wgisd|minneapple|applegrowthvision`) and `validate_scene` take both. Validation ids are now `<dataset>:<name>`; the OOB rule checks `boxes_xyxy` when present.
- MinneApple count = number of instances in the mask (images without a mask, e.g. hidden test labels, are skipped). AppleGrowthVision exposes two views: annotated images (sanity, sparse boxes so counts are lower bounds) and the 10 counted trees as `BenchScene`s (`load_dataset("applegrowthvision")`, opt-in `datasets:` entry, `enabled: false`).
- Every layout is assumed (`TODO: verify`); licenses per loader are in module docstrings. Download scripts not run.

## Detector adaptation harness (`countonce/detect/finetune.py`)
- **Detector contract for training is a torch density module** (`(B,3,H,W) -> (B,1,H,W)`, count = sum, MSE on Gaussian density maps), not the `Detector` protocol, because training needs gradients. `ModuleDetector` wraps such a module back into a `Detector` (density peaks -> detections). CountGD's real interface is unverified; an adapter module is needed when its weights arrive.
- **Adapters register in a separate `Registry` root** (`models/detector_adapters`) reusing `countonce.match.registry.Registry`; the base-checkpoint hash is stored in `config.base_checkpoint_sha256` (the registry has no dedicated field) and the dataset version is `finetune-splits@<git sha>` from the CLI.
- **Fuji-SfM is scored end to end only**: its GT is 3D instances, so there is no per-image 2D count. The unseen per-image cell stays `TODO: fill from results` unless a 2D-labelled unseen set is configured. The end-to-end scorer is a callable passed to `run_adaptation(e2e=...)`; the CLI does not wire it yet (needs real weights and data).
- `countonce finetune-detector` requires `--module PKG.MOD:FACTORY`; without it, it exits 2 with the weights-gated message. Config `target_layers` globs for CountGD are `TODO: verify`.

## full-cpu: dvc extra, Postgres service, workflow test
- `full-cpu` now installs `.[dev,test,data,api,db,dvc]` and runs a `postgres:16` service (health-checked, `COUNTONCE_PG_URL=postgresql://postgres:postgres@localhost:5432/countonce_test`). The skip allow-list is unchanged.
- `tests/test_ci_workflow.py` keeps a hand-maintained module -> extra map; a new `importorskip` module not in the map fails the test on purpose.
- pgserver (embedded Postgres, unix socket) ran the 49 contract + schema cases once with no failures, so no `PostgresJobStore` changes were needed. It was uninstalled afterwards.

## VLM count parser hardening (`bench/vlm/parse.py`)
- Dropped before parsing: numbers followed by `%`/`percent`, whole lines starting with `confidence`/`certainty`/`probability`, and image indices (`image 3`, `photo 2`, `view 4`, `images 1-3`).
- **Range rule:** `a-b` / `a to b` (a < b, comma thousands allowed) is read as the midpoint rounded half up, so "about 40-45" -> 43. Chosen over first/last value because it minimises worst-case error for a hedged estimate. An explicit `ANSWER:`/`total` marker with a range gets the same treatment.
- **Zero phrases** (`none`, `zero`, `no <noun>`, `not any`, `aren't any`) give `0` with `method="zero_phrase"` only after every numeric path fails, only when the reply is not a refusal, and only when no number word appears in the text (so "Three apples. None are red." stays 3). `zero` was removed from the number-word path.
- Known limits: "photo 1 and 8" keeps 8 (only `-` joins image indices); a confidence line that also holds the count is dropped.

## Trace propagation API -> worker (overnight)
- Added a `memory` tracing exporter (`tracing.MEMORY`) as the in-memory exporter for tests; no OTel SDK needed.
- The request trace is exported under its own id (`req-<hex>.jsonl` with `jsonl`), so `countonce trace show <job_id>` shows the worker half only; both halves share one `trace_id`.
- The middleware is opt-in (`create_app(tracing_cfg=...)`, `serve-api --tracing`). With tracing off there is no span, no `traceparent` header and no stored context.
- `submit` stores the context on `queue_draft`; `trace_context` is carried through retries. A malformed stored context starts a fresh trace.
- SQLite upgrades old DBs in place with `ALTER TABLE`; Postgres uses Alembic `0002` (not run against a real Postgres here; the contract cases skip without `COUNTONCE_PG_URL`).

## HTTP API load test (overnight)
- No k6/Locust: `countonce/loadtest_api.py` (`countonce loadtest-api`, `scripts/api_load_test.py`, `configs/loadtest_api.yaml`) uses `httpx` clients and uvicorn in-process, and reuses `countonce.loadtest._fake_worker` (no failures or crashes injected: this test is about the HTTP layer).
- The clients run in the server's process (asyncio in the main thread, uvicorn in a thread), so they share the GIL. Absolute latencies are pessimistic and only comparable between runs on the same machine. A separate-process client pool would be more faithful but adds nothing the relative comparison needs.
- `results/api_load.csv` is long format: one row per route plus an `ALL` row per run, with git sha, dirty flag and host info repeated. Gitignored like the other `results/` CSVs; `scripts/make_serving_report.py` renders the numbers into `docs/SERVING.md`.
- "Double-queued" = jobs seen with more than one fresh submit response + job rows beyond the expected count + repeated worker completions + jobs claimed more than once.
- Fixed an API bug the test found: `POST /jobs/{id}/submit` always answered `idempotent_replay: false` for the request that read the job as `draft`, so two concurrent duplicate submits could both claim to be fresh. The queue state was always right (`queue_draft` is one transition). The route now uses `queue_draft`'s return value. The smoke test asserts exactly one replay per duplicate create and per duplicate submit.

## First real-VGGT GPU run and baseline profile (overnight)
- GPU env is `.venv-gpu/` (gitignored): cu128 torch 2.11, `-e .[recon]`, and `vggt` from the public GitHub repo (the `recon` extra does not pull vggt, so it is installed separately; not added to pyproject).
- `countonce run` requires `--prompt`; used `red ball`. The default dedup kept 1 of the 5 views, so the correctness check ran on 1 frame (count 5, same as the fixture backbone).
- The profile uses a driver, `scripts/profile_gpu_baseline.py`, that builds scratch scenes of 3/5/10 frames, sets `dedup_threshold: 0` and `max_frames: N`, and runs `countonce profile` per N. The tiny scene has only 5 views, so N=10 repeats each view twice.
- "Memory guard on" = the default OOM ladder in `countonce/oom.py`. No OOM occurred, so no rung was recorded; the OOM-ladder table stays TODO.
- No separate `lift` timing exists in the pipeline; it is inside `associate`, which PROFILE.md says.
- Recon time is non-monotonic in frame count; recorded as measured, cause not isolated.

## Static UI mockups (PLAN §17)
- Pages share `style.css` and a ~20-line `theme.js` (light/dark/auto toggle, localStorage); no framework or build step. Status colours (green/blue/gray) are kept beyond the single accent because §17 requires them, and always carry a text label plus a distinct outline style.
- `playwright` is not installed, so no PNG screenshots were generated; `scripts/render_mockups.py` skips with a message. Nothing was added to dependencies.
- Photo placeholders are local SVGs, and the share link uses `example.invalid`. All numbers are labelled placeholders.

## Fuji-SfM loader verified against the real download
- `_read_layout` rewritten for the real layout (see `docs/DATASETS.md`); every `TODO: verify` in the module and the Fuji-SfM entry is closed. Loads 1,455 apples, 582 images in about 0.25 s; `validate_scene` reports no errors (about 14 s for the header and tail checks of 582 JPEGs).
- **Surprises:** (1) No camera poses and no frame-to-tree mapping, so per-tree evaluation must select GT by tree and use the east/west sequences as a whole (or recover poses and crop by GT proximity). (2) `1-Mask-set` images are crops of 12 raw frames, not the SfM frames, so `FujiScene.mask_paths` (keyed by SfM index) stays empty; the crops live in `mask_images` / `mask_annotations`. (3) Apple files have no id besides a per-tree counter, so ids are `TreeNN/NNNNNN`. (4) An apple's GT is a small point set (~0.10 m box, 5 cm grid), not a single point; the centroid is the position and the range is `gt_extent`. (5) Counts per tree are the file counts (e.g. Tree01 = 147); the README's "11 trees, 1,455 apples" holds.
- **GT frame:** kept in the shifted frame (saved = global + T), exposed as `gt_shift`, to preserve float precision.
- **Split:** the dataset's `test_set` (Tree01-08, 1,022 apples) is the frozen real test split; `training_set` (Tree09-11, 433) is `val`. Written to `bench/splits_fuji_sfm.json` (a separate file, so the synthetic `bench/splits.json`, which is not frozen yet, is unaffected and consumers that treat its existence as "frozen" do not change behaviour). Only per-tree GT hashes are stored. With only 3 dev trees, thresholds chosen on `val` are noisy.
- **License:** CC BY-NC-SA 4.0 recorded in DATASETS.md and the README; the old "conflicting licenses" note is retired. No HF upload of images or derived GT.
- **Tests:** `tests/test_fuji_sfm_real.py` skips with reason "Fuji-SfM data absent" (allow-listed as `fuji-sfm data` in `scripts/check_skips.py`). The two fixture tests that built the old guessed layout now build the real one.

## HF GroundingDINO backend (2026-11-24)
- **Backend:** `GroundingDinoDetector` takes `backend="hf"` (default when no `config_path`/`checkpoint_path`) or `"groundingdino"` (original package path, unchanged). HF uses `AutoProcessor` + `AutoModelForZeroShotObjectDetection` on `IDEA-Research/grounding-dino-tiny` (Apache-2.0, ungated), `transformers` 5.24.1 in `.venv` and `.venv-gpu`, new `hf` extra.
- **Config name:** the existing detector literal is `gdino`; `groundingdino` was added as an alias (config, CLI, `pipeline.build_detector`), both build this class and so use the HF backend by default. Not renamed, to keep saved configs valid.
- **Cache key:** `version` is `hf:<model_id>@<commit hash>`; the revision comes from the local cache `refs/main` (no network), else `revision=`, else the loaded model's commit hash. `config()` also carries `backend`.
- **Box clipping (found on the real model):** the model returns boxes that overshoot the image by a fraction of a pixel (e.g. y0 = -0.10). The HF path clips to the image and drops boxes left with zero area. The `groundingdino` package path does not clip (not run here).
- **Scores:** HF `post_process_grounded_object_detection` keeps `score > floor` (strict) with the score the max over text tokens, matching the `logits_boxes_to_detections` rule; `floor` stays 0.05 so score distributions are stored. Detection `label` holds the text label. Negatives need no new code: they are a second caption through the same `detect`.
- **Sanity run:** FSC-147 data is absent, so `bench/detector_sanity.py` was run on 20 Fuji-SfM mask-set `validation` images (`fuji_mask` dataset, new `load_fuji_mask_set`; GT = VIA polygon region count). MAE 2.20 [1.10, 3.65], 0.32 s/image warm; numbers and caveats in `docs/PROFILE.md`. Polygon counts may not label every visible apple; MAE is relative to those labels only.

## SAM 2.1 via Hugging Face (`Sam2HfMaskProducer`)
- **Selection:** `Config.masks: sam` uses the HF producer when `mask_checkpoint` is unset, else the `sam2`-package `SamMaskProducer`. `Config.masks` stays `none` by default (the task said "keep `box_fill` the default"; the existing default was `none` and the bench ablation row picks `box_fill` explicitly, so the config default was not changed).
- **Behaviour:** one batched box prompt per frame, `multimask_output=True`, the candidate with the highest predicted IoU per box, clipped to its box. Cache key carries `hf:<model_id>@<commit>` (commit from the local HF cache, else `revision=`).
- **Measured (`pytest -m gpu tests/test_masks_gpu.py`, RTX 5070 Laptop):** one Fuji-SfM mask-set `val` image (`_MG_3004_06.jpg`, 1 GT region only): mean IoU 0.875 against the GT polygon, 0.36 s per call (warm, after model load). A single instance, so this is a smoke number, not a benchmark. transformers logged a harmless `sam2_video` -> `sam2` model-type warning on load.
- **CPU tests** use a fake processor/model; the real model runs only in the `gpu` test.

## Fuji Core cleanup and tests (overnight)
- Dense GT cloud is now memory-bounded: `load_gt_cloud` streams the file with a stride, then `subsample_cloud` (0.02 m voxel, cap `MAX_CLOUD_POINTS` = 200,000, float32); `register_dense` thins both clouds to the cap before building the KD-tree. The exit-137 cause was not reproduced (no GPU/real data run here); the bound is by construction.
- Registration limits found by the new synthetic tests: `register_to_gt` recovers a known Sim(3) (tolerances: scale 2 %, rotation 2 deg, translation 5 cm) with 1 cm noise and 6 outliers in 96 points but fails at about 12 outliers; `register_dense` handles about 1 % stray points and sometimes fails at 2 %. The PCA initialisation is the weak point. Not changed here; worth a robust initialisation if real data shows it.
- `scripts/run_fuji_core.py`: real annotations replace `# type: ignore`; parser split into `build_parser()`/`main(argv)`; `transform_extrinsics`/`camera_centers` imported from `countonce.recon.align`. Stray `runs_v3_train.out` removed; `*.out` ignored.

## 2026-11-24: Fuji Core step 2: resumable runner built, test-split run NOT executed
- Missing slices: **all** test-split rows (Tree01-08, k in 3/5/10/20, east/west, v1/v2/sum/max/median). Reason: no frozen `configs/fuji_core.yaml` (step 1 unfinished; see GATES). `results/bench.csv` has no Fuji rows from this task.
- Choice: resume is keyed on (tree, side, k, method) with status ok (`fuji_core.pending_jobs`), not on `Job.key()`, since the key embeds host and config metadata. Error rows are retried. Every row's config_json carries the frozen-config hash, host and model ids. `--split test` refuses to run without the config file. Only `score_threshold` is read from the frozen file so far; the other thresholds (inlier distance, ROI margin, association distances) will need wiring when step 1 decides them.
- Tests: resume after a simulated interruption, no rerun of finished slices, error/other-split behaviour, config requirement (`tests/test_fuji_core.py`, fake runner, CPU).

## Fuji Core report, step 3 (overnight 2026-11-24)
- **No Fuji rows exist:** `results/bench.csv` is absent because step 2 never ran (no frozen `configs/fuji_core.yaml`, see `docs/GATES.md`). `bench/fuji_report.py` therefore filled **zero** cells: README block (`<!-- fuji-core:begin -->`), `docs/RESULTS_FUJI.md` and `paper/tables/fuji_core.tex` (git-ignored, like the other paper tables) are all `TODO: fill from results`. The failure analysis says plainly that there are no runs to analyse.
- **Cell rule:** a cell is filled only if all 8 test trees x 2 sides have an ok row with a defined metric; per-tree value is the mean over its sides; CI is a 1,000-sample percentile bootstrap over trees (`bench.stats`); paired tests (v2 vs v1, sum, max, median, on MAE) need the same complete set. `write_cell` raises `MissingBackingRow` otherwise (`tests/test_fuji_report.py`). Only rows tagged `protocol: fuji_core_v2` in `config_json` with `split == test` count.
- **"assoc_f1 is not filled" TODO (above) stays:** no Fuji cell is filled yet, so there is nothing to remove it for. Re-running `python -m bench.fuji_report` after step 2 fills the cells that have complete rows.
- **Figures not generated:** `bench.figures.make_all` raises on an empty CSV, so no Core figures exist. Run `python -m bench.figures` after step 2.
- Foliage occlusion is not recorded per row by the runner, so the analysis can only point to miss rate.

## Matcher training speed-up (2026-11-25)
- **Measured s/epoch, GPU, 800-episode `data/dag_v3_capped`, one epoch incl. val pass:** 703.4 s
  before (the recorded epoch-1 `wall_s` of the slow capped run, same data and config) vs 85.4 s
  after. No target was set; this is the measured ratio (about 8x). Table in `docs/PROFILE.md`.
- The "before" number was not re-measured on the old code; it is the earlier run's log. The "after"
  run used `epochs: 1`, so its cosine schedule decays to 0 and its val metrics are not comparable to
  the 40-epoch run. Only timing is meaningful.
- Equivalence is tested on CPU (1e-5, float32). On GPU under bf16 autocast, padded and per-pair
  Sinkhorn differ in summation order, so exact bit equality is not expected.

## First capped v3 matcher run, take 2 (2026-11-25)
- Data hash matched the recorded one (`e9673a1bbeb2`), so nothing was regenerated. The old
  run was moved to `runs/match_v3_capped_slow/`.
- Trained 37 epochs (about 72 s each) and stopped at the 45-minute cap, not by early stop. Best
  val F1 0.9430 at epoch 35. Epoch-to-epoch val F1 drops: [(10, 0.0048), (16, 0.0009), (18, 0.0001), (22, 0.0008), (25, 0.0014), (27, 0.0007), (33, 0.0006), (36, 0.0005)] (no divergence).
- The cap fired while val F1 was still flat-to-rising only marginally; a longer run is the human's call.
- `v001` registered and promoted. With no incumbent the existing gate promotes the first model without
  comparison (integrity check only), so this promotion is not evidence of quality; the synthetic eval
  is in `docs/RESULTS_V3_SYNTH.md`.
- `scripts/eval_v3_synth.py` needs `PYTHONPATH=.` when run directly.

## PROFILE.md fix and variant/cache tables (2026-11-25)
- `scripts/make_profile_report.py` now rewrites a doc only when `--doc` is passed (the old `--out`
  defaulted to `docs/PROFILE.md`, which is how throwaway `/nonexistent*.csv` text got committed).
  Without `--doc` it prints to stdout. Tests pass a `tmp_path` doc; a test asserts the committed doc has no
  `nonexistent`.
- Measured with `configs/profile_tiny.yaml` (tiny scene, 5 frames, GroundingDINO-tiny + VGGT-1B, RTX
  5070 Laptop, 1 warm-up + 3 repeats, `results/*.csv` are gitignored so the numbers live in
  `docs/PROFILE.md`). Baseline is fp32 weights; the tiny scene counts 7 vs ground truth 5 in every variant
  (count delta vs baseline 0), so these runs say nothing about accuracy.
- `compile_default`: no speedup. First repeat 52.5 s (compile), later repeats 17.0 s and 20.2 s;
  p50 is 0.87x of baseline. `compile_reduce_overhead` failed with the exact error
  `reconstruction failed: cuda_error: CUDA driver error: device not ready` (CUDA graphs), recorded in
  the table, not a number. Not confirmed to be an `sm_120` compiler limit; not investigated further.
- bf16 weights cut peak VRAM from 7298 MB to 3833 MB at about the same time; SDPA check has no
  measurable cost. Differences of about 5% between variants are inside the repeat-to-repeat noise
  (baseline totals 16.3 to 22.3 s).

## Confidence stage speed-up (2026-11-25)
- Measured on a 10-frame cached Fuji window: confidence stage 2.50 s to 1.05 s (median of 3), total
  3.58 s to 2.06 s, identical count 42 and interval (14, 120). The 97-frame overview case was not
  re-run, so its 25.0 s is unchanged as a recorded number. Details in `docs/PROFILE.md`.
- `pipeline.make_count_at_reference` keeps the original sweep for the equivalence tests only.

## Fuji-SfM Core benchmark: steps 1 and 2 and report (2026-11-25)
- **Step 1 (training trees Tree09-11, windows 5 and 10, 12 rows per setting):** grid score_threshold
  {0.2,0.3,0.4,0.5} x merge_k {0.3,0.5,0.8}, mean absolute count error. v1 / v2: 0.2/0.3 54.4 / 60.9;
  0.2/0.5 59.6 / 60.9; 0.2/0.8 63.5 / 60.9; 0.3/0.3 79.1 / 81.8; 0.4/0.3 88.2 / 89.3; 0.5/0.3 91.2 / 91.3.
  Error falls as the score threshold falls. Settings below 0.2 were also tried (0.15: 7 of 12 rows,
  0.1 and 0.05: 6 of 12 rows, hit the per-run budget), are not comparable, and 0.05 and 0.1 overcount
  by 100+ apples, so the frozen values are the best complete setting: `score_threshold: 0.2`,
  `merge_k: 0.3` (`configs/fuji_core.yaml`).
- **Not tuned:** the registration inlier distance and ROI margin are code constants in
  `bench/fuji_core.py` (the oracle alignment uses them only to place cameras), and v2's gate is
  scaled from `merge_k`. The task asked to tune these too; only the two thresholds the runner reads
  were tuned. `scripts/run_fuji_core.py bench` gained `--merge-k` and reads `merge_k` from the frozen file.
- **Step 2 (test trees Tree01-08):** all 320 rows ok (k 3/5/10/20, east and west, 5 methods), in two
  invocations because the first stopped at its 38-minute budget with 25 rows missing and the
  runner resumed them (6.9 min). Test numbers were never used to pick settings.
- **Report:** `python -m bench.fuji_report` filled 27 cells; a cell needs all 8 trees x 2 sides
  defined, so most MAPE-style and assoc F1 cells for small k stay `TODO: fill from results`.
- **Failure analysis (observed, not root-caused):** Tree06/west k=5 returned 0 predictions for all
  five methods including the per-image baselines against GT 417, which points at detection or
  frame selection for that window, not association. Tree01 and Tree02 windows frequently have small
  GT counts (down to 0) with far larger predictions, which drives their very high MAPE. At k=20
  every method overcounts (v1 MAE 99.6, v2 101.5, sum 156.9), and v2 is better than v1 at k=10 only
  (-13.4 [-22.3, -5.0]). Foliage occlusion is not scored separately.

## M1: GroundingDINO default, CountGD planned (2026-11-25)
- `Config.detector` and `LiveRoute.detector` default to `gdino` (HF GroundingDINO). `countgd` and `countgdpp` stay valid names (saved configs still load) but `build_detector` raises `PlannedComponentError` (a `NotImplementedError`, so the CLI exits with the message). The `CountGDDetector` class is untouched.
- Removed the `countgd` vs `gdino` ablation from `configs/bench.yaml` (default is now gdino; no second implemented detector to compare). `configs/profile.yaml` uses `gdino`.
- README quickstart installs the `hf` extra (it was missing; the default detector needs `transformers`). README "no real-data benchmark" claim corrected to match the Fuji Core run. New `docs/STATUS.md` is the source of truth for evidence levels.
- Not verified: a literal clean-venv install (no network install attempted); the tiny-scene quickstart command was run in the existing `.venv`.
- Ruff: wrapped/fixed pre-existing lint in `scripts/diag_fuji_protocol.py`. `tests/test_fuji_protocol.py` (untracked, not mine) has a failing test and is left uncommitted.

## M0: Fuji protocol fixes (sixteenth audit)
CPU only, no GPU run; existing `results/bench.csv` rows were produced with the old protocol and are **not** re-scored here (a re-run is a separate task). Tests: `tests/test_fuji_protocol.py`.
- **`visible_gt`:** the depth map is read at the nearest pixel (was truncated) and the last row/column is in frame (was cut by `w-1`/`h-1`). A pixel with NaN/inf/<=0 depth is no occlusion evidence, so the apple stays visible (was silently dropped by a NaN comparison). Depth must be in GT metres (`scaled_depth`).
- **`window_for_tree`:** raises when the side has fewer than `k` frames, `cam_y` length differs from the side length, or `cam_y` is non-finite (was a negative start / silent bad window).
- **Scale alignment:** `scaled_depth` (depth x `sim.s`) is tested against `project_points` z in the GT frame. `check_alignment` records `cam_rms_m`/`cam_span_m` and raises `AlignmentError` when the camera misfit exceeds 50 % of the window span or the scale is not positive (threshold chosen as a coarse sanity gate, not tuned). `scripts/run_fuji_core.py` now passes the overview pseudo-poses to `evaluate_window`.
- **Identity matching:** detections are matched to the nearest GT apple in frame (visible or not), and the label is kept only if that apple is visible. Before, only visible apples were candidates, so a detection on an occluded apple could be labelled as a visible neighbour within 10 px.
- **Interval recording:** pipeline rows now write `low`/`high` (were blank). The pipeline interval covers all instances while `pred` counts only the ROI, so both ends are shifted by the number of out-of-ROI instances and clamped to `low <= pred <= high`. This shift is an approximation (documented, untested against real runs). Baseline rows stay blank.
- **MAPE with GT = 0:** such slices are skipped in the per-tree mean; a tree with GT = 0 on every side drops out of the MAPE cell (`n_trees` shows how many); a missing row still blocks the cell. MAE is unchanged and still includes those slices.

## Tiny-scene over-count, root cause (M2, 2026-11-25)
Real GroundingDINO-tiny + VGGT-1B counted 7 on `examples/tiny_scene` (truth 5). Diagnosed per stage:
- **Frames: five views deduplicated to one (the "smoke-check note").** The scene is a pentagon of reds
  seen from azimuths 72 degrees apart, so the views are near-identical; the 16x16 mean-abs-diff
  thumbnail distance is 0.008-0.0095 against the 0.02 default `dedup_threshold`. The default run
  therefore used 1 frame (0.005 keeps 4, 0 keeps 5). The pipeline now logs a warning when the
  filter keeps 1 of N>1 frames. The metric was not changed: with all five views the count is worse (below).
- **Detections: blue/green distractors scored as "red sphere".** Per frame the 5 reds score 0.52-0.64
  (0.48 for one occluded red in view 2) and the blue and green spheres 0.40-0.50, all above the 0.3
  default. These two false positives are the 7 = 5 + 2.
- **Per-frame duplicate boxes: not the cause.** No same-frame box pair was a duplicate object; added
  `nms_per_frame` / `Config.nms_iou` (default 0 = off, so published runs are unchanged; the test uses 0.7).
- **Lifting / merge radius: not the cause of the single-view 7, but a real limit with 5 views.** VGGT
  on the 128x96 flat-shaded symmetric scene recovers cameras poorly (Sim(3)-aligned camera error 0.1-0.9
  on a radius-4 orbit) and lifted points sit about 0.3-0.7 from the true sphere centres (radius 0.22), so
  views of one sphere do not fall within `merge_k` object sizes. With dedup off and 5 views the count
  is 22 at k=0.5 and 8-10 for k in 1-6 at the default threshold; no merge radius gives 5.
- **Fix used (regression only, defaults unchanged):** `score_threshold=0.5` splits reds from
  distractors on every view but the occluded one, giving 5 on the default single-frame path
  (`tests/test_tiny_scene_real_gpu.py`, GPU marked). This is a threshold chosen on the tiny
  fixture, not evidence about accuracy elsewhere; the Fuji grid still picks the defaults. The
  5-view real path stays open (needs better geometry on this scene, not a merge-radius change).

## M3: count-first metrics
- **Sign convention:** PLAN §9.4 writes `missed + duplicates + FP − wrong merges = predicted − true`. Missed objects lower the count, so the exact identity is `duplicates + false_positives − missed − wrong_merges = predicted − true`. `bench.metrics.count_decomposition` returns the four non-negative magnitudes and `decomposition_error` applies the signs; the figure stacks over-counts up and under-counts down.
- **Wrong merge vs missed:** a true object with detections that is the majority of no cluster counts as a wrong merge; one with no detection at all counts as missed.
- **Decomposition scope:** `bench.evaluate` computes it only for scenes with identity labels (`instances`, optional `objects`) whose object count equals GT `count`; it is a mean over those scenes. Cluster count, not the submitted `count`, is what it explains.
- `assoc_f1` stays in the result but is labelled explanatory and listed after every count metric. Fuji report gained a signed-bias column; the old `bench.csv` needs no change (bias is computed from pred - gt).

## M4: photo-order invariance (2026-11-25)
- **The old test did not test this.** `test_invariant_to_view_and_detection_order` shuffles the detection list and reorders whole-frame blocks, but every frame keeps its frame id and camera. v2 visits frames in ascending id, so it passed while still depending on photo order.
- **True test:** `bench/order_invariance.py` moves photo `p` to frame id `sigma[p]` and permutes `intrinsics/extrinsics/depth/depth_conf/world_points*` with it (detection ids stay tied to the photo). `tests/test_photo_order_invariance.py` runs 12 random permutations per scene, 4 synthetic scenes (15 objects, 6 views, box noise 0.03 / 0.15 / 0.4 of object size, `make_scene`), for v1, v2, sync, v3 and v3+sync, plus a check that the harness catches the old behaviour.
- **Measured before the fix** (4 scenes x 10 permutations per noise level; "changed" = partition differs from identity order):
  - v1, v3 (threshold): 0/40 changed at every noise level.
  - v2: 0/40 at noise 0.03, 38/40 at 0.15 (count range 3), 38/40 at 0.4 (count range 4); worst Rand index 0.988.
  - sync: 0/40, 19/40 (range 1), 0/40. v3 + sync: 34/40, 34/40, 33/40 changed (count range 0, Rand index down to 0.94).
  - Cause: v2 builds tracks greedily in frame-id order; the spectral rounding in `synchronize` assigns views in sorted view-id order.
- **Decision: fix, do not narrow the claim.** New `countonce/merge/order.py` orders frames by camera centre (frame id only breaks exact ties). v2 visits frames in that order, `SyncAssociator` sorts detections and passes the rank as the view id, and `V3Associator(synchronize=True)` does the same. No global refinement was added: a canonical visit order makes the result a function of the photos, which is what the invariance claim needs. It does not make the greedy result optimal; quality vs the old order was not re-benchmarked on Fuji (a GPU/benchmark re-run is needed for that).
- **After the fix:** 0 changed partitions and count range 0 for all five associators, all scenes, all permutations. Scope: synthetic scenes with finite-precision cameras; exact ties in camera centre (duplicated cameras) fall back to frame id and are not invariant.
- Plain v3 and v1 needed no change. Defaults and published numbers for v1 and v3 are unchanged; v2/sync outputs can differ from before on scenes where the frame-id order mattered.

## M6: controlled failure-source harness (2026-11-25)
- **What:** `bench/failure_sources.py` + `scripts/run_failure_sources.py` + `tests/test_failure_sources.py`. A `SceneBundle` (GT positions and radius, oracle cameras and depth, an estimated recon, real detections) is run under four conditions: oracle detections + estimated geometry, real detections + oracle geometry, full pipeline, and an oracle/oracle ceiling. Detections are lifted once per condition; every associator gets that same `lifted` list and `recon`, and each row carries an `input_hash` (tests assert it is constant across methods within a condition). `tune_associators` grid-searches v1/v2/sync on validation bundles; `build_associators` takes the frozen result (v3 added with a checkpoint).
- **Oracle detections:** GT objects projected with the oracle cameras, a box of the object's true radius, kept only where the oracle depth map shows the object (nothing nearer hides its centre). They are lifted through the normal `lift_detections`, so lifting error is part of what the harness measures, not removed.
- **Stand-in geometry (CPU):** no weights here, so "estimated" is `perturb_recon(oracle)`: per-view pose jitter (1.5 deg, 0.05 units on the tiny scene), per-view depth scale drift, per-pixel noise, world points re-derived from the perturbed cameras. This is a stand-in for VGGT error, not a model of it. On a GPU box pass a real reconstruction (`tiny_scene_bundle(estimated=...)`) and real detections (`detections=...`). Real-photo conclusions need those runs; the numbers in `results/failure_sources_*.json` are component-test numbers.
- **Tiny scene runs at default associator settings** (one scene, no validation split to tune on); the synthetic fixture tunes on separate validation episodes. Tuning on a scene and scoring on it would be leakage, so the tiny-scene test of the ceiling is labelled as such.
- **Lift failures** never reach an associator, so they are left out of the error decomposition (an object seen only through failed lifts counts as missed) and are reported as `n_lift_failed`.
- **Observed (not claims about real data):** on the tiny scene, default v1 over-counts even with oracle detections and geometry (front-surface lifting puts one sphere's views about 0.2 apart; tuned k=0.75 gives 5), and the perturbed geometry is the dominant error source. On the synthetic fixture the injected false positives are the main error for v1 under real detections.
- **Not done here:** Fuji-SfM and own captures need a loader that fills a `SceneBundle` (GT trees from `bench.fuji_core`, dataset cameras as oracle); that is the M8 task. No GPU run was made.

## M8: Fuji-SfM Core run and G3 (2026-11-25)

- Resumed the `fuji_core_v2` test sweep with the frozen `configs/fuji_core.yaml`: 320 rows, 0 failed. A second writer started by mistake appended 2 duplicate rows; both were removed (identical preds). Report, figures and README table regenerated from `results/bench.csv`; the hand-written reading lives in `docs/RESULTS_FUJI_ANALYSIS.md` and is appended by `bench.fuji_report`.
- Failure-source experiment on Fuji (`scripts/run_failure_sources_fuji.py`, k=10): Fuji has no depth or own poses, so oracle geometry is not available; only oracle-detections and full conditions ran. Assumed apple radius 4 cm for oracle boxes (not measured). Predictions there are not ROI-filtered, so MAE is not comparable with the Core table.
- Not done: baselines beyond Sum/Max/Median (sync, COLMAP), sweep axes other than k, v3 on Fuji (G4 evidence), own captures.

## M9: CI consistency (overnight, 2026-11-25)
- **`db` extra:** added to `pyproject.toml` (`sqlalchemy`, `alembic`, `psycopg[binary]`) rather than dropped from `ci.yml`, because `countonce/db` and `jobs/pg_store.py` import them. `uv.lock` updated.
- **mypy runs only in `full-cpu`.** Without torch/sqlalchemy/httpx the results differ (unfollowed imports become `Any`), so one config cannot be clean in both jobs. `tests/test_mypy_clean.py` skips when those are absent.
- The `test` job now installs `.[dev,test,data]`; with `.[dev]` alone, collection failed and several tests errored on matplotlib/pyarrow.
- `full-cpu` test step uses `shell: bash` (pipefail): before, `| tee` hid a pytest failure.
- GPU modules that skip at import time now say `gpu test: ...` so `scripts/check_skips.py` accepts them.
- `tests/test_fuji_core.py` resume test was stale since `fuji_core_v2` (protocol belongs in the job config); it also failed locally.
- `huggingface_hub.*` added to mypy `ignore_missing_imports`; `warn_unused_ignores` off for the two HF model modules.
- **Node job:** `web/` does not exist yet (M12). The `web` job detects `web/package.json` and is a no-op without it. It expects a committed `package-lock.json` and runs `npm ci`, lint/typecheck/test if present, then build. M12 may need to add Playwright browser install.
- **Verification is a dry run, not `act`** (no Docker here): the `run:` commands in fresh Python 3.11 venvs on a copy of the tracked files, with embedded Postgres (`pgserver`) in place of the `postgres:16` service. The GitHub Actions themselves and the `web` job did not execute. `golden-gpu.yml` (self-hosted) was not run.

## 2026-11-25: M10 related-work rewrite
- `docs/RELATED_WORK.md` is organised around the §2.4 hypothesis. The C1-C5 coverage matrix and the "claim survives" verdict were removed: they rested on negative claims about unread papers.
- Only CountVid, DG-Net and OneScene-Bench were read in full (the task's scope). Other works are tagged abstract/search/repo level and described only by what they state.
- §2.4 does not define the two sides of "geometry-aware vs simpler geometric matching"; read as v2/sync/v3 vs v1/C-DOG.
- Also updated where the full read contradicted them: PLAN §2.3 rows and §2.4 note, MODEL_SURVEY (DG-Net accepts estimated depth; venue), DATASETS (OneScene-Bench release has no images; loader layout assumption is wrong), EXPLAINER Q2, README license table, `paper/refs.bib`. The OneScene loader code was not changed (no images to test against).

## M11 capture kit
- `scripts/ingest_captures.py` re-encodes pixels to JPEG (q95, orientation applied, long side <= 2048, no EXIF/ICC) and renames to `img_NNN.jpg`; original filenames are not kept (they can carry dates). HEIC is rejected with a message rather than adding a plugin dependency.
- Hand counts are required (`--counts A B [C]`, or `hand_counts` in the scene list); two must agree. A count of 0 needs `--zero` (and `--zero` needs count 0), so zero scenes are deliberate. `gt.json` adds `prompt`, `physical_scene`, `zero_target`, `hand_counts`, `count_definition`, per-image sha256; extra keys are ignored by `bench.sweep`.
- Template `configs/capture_scenes.template.yaml` lists 20 scenes (16 positive across tiers, 4 zero-target); PLAN asks for 20-30, so add more as places are found. Video is not ingested.

## M12 web scaffold (2026-11-25)
- Scaffolded `web/` by hand (not `create-next-app`/`shadcn init`, both interactive): Next 16, React 19, Tailwind v4 (tokens in `globals.css` via `@theme`), shadcn-style `Button`/`Card` with `components.json` so `shadcn add` works later. TypeScript pinned to 5.x (7.x is too new for the Next toolchain).
- `scripts/export_openapi.py` writes `web/openapi.json`; `npm run gen:api` makes `schema.gen.ts`. Committed both; `tests/test_export_openapi.py` fails when the spec is stale.
- Backend fix found by type generation: `/health` was one `api_route(GET, HEAD)` giving duplicate operation IDs; now `GET` documented, `HEAD` hidden from the schema.
- Zod schemas (`web/src/lib/schemas/result.ts`) are strict and upgrade 1.0 -> 1.1 like the backend; Vitest reads the backend's `tests/fixtures/schemas/v*` so versions cannot drift unnoticed.
- CI `web` job now also runs Playwright (+ axe, desktop and phone projects). Not yet run on hosted CI.
- Not done (later tasks): Lighthouse CI, the "no `TODO: fill from results` outside empty states" build check, charts, auth, upload client.

## M14: results experience on demo assets
- **Directory name `result-view`, not `results`:** the repo `.gitignore` has `results/`, which would have excluded `web/src/**/results/` from git *and* from Tailwind v4's source scan (utilities such as `py-2` silently vanished from the CSS). Components live in `web/src/components/result-view/`, helpers in `web/src/lib/result-view/`.
- **Detection boxes are fixed-size squares (18 px) centred on `uv`:** `result.json` stores only the detection centre, not box extents, so the Photos tab and strip draw a square marker with the status outline style (solid / dashed / dotted) and label. The exported thumbnails (`thumbs/*.jpg`) already have the backend overlay burned in, so each tile shows the exporter's boxes plus the interactive ones. A follow-up should export clean thumbnails and real box extents.
- **Camera fly-to keeps world-up (+z):** orbit controls stay z-up, so the view matches the photo's position, direction and field of view but not any roll. Verified visually that the point cloud lines up with the photo on the tiny scene.
- **Interval wording:** the tiny scene's interval method is `sensitivity` (no calibration set yet), shown as "sensitivity to settings"; the "split conformal, n = ..." wording appears only when `n_calibration` is present.
- **Confidence badge rule** (`confidenceBadge`): Low if the range is wider than the count itself, Medium if wider than 20% or any warning, else High. The thresholds are provisional until calibrated (P-track); the reason is always shown next to the badge.
- **Report / Share:** Report calls `window.print()` (a real PDF report needs the backend); CSV is generated in the browser; Share copies the page URL. Corrections, feedback, the Gaussian splat toggle and `/jobs/[id]`, `/s/[token]` routes are out of scope for M14 (demo assets only).
- **Match lines** draw for the selected or hovered object; the numbered 3D markers are 1-based (`#1` is `instances[0]`).

## M15: landing page and demo gallery
- **Hero is a static photo, not a spinning 3D scene.** PLAN 17.3 asks for an interactive 3D hero. Loading three.js on `/` would cost most of the Lighthouse performance score, so the hero is the scene's first photo thumbnail linking to `/demo/<scene>` where the real viewer lives.
- **Proof points are empty states** ("No published number yet") until benchmark numbers are final; no figures were invented.
- **"Count your own photos" is a disabled button** with an explanation: `/new` and sign-in do not exist yet.
- **No links to unbuilt routes** (`/benchmark`, `/how-it-works`, `/about`): the nav only has the gallery. The benchmark teaser section is omitted until `/benchmark` exists.
- **Only one demo scene exists** (synthetic `tiny`), so the gallery has one card and says real scenes are added after review.
- Components live in `web/src/components/chrome/`, not `site/`, because `.gitignore` ignores `site/`.
- **Lighthouse** (v13, mobile preset, production build, headless Chromium): performance 0.96, accessibility 1.00, best-practices 0.96 on `/`. Reproduce with `npm run build && npx next start -p 3100` then `npm run lighthouse` in `web/`.

## M16: signed-in web flow (2026-11-25)
- **No Auth.js** (PLAN 17 names it): a small authorization-code + PKCE flow in `web/src/app/api/auth/google/` with an AES-GCM httpOnly session cookie. The only identity needed is an API key from the existing API, so a session library added little. The id_token comes straight from Google's token endpoint over TLS, so its claims are checked but its signature is not.
- **New API routes:** `POST /auth/exchange` (off unless `COUNTONCE_AUTH_SECRET` is set), `GET /me`, and corrections on finished jobs (migration 0003 adds `corrections.key` and `consent`). Every sign-in mints a new API key; old keys are not revoked yet.
- **Backend-for-frontend proxy:** the browser never holds an API key and no CORS is needed. Upload chunks pass through the web server, which on Vercel is subject to request-size and duration limits; M18 must check this.
- **Share pages and corrections:** shared links show the original result, not the owner's corrections.
- **Test sign-in** (`COUNTONCE_DEV_AUTH=1`) exists so both end-to-end suites run without Google.
- **CI:** the `web` job runs the mocked-API suite only. The live suite needs the Python env and is run locally (`npm run test:e2e:live`).
- **Not done:** email-me-when-done toggle (Moon), real Google sign-in against live credentials.

## M17: benchmark and information pages (2026-11-25)
- **`/benchmark` reads a summary, not raw `bench.csv`:** `bench/web_summary.py` (also run by `bench.fuji_report`) writes `results/benchmark_summary.csv` with the same statistics as `docs/RESULTS_FUJI.md`, so the two cannot disagree. Only aggregates; no images or per-tree ground truth.
- **That one file is committed** (`git add -f`; the rest of `results/` stays ignored) so a clean checkout builds with real numbers. `v3_capped_synth_eval.csv` stays local, so elsewhere that section shows the empty state. A deployment whose root is `web/` cannot see `../results`: M18 must include the folder in the build or set `COUNTONCE_RESULTS_DIR`.
- **Fuji runs are marked dirty** in `bench.csv`; the page says so next to the commit.
- **Synthetic v3 numbers predate M5** (not done: baselines untuned, no occlusion check); the page says this and labels them a component test.
- **Chart is hand-written SVG**, not Recharts/Observable Plot: one dot-and-interval figure did not justify a dependency.
- **Not on the page yet:** ordered vs. shuffled, scene type, VLM baseline, failure gallery, leaderboard (listed as not measured).
- **`/how-it-works` uses the synthetic tiny scene** (colour detector, fixture geometry) and says so.
- **New public `GET /status`:** queue state and timings of the last 10 finished jobs; no ids, users, prompts or worker names.
- **`/privacy` states the gaps** instead of the PLAN 14.8 intent: no automatic 30-day deletion yet (M19), no automatic face/plate blurring, video metadata kept. Update the page when these change.
- **`/about` has no email address;** contact is through GitHub issues.
- **Build check** scans prerendered pages, browser bundles, `public/` and page sources (server-rendered pages have no prerendered output).

## M18: API fit for Vercel
- **Separate API project (`api.`), not `/api` in the site.** Python and Node in one project would need rewrites and two runtimes; the site proxy already makes the API same-origin for the browser. Details in `docs/DEPLOY.md`.
- **SSE capped, polling fallback kept.** `COUNTONCE_SSE_MAX_S` ends the stream before the function limit; the existing client falls back to polling. No change to the client was needed; the proxy gets `maxDuration = 60`.
- **`/` on the API redirects** to `COUNTONCE_WEB_URL` (else `/docs`) instead of returning JSON. Health probes should use `/health`.
- **Benchmark data for a `web/`-rooted deploy:** `web/content/benchmark/benchmark_summary.csv` is committed (copy of `results/`, `npm run sync:results`). The `content/` name avoids the repo `.gitignore` entries for `data/` and `results/`.
- **Not verified:** no real deployment was made (needs your Vercel/Neon/R2 accounts), so "frontend talks to the deployed API" is covered by the existing live Playwright run against a local API plus the proxy, not by a hosted one. Logged in `docs/GATES.md`.
- Still open from M16: old API keys are not revoked on re-sign-in; `example_box` option rejected by `Config` in the worker (M19).

## M19: worker hardening
- **Supervisor in-process, not a systemd dependency.** `--supervise` works anywhere (WSL2 without systemd included); `--print-unit` covers boot start.
- **Retention lives in the worker loop** (hourly, default 30 days) rather than a separate scheduler, so the one-command start covers it. No worker, no purge; `/privacy` states this.
- **`example_box` is dropped before `Config`** (`UI_ONLY_OPTIONS` in `countonce/jobs/worker.py`): the pipeline has no use for it yet, so it is stored with the job but not applied. Fixes the M16 failure.
- **Not verified:** the "upload from the frontend -> worker on GPU -> result page" run on real hardware was not done unattended (see `docs/GATES.md`). Covered: worker/queue logic by `tests/test_worker_hardening.py`, and the frontend-to-API path by the M16 live Playwright run with a fake worker.

## M20: README front page (2026-11-25)
- **No hand-counted photos exist and Fuji images cannot be shown** (PLAN 0.1), so the example is the synthetic tiny scene, labelled as such; "true count" is by construction, not a hand count. Replace with an own capture in P4/P5.
- **Two example runs, both labelled:** consistent IDs across views come from the CPU stand-ins (`color_blob` + `fixture`); the failure is the real GroundingDINO + VGGT path on all five views (14, truth 5). The real path's correct 5 uses one view, so it cannot show cross-view IDs.
- **The tiny-scene table was re-measured** (2026-11-25, `.venv-gpu`, commit b22b2ba, `backbone_options.weights_dtype: fp32`, v1 and v2). It replaces nothing; M2's "22 at k=0.5" was a different association setting/code state, today's v1 default gives 23.
- **One measured table is MAE only**, copied from `results/benchmark_summary.csv`; `tests/test_readme.py` fails if they diverge. `bench.fuji_report` no longer writes a block into the README (it would bring the TODO cells back); the full table stays in `docs/RESULTS_FUJI.md`.
- **Live demo link:** none exists, so the README says "not hosted yet" (see `docs/GATES.md`).
- **Found, not fixed (outside M20):** the CLI's `expandable_segments:True` (`countonce/oom.py`) made the five-view VGGT run fail with `CUDA driver error: device not ready` on WSL2 + RTX 5070 8 GB (3 of 3 CLI attempts); with `expandable_segments:False` it succeeded, as did the Python API path, which does not set it. The README command sets the override.

## M21 repo hygiene (2026-11-25)
- Apache-2.0 covers the code only; model weights and datasets keep their own licenses (README note).
- Agent files are untracked and gitignored but stay on disk. `CLAUDE.md` is still pending (see GATES.md).
- The demo exporter rewrites the absolute repo root to `<repo>` in exported JSON; the tiny demo was re-exported.
- Author name and GitHub URLs stay: they are attribution, not personal paths. History was not rewritten.


## M22: final consistency pass (2026-11-25)
- **README setup was wrong:** `.[dev,recon]` has neither `transformers` (default detector), `torchvision`
  nor `vggt`. The README now installs them, with `vggt` pinned to the commit used for measured runs.
  `pyproject.toml` and `uv.lock` were not changed.
- **Landing proof points** said no number was published while README and `/benchmark` published them.
  They now read the benchmark summary at build time, with titles that do not claim more than measured.
- **No-WebGL crash:** a failed 3D renderer replaced the whole results page with an error screen. It is
  now contained by `ViewerBoundary` (Playwright test).
- **EXPLAINER** rewritten for the current code, with 33 interview questions. Numbers are copied from
  the named reports; there is no test tying them to the source files.
- **M5 was never completed;** STATUS, EXPLAINER and `RESULTS_V3_SYNTH.md` now say so.
- **§0.6:** rows are ticked on evidence only, so most stay open (GATES).

## M22 follow-up: allocator setting on WSL (2026-11-25)
- `configure_allocator()` no longer sets `expandable_segments:True` when the kernel release names
  Microsoft (WSL). Native Linux is unchanged, and a user's own `PYTORCH_CUDA_ALLOC_CONF` is never edited
  on WSL. Chosen over documenting an override because the default CLI and worker path must not crash.
- Evidence (RTX 5070 Laptop 8 GB, torch 2.15.0+cu132, five-view tiny scene): before, 3 of 3 failed;
  after, 3 of 3 returned 14, the README figure. The driver-level cause is not known.
- Cost: WSL loses the fragmentation benefit; the OOM ladder still applies.

## M5a: synthetic component-test harness (2026-11-25)
- **Decomposition sign:** the task repeats the PLAN §9.4 wording; the exact identity stays
  `duplicates + FP - missed - wrong merges = predicted - true` (see the M3 note). Fixed the same
  slip in the `count_decomposition` docstring.
- **Rejected detections** are no longer scored as one-detection clusters in the decomposition,
  duplicate rate and miss rate: they are not predicted instances, and counting them broke the identity.
  `bench/synth_assoc.py` (the old v3 table) is unchanged and still treats them as singletons.
- **True count** stays `Episode.true_count` (objects with at least one detection), as in the first
  comparison, so `missed` is non-zero only when the associator rejects all of an object's detections.
- **Zero-scene false count** is `nan` with `n_zero = 0` when no episode has a true count of zero. The
  generator's false positives scale with the instance count, so it cannot make a useful zero scene;
  none were added.
- **CSV layout:** `--out-csv` is now the count-first summary (one row per variant and method);
  per-episode rows moved to `--out-rows-csv`.

## Worker image extra (2026-11-25)
- Added `worker` extra (recon + hf + torchvision + huggingface_hub + data + `vggt` pinned to commit a288dd0, same pin as README). Dockerfile installs `.[worker]` and apt-installs `git` (needed for the vggt git URL). Image not built (Docker is a human gate).
- `uv lock` re-resolved other packages (e.g. opencv-python-headless 5.1 -> 4.11, as vggt constrains it); `uv lock --check` passes.
- README setup now says `uv pip install -e ".[dev,worker]"`, the verified path; plain pip not run.
- Fixed a broken M5b-WIP test (`tests/test_synth_component.py` lambda lacked `device` kwarg).

## Order-invariance on real Fuji data (2026-11-25, validation trees only)
- **Run:** `scripts/run_order_invariance_fuji.py` on Tree09-11 x east/west x k in {5, 10} (12 windows), frozen thresholds (score 0.2, merge_k 0.3), cached recon and detections, 10 photo permutations per window per associator (`bench.order_invariance.fuji_window_row`). Output: `results/order_invariance_fuji.csv`. No test-split tree touched.
- **Result:** v1, v2 and sync all gave count range 0, 0 changed partitions, Rand index 1.0 in every window.
- **But it is mostly vacuous:** v2 and sync merged nothing in all 12 windows (instance count = number of lifted detections), and v1 merged nothing in 4 of 12. Invariance holds trivially when nothing is merged. Only v1's 8 merging windows are real evidence.
- **v2 count error, old frame-id order vs new camera-centre order** (`results/order_invariance_fuji_v2_mae.csv`, ROI counts vs GT): MAE 61.25 for both. Four windows move by 1-3 instances (Tree10 west k5 71 to 69, Tree09 east k10 86 to 89, Tree10 west k10 67 to 66), the rest are identical. The fix did not change v2's accuracy on Fuji; v2's weakness there is that it merges almost nothing, not visit order.
- **Does the "unordered photos" claim hold?** Supported for v1 on real data. For v2/sync it is supported only by the synthetic test (M4); the real-data run cannot confirm or refute it. **The §0.6 row stays unticked.** Needed: a real scene (own captures, P2) or a looser v2 gate where v2 merges.
- **Side finding:** `canonical_frame_order` sorts camera centres lexicographically by (x, y, z). On a row walk along y this makes x noise the main key. It is still permutation-invariant, but not a meaningful "along the path" order. Not changed.
- Test: `tests/test_order_invariance_fuji_adapter.py` (CPU, tiny synthetic window).

## `/demo/tiny` Lighthouse performance (2026-11-25)
- **Method:** Lighthouse 13, default mobile preset (simulated throttling), headless Playwright Chromium on WSL2 (software rendering), production build served by `next start -p 3100`. Same flags as the `lighthouse` npm script, with the URL swapped per page and run from `/tmp`. Local runs, not Lighthouse CI or a real phone.

| `/demo/tiny` | performance (3 runs) | median | LCP | TBT |
|---|---|---|---|---|
| before | 0.51, 0.54, 0.56 | 0.54 | 5.7-5.9 s | 1.12-1.85 s |
| lazy viewer only | 0.76, 0.78, 0.77 | 0.77 | 3.2 s | 620-702 ms |
| final | 0.93, 0.92, 0.94 | **0.93** | 0.77-0.78 s | 282-357 ms |

- **Final build, accessibility / best practices:** 1.00 / 0.96. Script transfer before the viewer: 168.2 KiB (budget 200 KB).
- **Other pages, 3 runs:** `/` 0.95, 0.94, 0.95 (median 0.95); `/benchmark` 0.94 in all three.
- **Other pages, 1 run each:** `/demo` 0.93, `/how-it-works` 0.98, `/about` 0.95, `/privacy` 0.90, `/status` 0.92, `/history` 0.94, `/new` 0.96, `/signin` 0.93. `/jobs/[id]` and `/s/[token]` were not measured (they need backend data).
- **The margin is thin and noisy:** an intermediate `/demo/tiny` build with the same fix plus `<Suspense>` boundaries scored 0.90, 0.84, 0.82, 0.93, 0.92. TBT swings widely on this shared machine. Most of it is evaluation of the shared framework chunks, which `/` pays too.
- **Changes:** scene JSON is read and validated at build time (zod leaves the client path, count and photos are in the static HTML); `points.bin` and the three.js viewer load only on activation; media queries are read only in the client-only viewer wrapper and an effect. The last one removed a forced second render on phones and took LCP from 3.2 s to 0.8 s.
- **Behaviour change:** phones show a "Load 3D view" button (selecting an object or photo also loads it); desktop loads the viewer 1.2 s after the page load event. The poster is a text panel, not an image.
- **No worker for point decoding:** `parsePoints` only creates typed-array views over the fetched buffer, so there is no decode work to move off the main thread.
- **`<Suspense>` boundaries rejected:** React's prerender moved the large sections into hidden segments revealed by a script, one Playwright test then saw duplicate nodes, and the long hydration task did not shrink.
