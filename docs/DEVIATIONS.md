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
- COLMAP results have no per-view depth, so `to_dense_recon` raises `ReconAdaptError` for them. COLMAP stays a baseline (sparse metrics only) instead of a lifting backbone.
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
