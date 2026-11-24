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
