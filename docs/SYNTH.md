# Synthetic data (PLAN §10.2, §16B.1)

Renderer-agnostic: `synth/recipe.py` describes a scene, a renderer (Kubric or BlenderProc, not yet
written) consumes it, and `synth/validate.py` gates the output before any GPU model runs.

## Recipe (`synth.recipe.make_recipe(seed)`, `RECIPE_VERSION = "1.0"`)
Pydantic `SceneRecipe`, deterministic per seed, JSON round-trip via `to_json` / `from_json`.

| Field | Contents |
|---|---|
| objects | 5–150 instances, one asset per scene (many visually identical), scale, texture id, position, yaw |
| arrangement | `scattered`, `piled`, `canopy`, `shelf_rows` |
| occluders | `leaf_plane`, `branch`, `box` |
| distractors | other fruit assets (negative examples) |
| lighting | HDRI id, sun azimuth/elevation/intensity, ambient |
| rig | 3–40 cameras along `arc`, `circle` or `random_walk`; varied distance, height, FOV |
| randomization | texture variety, blur, noise, exposure EV, JPEG quality, detection drop, false-positive rate |
| meta | `recipe_version`, `seed`, `prompt` |

## Data-quality gates (`synth.validate`)
A scene is a `RenderedScene` (images, depth, instance masks, camera centres, metadata dict).
`validate_scene` returns quarantine reasons; `validate_scenes` builds a `QuarantineReport`
(`quarantine.json` + `quarantine.md`, counts by reason). Target: < 2% quarantined.

| Reason | Rule |
|---|---|
| `schema_invalid` | metadata fails `SceneMetadata` (Pydantic) |
| `nan_or_inf_depth` | any non-finite depth value |
| `all_black_frame` | any frame with mean intensity <= 2 (uint8) |
| `overexposed_frame` | any frame with >= 98% of pixels >= 250 in all channels |
| `zero_visible_target_pixels` | no target-instance pixel in any view |
| `fewer_than_2_views_see_target` | no target instance visible in >= 2 views |
| `degenerate_camera_rig` | camera centres (and viewing directions) nearly identical, non-finite, or < 2 cameras |

## Dataset versioning with DVC (PLAN §16B.1)
`dvc.yaml` and `params.yaml` are generated from `synth.dag.STAGE_GRAPH` (each stage's deps, outs and
params), so they cannot drift from the DAG. Never edit them by hand:

```
make dvc-yaml     # regenerate (params.yaml = configs/dag.yaml over DagConfig defaults)
make dvc-check    # fail if the committed files are stale (also covered by tests/test_dvc_yaml.py)
```

Each DVC stage runs `countonce dag run --config params.yaml --only <stage>`. The optional extra
(`uv pip install -e .[dvc]`) is needed only to run `dvc`; tests do not need it (`dvc dag` is
checked under `importorskip`).

Workflow once `dvc` is installed: `dvc init` (once), `dvc repro`, then commit `dvc.lock` with the
code. When `dvc.lock` exists in the working directory, the feature-store/catalog data version and
the checkpoint `data_version` become `dvc:<12-char sha256 of dvc.lock>` (`synth.dag.dvc_lock_hash`);
without it they keep the episode-content hash. Checkout of an old git commit plus `dvc checkout`
restores the matching data.

**No remote is configured.** Adding the Cloudflare R2 remote (`dvc remote add`, bucket and
credentials) is a human gate; see `docs/GATES.md`.

## Cost
- Storage per scene / per 1,000 scenes: TODO: fill from results
- Render time per scene (renderer, GPU/CPU): TODO: fill from results
- Quarantine rate on the 1,000-scene smoke set: TODO: fill from results
