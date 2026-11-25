# CountOnce capture guide (print one page)

Goal: 20–30 independent static scenes, each shot as **unordered photos** with a hand-counted truth.
Scene list: `configs/capture_scenes.template.yaml` (≥ 3 zero-target scenes).

## What to shoot
- A **static** scene: nothing moves, nothing is added or removed between photos.
- One physical place = one `physical_scene` id. Reshooting the same place is allowed but shares the id (splits are by place).
- Mix tiers: **easy** (bike racks, chairs, shelf items), **medium** (plants, cones, bins), **hard** (fruit on trees, dense look-alikes, heavy occlusion).
- **Zero-target scenes (≥ 3):** the target is absent. Use an empty place, look-alike distractors (stools for "chair", oranges for "apple") or a prompt for something not there.

## How to shoot
1. Pick the prompt (one noun, e.g. `bicycle`) **before** shooting.
2. Walk a slow arc around the objects; **20 stills** at ~30–50 % overlap between neighbours, 6–10 minimum.
3. Plus an optional 30–60 s video along the same arc (kept outside the scene folder).
4. Keep every object visible in at least two photos where possible; vary height a little.
5. Phone defaults: JPEG (not HEIC), no filters, no portrait mode, no digital zoom, flash off, exposure locked.
6. Note lighting and problems (glare, motion blur) in `notes`.

## Ground-truth definition (fixed)
`count` = number of target objects **visible in at least one submitted photo**. Objects fully hidden in every photo do not count. The count is made **from the photos, not from the physical scene**. `gt.json` stores this definition.

## Hand-count procedure
1. Person A counts the submitted photos alone: list each object once across all photos (sketch or tally by landmark). Write the number down. Do not look at any model output.
2. Person B repeats independently (same person, a day later, if alone).
3. Equal → that is the count. Different → a third count; two of three must agree. All three different → recount together and write down why.
4. Zero-target: both counts are 0; confirm no look-alike is a target.
5. Record the counts: `hand_counts: [A, B]` (or `[A, B, C]`).

## Ingest
```bash
python scripts/ingest_captures.py --scene-list configs/capture_scenes.yaml \
    --src <phone folder> --scene-id <id> --counts A B [C]
```
Creates `data/scenes/<id>/{images/, gt.json}`: photos re-encoded as `img_000.jpg`…, **EXIF/GPS stripped**, orientation applied, long side ≤ 2048 px, original filenames dropped. Export HEIC as JPEG first. Faces and licence plates in frame: retake or crop before ingest.
