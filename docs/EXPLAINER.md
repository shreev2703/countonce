# CountOnce EXPLAINER

A study guide to how CountOnce works, written from the implemented code. Measured numbers are
copied from the report each one names; anything not measured says so. What has been run, component
by component, is in [STATUS.md](STATUS.md), which wins if this guide disagrees.

**The short version:** the pipeline runs end to end, and on held-out orchard trees its errors are
large. The "geometry beats simple baselines" gate (G3) was not met (recommendation; the owner's
decision is open in [GATES.md](GATES.md)). No own photos have been captured or hand-counted.

Module map: [`countonce/detect/`](../countonce/detect/), [`countonce/recon/`](../countonce/recon/),
[`countonce/lift.py`](../countonce/lift.py), [`countonce/merge/`](../countonce/merge/),
[`countonce/confidence.py`](../countonce/confidence.py),
[`countonce/warnings.py`](../countonce/warnings.py),
[`countonce/pipeline.py`](../countonce/pipeline.py), [`bench/`](../bench/).

## 1. The problem

- **Single-photo counting misses objects.** An open-vocabulary counter sees only what is visible
  in one image. Occluded or out-of-frame objects are never counted.
- **Multi-photo counting double-counts.** If you count every photo and add the counts, every
  object that appears in two photos is counted twice. Taking the max per-photo count instead
  under-counts, because no single photo sees everything. Both are implemented as baselines in
  [`bench/baselines.py`](../bench/baselines.py) (`sum`, `max`, `median`).
- **The fix.** Work out where each detection is in 3D, then decide which detections from different
  photos are the same physical object. The count is the number of distinct objects.
- **The hard constraint.** Two detections in the *same* photo are different objects, because one
  object cannot appear twice in one image. This is the cannot-link rule (§6).

## 2. Cameras 101

- **Intrinsics `K`** (3×3): focal lengths and principal point. They turn a camera-space point
  into pixels. [`lift.py`](../countonce/lift.py) uses the mean of `K[0,0]` and `K[1,1]` as the focal length.
- **Extrinsics `[R|t]`** (3×4, world-to-camera): `x_cam = R·x_world + t`. The camera centre in world
  coordinates is `-Rᵀt`, which is what `compute_scene_scale` and `recon/align.camera_centers` use.
- **Projection:** `uv = (K·x_cam)[:2] / z`. `_Views.project` in
  [`merge/match.py`](../countonce/merge/match.py) does this to ask "where would this 3D point show up in view *v*?".
- **Depth map:** per-pixel `z` along the optical axis. **Point map:** per-pixel 3D point in the world
  frame. Here `world_points` are computed by unprojecting depth with `K` and the extrinsics (not
  read from VGGT's point head), see `docs/DEVIATIONS.md`.
- **Relative scale.** VGGT does not know metres. A scene may be uniformly scaled and still be
  consistent, so every threshold in CountOnce is expressed relative to object size (§5).
- The dense container is `Recon` in [`countonce/types.py`](../countonce/types.py): `extrinsics`,
  `intrinsics`, `depth`, `depth_conf`, `world_points`, `world_points_conf`.

## 3. VGGT

[`recon/vggt.py`](../countonce/recon/vggt.py) wraps VGGT (ported from the author's VGGT
benchmark repo). What the code relies on:

- VGGT is a feed-forward transformer: images go in, and cameras, depth and point maps come out in
  one pass, with **no bundle adjustment** and no iterative optimization.
- Each image is split into patch tokens (the backbone uses DINO-style features). A learned camera
  token per frame is added. The network alternates **frame-wise attention** (tokens within one
  image) and **global attention** (tokens across all images), which is how views share geometry.
- Heads decode camera parameters (as a pose encoding, converted with `pose_encoding_to_extri_intri`),
  depth with confidence, and point maps. We use depth plus cameras to build points.
- **Memory grows with the number of frames** because global attention spans all tokens. The memory
  guard in [`recon/profiling.py`](../countonce/recon/profiling.py) calibrates a frame cap, and
  [`recon/chunked.py`](../countonce/recon/chunked.py) handles captures beyond it (§5).
- Inputs are loaded at width 518 with bf16 autocast (see the module docstring).
- **VGGT-Ω and VGG-T³:** accepted as `backbone` names in `Config`, but not implemented here;
  what they change is `TODO: verify` against the papers (see `docs/MODEL_SURVEY.md`).
- **COLMAP** ([`recon/colmap.py`](../countonce/recon/colmap.py)) is a classical SfM baseline. It
  yields sparse points only, so it cannot be a lifting backbone (`ReconAdaptError`).

## 4. Open-vocabulary counting

- The detector interface is [`detect/base.py`](../countonce/detect/base.py). A detector takes
  frames plus a text prompt (and optionally exemplar boxes) and returns scored boxes.
- **Text prompt:** "apple". **Exemplar prompt:** a box around one example, tied to a frame
  (`Exemplar(frame_id, box_xyxy)`); the default detector ignores exemplars. **Negative prompts**
  ("red apples, not leaves") are handled in the pipeline by dropping a box whose best negative
  phrase scores higher than its positive one; this is not CountGD++'s mechanism and its effect on
  accuracy is not measured.
- **Default:** GroundingDINO-tiny through Hugging Face `transformers`
  ([`detect/groundingdino.py`](../countonce/detect/groundingdino.py)), run on the tiny scene and on
  Fuji-SfM. Text prompts only; it reduces per-token logits to a per-query score.
- **CountGD / CountGD++** are *planned*: selecting them raises an error that says so
  ([`detect/countgd.py`](../countonce/detect/countgd.py) has a stub forward pass).
- The detection **cache** ([`detect/cache.py`](../countonce/detect/cache.py)) stores every raw
  detection above a low floor (0.05), keyed by frame hash, prompt, detector, version and config.
  The score threshold can then be swept (§8.8 confidence range) without re-running the model.

## 5. Lifting: from pixels to 3D

[`lift.py`](../countonce/lift.py), per detection:

1. **Region:** the mask if present, else the central 50% of the box (its middle half in each
   axis, to avoid background at the edges), else a small disk around a point.
2. **Filter:** drop non-finite points and anything under the confidence threshold. By default the
   bottom 30% by confidence within the region is discarded, and masked-out (privacy) pixels are removed.
3. **Median:** `xyz` is the per-axis median of the surviving world points. The median resists
   outliers at silhouette edges where depth bleeds between object and background.
4. **Size:** `size = extent_px × median_depth / focal`, i.e. the pinhole relation from pixels to
   scene units. It is in VGGT's arbitrary units, so only ratios mean anything.
5. **Failure is data:** a failed lift sets `fail_reason` (`low_conf`, `too_few_pixels`,
   `privacy_mask`) instead of raising. Those detections are reported as `rejected`.
6. `lift_conf` = detector score × a saturating function of mean depth confidence × a penalty for
   depth spread inside the region. It is a heuristic, not a probability.

**Relative scale.** Because there are no units, the v1 merge threshold `k` is in object sizes
("merge if centres are closer than half an object"), and `compute_scene_scale` normalizes by the median
camera-to-scene distance.

**Chunked reconstruction.** When a capture exceeds the frame cap, `ChunkedReconstructor` splits it
into overlapping chunks, fits a Sim(3) (scale, rotation, translation) on the shared frames' camera
centres with the Umeyama algorithm ([`recon/align.py`](../countonce/recon/align.py)), and merges.
Each chunk aligns to the already-merged result, so errors accumulate along the chain. That is the
"chunk drift" limitation.

## 6. v1: clustering and the cannot-link rule

[`merge/cluster.py`](../countonce/merge/cluster.py):

- Normalized pair distance: `‖xyz_i − xyz_j‖ / mean(size_i, size_j)`. Pairs under `k` (default 0.5)
  are merge candidates, processed closest first, with a deterministic tie-break on detection id.
- Merging is single-linkage with union-find.
- **Cannot-link:** a merge is refused if the two *clusters* share any frame, so same-frame
  detections stay apart even through a chain (A–B close, B–C close, but A and C in one photo).
  The reasons `same_frame_cannot_link` and `cluster_cannot_link` are logged as `MatchEvidence`.
- Each cluster becomes an `Instance`; the highest-`lift_conf` member is marked `new`, the others
  `duplicate`. Instance confidence is the mean lift confidence × `(1 − 0.5^n_views)`, a heuristic
  that rewards being seen from more views.
- Count = number of instances. The count is non-increasing in `k` (property-tested).
- Weakness: geometry only, so two touching objects closer than `k` object-sizes but in different
  photos can merge wrongly, and a noisy reconstruction can split one object.

## 7. v2: geometric cross-view matching

[`merge/match.py`](../countonce/merge/match.py) processes frames in increasing id, keeping running
instance tracks:

1. **Visibility:** project each track's 3D position into the current view. It is a candidate only
   if it lands inside the image and passes the **depth test** (projected depth within `depth_tol`
   object sizes of the reconstructed depth there). This stops matching to something occluded.
2. **Cost:** reprojection error divided by box size, plus an optional appearance term
   `weight × (1 − cos)/2` from [`merge/appearance.py`](../countonce/merge/appearance.py)
   (DINOv2 crop embeddings, pluggable `FeatureExtractor`).
3. **Gates:** reject if normalized reprojection error exceeds `max_reproj_norm`, or if appearance
   similarity falls under `min_appearance_sim`.
4. **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) per frame. One detection per
   track per frame makes cannot-link hold by construction. Unmatched detections start new tracks.
5. **Split check:** a track is recursively split if its members are spread wider than
   `split_extent_factor × size` or two members look too different.
6. **Hallucination filter:** an instance seen in only one view, but *expected* (visible by projection)
   in several other views where nothing matched, is probably a false positive. It is dropped or
   down-weighted (`hallucination: off | downweight | drop`).

**Photo order.** Greedy track building depends on the order frames are visited. v2 used to visit
them by frame id, so uploading the same photos in another order could change the count. It now
visits them in order of camera centre ([`merge/order.py`](../countonce/merge/order.py)). See
Q "Does photo order change the count?" in §13.

**Synchronization baseline** ([`merge/sync.py`](../countonce/merge/sync.py)): pairwise matches built
from v2's cost are noisy and not transitive. Spectral permutation synchronization (Pachauri et al.,
2013) takes the top eigenvectors of the block match matrix and rounds to a consistent partition,
with a per-view Hungarian step so same-view detections never merge. It is a baseline for the
learned matcher (v3, §8).

## 8. v3: learned matching

[`merge/v3.py`](../countonce/merge/v3.py), [`match/`](../countonce/match/). **Trained and tested on
synthetic geometry only.**

- One graph node per detection: 3D position, size, box, score, lift confidence, viewing ray, camera
  centre and log depth, all normalized by scene scale.
- Layers alternate **self-attention** (within a view) and **cross-attention** (across views), each
  with a learned bias from the relative 3D position of the two detections. No view index is
  embedded, so permuting views permutes the output.
- **Sinkhorn with a dustbin** ([`match/sinkhorn.py`](../countonce/match/sinkhorn.py)) turns pair
  scores into a soft assignment; the extra row and column absorb detections with no partner.
- **Loss** ([`match/train.py`](../countonce/match/train.py)): negative log-likelihood of the true
  assignment, plus a soft-count term and an optional cycle-consistency term (off in the first run).
- **Instance formation** ([`match/instances.py`](../countonce/match/instances.py)): threshold the
  probabilities, take connected components, and split any component holding two detections of one
  photo with a min-cut.
- **Result** ([RESULTS_V3_SYNTH.md](RESULTS_V3_SYNTH.md), 100 synthetic test episodes): association
  F1 0.964 against 0.741 for v2, and mean absolute count error 19.1 against 47.6. This is a
  component test: v1/v2 were not tuned for it and its depth is NaN, so v2's occlusion check never
  fires. It is not evidence about real photos, and v3 has no real-data result.

## 9. Synthetic data

[`synth/geometry.py`](../synth/geometry.py) is a seeded numpy generator, **not a renderer**: random
3D instances, random cameras, noisy 2D detections with drops and false positives, and identity
labels. Kubric or BlenderProc renders pushed through the real detector and VGGT are planned
(PLAN §10.2) and not done. Until then the sim-to-real gap is unmeasured.

## 10. Evaluation

- **Baselines** ([`bench/baselines.py`](../bench/baselines.py)): `sum`, `max`, `median` of per-image
  detection counts, computed from the *same* cached detections and score threshold as CountOnce,
  so the only difference is association. The COLMAP backbone is a second comparison.
- **Count metrics lead** ([`bench/metrics.py`](../bench/metrics.py)): MAE, signed bias, MAPE
  (undefined where the ground truth is 0), within-10%, interval coverage, and the **error
  decomposition**: `duplicates + false positives − missed − wrong merges = predicted − true`.
  Association F1 is reported only to explain a count error.
- **Association metrics:** detections are labelled with a ground-truth object id by projecting GT 3D
  points into each frame and taking the nearest within a pixel threshold
  ([`bench/gt_matching.py`](../bench/gt_matching.py)). From those labels: duplicate rate, miss rate,
  pairwise precision/recall/F1, cluster purity. Pairwise F1 ignores unlabelled detections, so an
  injected false positive does not lower it.
- **Failure-source harness** ([`bench/failure_sources.py`](../bench/failure_sources.py)): swap the
  detector or the geometry for an oracle, one at a time, with identical inputs for every associator.
- **Zero-count scenes:** false-positive rate, mean false count, rate at which the interval
  includes 0.
- **Statistics** ([`bench/stats.py`](../bench/stats.py)): the resampling unit is the *scene*, not the
  run. Bootstrap CIs, paired bootstrap and Wilcoxon for method comparisons.
- **Sweeps** ([`bench/sweep.py`](../bench/sweep.py), `configs/bench.yaml`): 3/5/10/20/40 views,
  random and evenly spaced draws, ordered vs. shuffled, one-factor ablations around a base method.
  Results are written as one CSV row per job; figures are rendered from the CSV
  ([`bench/figures.py`](../bench/figures.py)).
- **Error slicing** ([`bench/slices.py`](../bench/slices.py)): per-slice MAE and association F1 over
  scene covariates (density, views, depth confidence, blur, …) with bootstrap "significantly worse" flags.
- **Fair comparison:** same detector cache, same frames, same threshold; the grid and seeds are in
  the config; no result is committed from a run that did not happen.
### Results on Fuji-SfM

8 held-out test trees, thresholds frozen on training trees, mean absolute error in apples
([RESULTS_FUJI.md](RESULTS_FUJI.md), with CIs and paired tests):

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| v1 | 61.6 | 67.2 | 52.6 | 100.0 |
| v2 | 62.8 | 65.7 | 40.1 | 101.8 |
| Sum | 61.0 | 68.2 | 68.2 | 157.4 |
| Max | 66.2 | 60.1 | 31.1 | 153.8 |
| Median | 72.6 | 57.9 | 43.8 | 185.0 |

- G3 ("v1/v2 beat Sum and Max on most scenes") holds only at k=20. At k=10, Max is best.
- **Failure sources** (k=10): with the real detector, about 121 false positives and 52 misses per
  window; with perfect detections, about 34–37 duplicates remain. So the detector is the largest
  source, then duplicates from association under estimated geometry. Fuji has no oracle geometry,
  so the geometry share is **not** separated from the associator's. Some "false positives" may be
  real apples missing from the ground truth.

### Failure cases and causes

- **Tiny scene, 7 instead of 5** ([DEVIATIONS.md](DEVIATIONS.md)): the near-duplicate filter kept
  1 of 5 views, and blue and green distractors scored above the 0.3 detector threshold. A 0.5
  threshold chosen on this scene gives 5.
- **Tiny scene, all five views: 14 (v2):** VGGT recovers poor cameras on this small, flat-shaded,
  symmetric scene, so views of one sphere do not land close enough to merge.
- **Many short Fuji windows have no visible ground-truth apple** (7 of 16 at k=3), which makes MAPE
  very unstable there.
- Not yet tested by results: over-merging of adjacent objects, chunk drift, the hallucination
  filter's effect, moving objects.

## 11. Inference optimization

[PROFILE.md](PROFILE.md), RTX 5070 Laptop GPU (8 GB), a small synthetic profiling scene:

- In the baseline profile, VGGT reconstruction was 88.7% of wall time at 10 frames.
- **bf16** weights: about the same speed (1.04x), but peak VRAM 7298 → 3833 MB.
- `torch.compile` (default mode) was slower (0.87x); `reduce-overhead` crashed with a CUDA error.
- **Stage caching:** a new prompt on the same photos is 4.8x faster (reconstruction reused); an
  exact repeat 42x. A first-time job gets nothing from it.
- Confidence stage: 2.5 s → 1.05 s by lifting each detection once and memoising sweep cells.
- Matcher training: 703 s → 85 s per epoch by batching all view pairs into one padded Sinkhorn.

PLAN's ≥1.5x end-to-end speedup on a first-time job has **not** been reached.

## 12. System design

- **Web** ([`web/`](../web/)): Next.js. Demo pages are static files and need no backend. Signed-in
  pages call the API through a server-side proxy, so the browser never holds an API key.
- **API** ([`countonce/api/`](../countonce/api/)): FastAPI; imports no torch. Long work never runs
  in a request. Progress over SSE, capped for serverless hosts, with polling as the fallback.
- **Queue** ([`countonce/jobs/`](../countonce/jobs/)): a database table. Postgres claims with
  `FOR UPDATE SKIP LOCKED`; sqlite stands in locally. Idempotency keys stop double submits.
- **Worker:** heartbeats every 15 s; a job silent for 120 s is requeued, at most 2 tries. Reported
  retryable errors back off exponentially, then go to a dead-letter state an operator can retry.
  Jobs older than 30 days are purged by the worker loop.
- **Storage:** local disk or S3/R2 behind one `BlobStore` interface.
- **Not verified:** no hosted deployment, no real Google sign-in, no run from browser to real GPU
  worker, never run against real R2.

## 13. Interview Q&A

### Method

**1. Why lift to 3D instead of tracking through video? When does tracking win?**
Lifting works for unordered photos and wide baselines, where there is no temporal continuity to
track. Tracking should win on dense, short-baseline video where objects move, because the
static-scene assumption fails there and 2D motion is a strong cue. Neither claim has been measured
against a tracker here.

**2. How is this different from CountVid and DG-Net?**
Both count in video by tracking: CountVid propagates SAM 2.1 masks; DG-Net adds depth (sensor, or
estimated in some of its results) to detection and to the matching cost, so "RGB only" is not a
difference. CountOnce targets a static scene from a few unordered photos, associates through
estimated camera geometry, and keeps the evidence inspectable (`MatchEvidence` per pair). Whether
that counts better is a hypothesis: neither system has been run here. See `RELATED_WORK.md`.

**3. Why can't same-photo detections merge?**
One object projects to one region of an image, so two separate detections in one photo are two
objects. v1 enforces it per cluster, so it holds transitively; v2 and sync by one detection per
track per frame; v3 by min-cut. A detector that boxes one object twice breaks the assumption, which
per-frame NMS (`nms_iou`, off by default) addresses.

**4. How do you set thresholds when VGGT has no real units?**
Make them relative: v1's `k` is in object sizes (`pixel extent × depth / focal`), v2's reprojection
error is divided by box size. Then tune on a validation split. For Fuji the score threshold and `k`
were frozen on training trees (0.2 and 0.3). 0.2 is the lowest value in that grid, so the grid may
be too narrow.

**5. Why use the median of world points for a detection's position?**
Boxes include background and edge pixels where depth is smeared. The median resists those
outliers; the central-50% region and the confidence filter reduce them further. It still gives a
point on the visible surface, not the object's centre, so views from opposite sides disagree.

**6. How does v2 avoid matching to an occluded object?**
A depth test: the track's projected depth must be within `depth_tol` object sizes of the
reconstructed depth at that pixel, otherwise the track is not a candidate in that view.

**7. What is the hallucination filter and when is it wrong?**
An instance seen in one view but expected in at least two others where nothing matched is probably a
false detection. It is wrong when the other views' depth or cameras are bad. Its effect on real
data has not been measured.

**8. Why does the count never increase when `k` increases in v1?**
A larger `k` admits a superset of merge candidates, and merging only reduces clusters. A Hypothesis
test checks it.

### Findings (M2–M8)

**9. Your demo scene had 5 objects and the real pipeline said 7. Why?**
I checked each stage. The near-duplicate filter had kept 1 of 5 views (the views are 72 degrees
apart but look almost identical to a thumbnail metric), and blue and green distractors scored
0.40–0.50 as "red sphere", above the 0.3 threshold: 5 + 2 = 7. Duplicate boxes, lifting and merge
radius were not the cause of that number.

**10. Did you fix it?**
Only narrowly. `score_threshold=0.5` gives 5 on the single-view path, but it is a threshold picked on
that scene, so defaults are unchanged and the default run still says 7. With all five views the
count gets worse (v2: 14) because VGGT's cameras are poor on this synthetic scene. The pipeline now
warns when the filter keeps one frame.

**11. Why isn't association F1 your headline metric?**
The product is a count. v3 has F1 0.964 on synthetic episodes and still a mean count error of 19.1.
Pairwise F1 also ignores unlabelled detections, so false positives cost nothing. Reports lead with
MAE, signed bias and the error decomposition.

**12. What is the error decomposition?**
`duplicates + false positives − missed − wrong merges = predicted − true`, an exact identity on
scenes with identity labels. It says which way each stage pushes the count, and shows when two large
errors cancel into a good-looking number.

**13. Does photo order change the count?**
It did. My first test shuffled input lists but kept frame ids, so it passed without testing
anything. A real test permutes which photo gets which id, with its camera arrays: v2 partitions
changed in 38 of 40 runs at moderate box noise, with a count range of 3. Visiting frames in
camera-centre order fixed it (0 changes for v1, v2, sync, v3, v3+sync). Scope: synthetic scenes;
duplicated cameras still fall back to id; not yet measured on real photos.

**14. Does the canonical order make v2 better?**
No claim. It makes the result a function of the photos. The greedy matching is still not optimal,
and quality against the old order was not compared.

**15. How do you know whether detection, geometry or association is to blame?**
Replace one stage at a time with an oracle and give every associator identical inputs (checked by
an input hash). On Fuji at k=10 the real detector brings about 121 false positives and 52 misses per
window; with perfect detections about 34–37 duplicates remain.

**16. What can't that experiment tell you?**
Fuji ships no per-pixel depth or camera poses for these photos, so there is no oracle-geometry
condition: bad cameras and bad association are not separated. Oracle boxes also assume a 4 cm apple
radius, and its predictions are not ROI-filtered, so its MAE differs from the main table.

**17. Does your method beat the simple baselines?**
Not consistently. v1/v2 beat Max in most slices only at k=20 (v2 vs Max MAE difference −51.9, CI
[−81.7, −20.3]). At k=10 Max has the lowest MAE (31.1 against v2's 40.1), and at k=3 and 5 the
methods are within noise. The gate's recommendation is "not met".

**18. Why are the Fuji errors so large?**
Detection first: at score 0.2 most visible ground-truth apples are still missed at k=20 (miss rate
about 0.95) while many boxes match nothing. Then duplicates. Foliage occlusion is not scored
separately, so I can't say how much it contributes.

**19. How did you keep the Fuji numbers honest?**
Thresholds frozen on training trees first, failed rows kept (none failed in 320), CIs resampled over
the 8 trees rather than 16 slices, cells left unfilled where identity labels are incomplete.
Weaknesses: the runs came from a working tree with uncommitted changes, and the test trees had
already been run once before scoring bugs were found and the protocol was revised (`fuji_core_v2`,
re-tuned on training trees, same thresholds). So they are not untouched.

**20. Which Fuji baselines are missing?**
COLMAP, permutation sync, CountVid, a vision-language model, and v3.

**21. What changed between your first v3 comparison and an honest one?**
Nothing in the numbers yet. The upgrade (tune v1/v2/sync on validation, finite noisy depth so the
occlusion test fires, count injected false positives) has code in `bench/synth_component.py`, but
the table was **not** regenerated. So the v3 result stays a component test with untuned baselines.

**22. What does the dustbin do in Sinkhorn matching?**
An extra row and column absorb detections with no match (false positives, objects seen in one view),
so the assignment need not be one-to-one. v2's analogue: unmatched detections start new tracks.

**23. Why train on synthetic data? How did you check it transfers?**
Identity labels across views are expensive to annotate and free to generate. I have **not** checked
transfer: training and testing were both on numpy geometry episodes.

### Evaluation, systems and honesty

**24. Why is the count's range a "sensitivity range" and not a confidence interval?**
It is the 10th–90th percentile of counts over a grid of `k` and detector thresholds, widened to
include the reported count. It shows fragility to tuning and has no coverage guarantee. Coverage has
not been evaluated.

**25. When would you use conformal intervals?**
Split conformal and Jackknife+ are implemented but need real calibration scenes (at least 15 for
split conformal), and coverage is average over scenes like the calibration set. I have none yet.

**26. What's the biggest weakness of your benchmark?**
One fruit, 8 test trees, no own captures, no zero-target scenes, missing baselines, and ground truth
that depends on projecting 3D labels into frames with a visibility test.

**27. Which optimization gave the biggest speedup, and why?**
For repeat jobs, stage caching (4.8x for a new prompt) because VGGT dominates and its output does
not depend on the prompt. For a first-time job nothing helped much; bf16 mainly halved peak VRAM.
For matcher training, batching Sinkhorn (703 s to 85 s per epoch) removed thousands of tiny GPU
launches.

**28. What happens if a worker dies mid-job?**
Its heartbeats stop; after 120 s the job is requeued, and after 2 tries marked failed. Cache writes
go to a temp file and rename, so a crash leaves no half-written entry.

**29. How would you scale to 1,000 jobs a day?**
That averages under one job a minute, but arrivals are bursty and one laptop GPU is the bottleneck.
Add workers (the `SKIP LOCKED` claim supports it), keep caches in shared storage, move the database
off sqlite. I load-tested the API with fake workers only (SERVING.md), so I have no real GPU
throughput or cost figure.

**30. What is a known bug right now?**
On WSL2 with an 8 GB card, the five-view CLI run fails with `CUDA driver error: device not ready`
unless `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is set. I suspect the allocator setting
in `countonce/oom.py`; it is not fixed or root-caused.

**31. How does the pipeline avoid recomputing expensive stages?**
The reconstruction cache is keyed by input hash, backbone, version and config; the detection cache by
frame hash, prompt, detector, version and config.

**32. How do you test geometry code without a GPU?**
Synthetic `Recon` objects with known cameras and fake detectors. Real weights appear only in
GPU-marked tests. A golden test pins a seeded synthetic count to catch regressions; it says nothing
about accuracy.

**33. What should you not claim from this repo?**
That it counts accurately on real photos; that geometry beats simple baselines; that the learned
matcher transfers to real data; that any user has used it.

## 14. Glossary

- **Association:** deciding which detections in different photos are the same object.
- **Cannot-link:** constraint that two detections from one photo are never the same object.
- **Extrinsics / intrinsics:** camera pose / camera internals (§2).
- **Depth test:** check that a projected 3D point is not hidden behind the reconstructed surface.
- **Hungarian algorithm:** optimal one-to-one assignment minimizing total cost.
- **Instance:** a cluster of detections believed to be one physical object; the count is the number of instances.
- **Lifting:** turning a 2D detection into a 3D position and size.
- **MAPE:** mean absolute percentage error (undefined when ground truth is 0).
- **Point map:** per-pixel 3D world coordinates.
- **Relative scale:** reconstruction defined only up to a global scale factor.
- **Sensitivity range:** spread of counts over a grid of merge/threshold settings; not a statistical interval.
- **Sim(3):** similarity transform (rotation, translation, uniform scale), used to align chunks.
- **Spectral synchronization:** rounding top eigenvectors of a match matrix into a consistent partition.
- **Umeyama:** closed-form least-squares Sim(3) between two matched point sets.
- **VGGT:** Visual Geometry Grounded Transformer, a feed-forward multi-view reconstruction model.
