# CountOnce EXPLAINER (draft)

A study guide to how CountOnce works, written from the implemented code. Sections 8, 9, 11 and
12 are stubs for work that does not exist yet. Every result is `TODO: fill from results`; this
guide states no measured numbers.

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
  (`Exemplar(frame_id, box_xyxy)`). **Negative prompts** (things *not* to count) are a CountGD++ idea
  and are not implemented here.
- [`detect/countgd.py`](../countonce/detect/countgd.py): adapter around CountGD; its forward pass is a
  stub until weights exist. [`detect/groundingdino.py`](../countonce/detect/groundingdino.py): adapter
  around GroundingDINO (never run here). Both reduce per-token logits to a per-query score.
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

Output is independent of input order of views and detections (frames and ids are sorted).

**Synchronization baseline** ([`merge/sync.py`](../countonce/merge/sync.py)): pairwise matches built
from v2's cost are noisy and not transitive. Spectral permutation synchronization (Pachauri et al.,
2013) takes the top eigenvectors of the block match matrix and rounds to a consistent partition,
with a per-view Hungarian step so same-view detections never merge. It is a baseline for the
learned matcher (v3, not implemented).

## 8. v3: learned matching — stub

`TODO` after Track C: graph neural network, self/cross attention, Sinkhorn with a dustbin, training
loss, instance formation. [`countonce/match/`](../countonce/match/) is an empty package today.

## 9. Synthetic data — stub

`TODO` after Track C: Kubric, domain randomization, running the real pipeline on synthetic images,
the sim-to-real gap.

## 10. Evaluation

- **Baselines** ([`bench/baselines.py`](../bench/baselines.py)): `sum`, `max`, `median` of per-image
  detection counts, computed from the *same* cached detections and score threshold as CountOnce,
  so the only difference is association. The COLMAP backbone is a second comparison.
- **Count metrics** ([`bench/metrics.py`](../bench/metrics.py)): MAE, MAPE (undefined where the
  ground truth is 0), within-10%, interval coverage.
- **Association metrics:** detections are labelled with a ground-truth object id by projecting GT 3D
  points into each frame and taking the nearest within a pixel threshold
  ([`bench/gt_matching.py`](../bench/gt_matching.py)). From those labels: duplicate rate, miss rate,
  pairwise precision/recall/F1, cluster purity.
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
- **Results:** `TODO: fill from results`.
- **Failure cases and causes.** Expected failure modes from the design (to be confirmed or
  refuted by results, each with the evidence slice that tests it):
  - *Over-merging* of adjacent objects in v1 when `k` is large: duplicate-rate vs. `merge_k` ablation.
  - *Splitting* of one object when the reconstruction is poor: depth-confidence slice.
  - *Chunk drift* on long captures: error vs. chunk count.
  - *Detector hallucinations* that survive because they appear in one view: hallucination filter on/off.
  - *Moving objects:* not handled.
  - Observed failures: `TODO: fill from results`.

## 11. Inference optimization — stub

`TODO` (Track F): profiling, SDPA/FlashAttention, bf16, compile/TensorRT, what worked. Only the
memory guard and the recon/detection caches exist today.

## 12. System design — stub

`TODO` (Track G): queue with `SKIP LOCKED`, heartbeats, retries, SSE, signed URLs, scale-to-zero
workers. No web product exists in this repo yet.

## 13. Interview Q&A

**1. Why lift to 3D instead of tracking through video? When does tracking win?**
3D lifting works for unordered photos and wide baselines, where there is no temporal continuity to
track. Association depends only on geometry, so shuffled photos give the same answer (the bench has an
`ordered` vs `shuffled` axis to test that). Tracking wins on dense, short-baseline video where objects
move, since the static-scene assumption fails there and 2D motion is a strong cue.

**2. How is this different from CountVid and DG-Net?**
Those target counting in video with temporal tracking. CountOnce targets a static scene from a few
photos and makes the geometric association explicit and inspectable (`MatchEvidence` per pair).
The related-work comparison is in `docs/RELATED_WORK.md`, written from abstracts with `TODO: verify` claims.

**3. Why can't same-photo detections merge?**
One physical object projects to one region of an image, so two separate detections in a single photo
are two objects. v1 enforces it at the cluster level (`cluster_cannot_link`), so it holds even
transitively. v2 and sync enforce it by construction, with one detection per track per frame.

**4. How do you set thresholds when VGGT has no real units?**
Make them relative. v1's `k` is in object sizes, where size is `pixel extent × depth / focal`, and
v2's reprojection error is divided by box size. The scene scale is the median camera-to-scene
distance. Absolute defaults (`k = 0.5`) are a starting point to be tuned on a validation split.

**5. Why train on synthetic data? How did you check it transfers?**
Not done yet (Track C). The plan is to run the real pipeline on Kubric renders so the learned matcher
sees real reconstruction noise. Transfer would be checked by evaluating on real scenes. Results: `TODO: fill from results`.

**6. What does the dustbin do in Sinkhorn matching?**
Not implemented yet. In general it is an extra row and column that absorbs detections with no match
(false positives, objects visible in one view only), so the assignment need not be one-to-one.
v2 has an analogue: unmatched detections start new tracks, and gated pairs have infinite cost.

**7. What's the biggest weakness of your benchmark?**
It has no results yet. Structurally: few scenes (the resampling unit is the scene, so CIs are wide),
Fuji-SfM covers one fruit type and its license is unresolved, ground truth depends on GT 3D positions
projected into frames, and the detector cache means detector errors are shared by all methods.

**8. Which optimization gave the biggest speedup, and why?**
None measured. Implemented so far: the reconstruction cache (keyed by input hash, backbone, version,
config) and the detection cache, which let threshold and `k` sweeps skip the model. Profiling: `TODO: fill from results`.

**9. What happens if a worker dies mid-job?**
There is no worker or queue in this repo. Locally, cache writes use a temp file plus rename
(`tempfile`/`os` in the cache modules), so a crash leaves no half-written entry, and corrupt entries are
treated as cache misses. The queue design is §12, `TODO`.

**10. How would you scale to 1,000 jobs a day?**
`TODO` (§12). What the code already supports is stage caching and a frame cap with chunking to bound
per-job VRAM. Throughput numbers: `TODO: fill from results`.

**11. Why is the count's range called a "sensitivity range" and not a confidence interval?**
It is the 10th–90th percentile of counts over a grid of `k` and detector thresholds
([`confidence.py`](../countonce/confidence.py)), widened to include the reported count. It measures
how fragile the answer is to tuning. It has no coverage guarantee. `Interval.method` is `"sensitivity"`
and `level` is nominal. Empirical coverage is measured in the bench (`interval_coverage`) and is `TODO: fill from results`.

**12. Why use the median of world points for a detection's position?**
Boxes include background and edge pixels where predicted depth is smeared. The median is robust to
those outliers, and the central-50% box region plus the confidence filter further reduce them.

**13. Why does the count never increase when `k` increases in v1?**
A larger `k` admits a superset of merge candidates, and merging only reduces the number of clusters. The
code is order-independent (tie-break on ids), and a Hypothesis test checks monotonicity.

**14. How does v2 avoid matching to an occluded object?**
The depth test in `_Views.project`: the track's projected depth must be within `depth_tol` object
sizes of the reconstructed depth at that pixel. Otherwise the track is not a candidate in that view.

**15. What is the hallucination filter and when is it wrong?**
An instance seen in one view but expected (visible by projection) in at least two others where nothing
matched is probably a false detection. Wrong when the other views' depth or cameras are bad, or when
the object is truly hidden but the depth map says it should be visible.

**16. Why Hungarian per frame instead of global clustering?**
It gives optimal one-to-one assignment per frame, and one-to-one is exactly the cannot-link rule.
The cost is order dependence in principle; processing frames in sorted id order and sorting
detections by id makes the output deterministic. Sync is the alternative that treats all views jointly.

**17. What does spectral synchronization buy over v2?**
Transitive consistency. Pairwise matches can say A=B, B=C, A≠C. Synchronization finds a global
partition close to the pairwise evidence. Whether it improves counting is `TODO: fill from results`.

**18. What does the chunk stitcher need, and how does it fail?**
At least two shared frames with non-degenerate camera centres (`MIN_OVERLAP = 2`; with two, axis
tips are added). Too little overlap raises `ChunkAlignmentError`. Success still accumulates drift
because each chunk aligns to the previous merged result.

**19. How does the pipeline avoid recomputing expensive stages?**
The recon cache key includes input hash, backbone, version and config, and the detection cache key
includes frame hash, prompt, detector, version and config. `CountResult` records `recon_cache_key`
and whether it hit.

**20. What does a quality warning look like and why?**
A `{code, message, fix}` dict ([`warnings.py`](../countonce/warnings.py)): too few frames, low overlap
(few instances seen in ≥ 2 views), low median depth confidence, many failed lifts, or a wide range.
Each includes a concrete fix such as "move less between photos". Thresholds are uncalibrated defaults.

**21. How are results kept reproducible?**
`CountResult` carries config, model versions and cache keys, and the bench writes one CSV row per
(scene, views, sampling, order, method) with a seeded view sampler. Provenance helpers live in
[`provenance.py`](../countonce/provenance.py).

**22. How do you test geometry code without a GPU?**
Tests build synthetic `Recon` objects (planes, known cameras, known Sim(3) per chunk) and fake
detectors, so CPU tests check clustering, matching, chunk alignment and the end-to-end pipeline. Real
VGGT and detector weights are never used in tests; marks `slow` and `gpu` are reserved for that.

**23. What should you not claim from the current repo?**
Any accuracy figure. Only synthetic and toy inputs have been run. The golden test pins a seeded
synthetic scene's count; it checks for regressions and says nothing about real-world accuracy.

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
