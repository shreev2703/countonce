# Related work, organised around the §2.4 hypothesis

Last updated 2026-11-26 (task: remaining related work). Builds on the 2026-11-25 rewrite (task M10).

## 0. How to read this file

| Tag | Meaning |
|---|---|
| **[full]** | Full paper text read, not just the abstract (CountVid, DG-Net, OneScene-Bench, and §3 works). Each §3 entry has a **Coverage** line naming any part that was not read verbatim; a statement about such a part is marked as unverified |
| **[abs]** | Only the arXiv abstract was read. Only what the abstract says is repeated; nothing is said about what the work lacks |
| **[grep]** | Abstract plus targeted text searches of the PDF for the specific points stated |
| **[repo]** | Read from the public repository or its GitHub metadata on 2026-11-25 |

Statements of the form "X is not used" appear only for **[full]** papers and describe the paper version listed
in Sources, not later code. Numbers are copied from the papers and were **not reproduced** here. None of the
comparator systems has been run by CountOnce.

## 1. The hypothesis (PLAN §2.4)

> Geometry-aware association improves unique-object counting from sparse views compared with video tracking and
> with simpler geometric matching.

This is a hypothesis, not a result, and no novelty claim is made until it has been tested. It has two halves:

| Half | Comparators | Status |
|---|---|---|
| **H-a:** better than video tracking | CountVid (SAM 2.1 tracking), DG-Track (DG-Net) | **Untested.** Neither has been run |
| **H-b:** better than simpler geometric matching | v1 3D clustering, C-DOG-style epipolar grouping | **Partly measured, mixed** (§1.1). C-DOG has not been run |

For H-b we read "geometry-aware association" as v2, synchronization and v3, and "simpler" as v1 and C-DOG
(see DEVIATIONS).

### 1.1 Evidence so far, from this repository

- **Fuji-SfM (real detector, VGGT geometry):** gate G3 is recommended *not met (mixed)*. v2 beats v1 on count
  MAE only at k=10, and the per-image Max baseline has the lowest MAE at k=10. With oracle detections the
  remaining error is mostly duplicates left by association under estimated geometry. See
  `RESULTS_FUJI_ANALYSIS.md`.
- **Synthetic component test:** v3 is ahead of v1 and v2 (`RESULTS_V3_SYNTH.md`). This is synthetic only, and
  the tuned-baseline re-run (task M5) is not complete, so it does not support H-b on real data.
- **Against video tracking:** no evidence either way.

## 2. The three closest works (read in full)

### 2.1 CountVid **[full]**
Amini-Naieni and Zisserman, *Open-World Object Counting in Videos*, AAAI 2026, arXiv 2506.15368v2.

**Task and input.** A video plus text and/or exemplar boxes; output is per-frame counts and the number of unique
instances in the video.

**Method.** Frames are subsampled (3 fps; 6-7 fps for the crystal videos). (1) CountGD-Box, their extension of
CountGD that also outputs boxes, detects on each frame and SAM 2.1 turns boxes into masks. (2) A temporal filter
tracks each detection forwards and backwards over a window (w=3) and drops detections that do not persist.
(3) SAM 2.1 propagates masks through the video; per-frame masks that overlap no existing masklet start a new
object. The count is the number of masklets. The authors state the framework is agnostic to the choice of counter
and tracker, and also report GeCo, SAM 2 and ByteTrack variants.

**Association signal.** Mask propagation and mask overlap (IoU > 0.5) in image space. The method as described
uses no camera poses, depth or 3D reconstruction.

**Data.** VideoCount: 370 videos (TAO-Count, MOT20-Count, Science-Count) with 1-1203 objects per video.

**Points that affect our comparison.**
- Frame rate is not monotonic. In their Table 9 (text only), Penguins MAE is 0.7 at 1 fps against 4.0 at 3 and
  6 fps, while Crystals is slightly worse at 1 fps than at 3 fps. So **it cannot be assumed that CountVid gets
  worse with fewer frames**; the sparse-view sweep has to measure it. This is a small study (3 penguin and
  7 crystal videos) on continuous footage, not on unordered wide-baseline photos.
- The authors name false negatives from the counter and tracker re-identification as error sources, with more of
  both under heavy occlusion and similar-looking instances. SAM 2 treats objects independently, so masks can
  overlap.
- Their temporal filter needs an object to persist across consecutive frames. With very few views it may remove
  true objects (our inference, not tested in the paper). Run CountVid both with and without it.
- On FSCD-147 (their Table 3, text prompt) Grounding DINO's counting MAE is far higher than CountGD-Box's
  (54.16 vs 15.01). CountOnce's current default detector is Grounding DINO, so an end-to-end comparison mixes
  detector quality with association quality.
- Reported cost: 17 GB of GPU memory per stage on a 24 GB RTX 3090 for a 30 s video of 39 penguins. Our laptop
  GPU has less, so batch sizes must be reduced; whether it fits has **not been measured**: CountVid has not been run here.
- Code is MIT **[repo]**.

### 2.2 DG-Net (DG-Det + DG-Track) **[full]**
Xu et al., *Depth-Guided Video Object Counting in Crowded Scenes*, ACM Multimedia 2026, arXiv 2608.06236v1.
The paper names the detector DG-Det and the tracking stage DG-Track; the repository is `streamer-AP/DG-Net`
and its README calls the framework Depth-VIC.

**Task and input.** A synchronized RGB-D video, a text query, a visual exemplar and a video-level region of
interest (ROI); output is the number of individuals inside the ROI.

**Method.** DG-Det is initialised from CountGD-Box with the backbone frozen, and adds a depth encoder, multi-scale
RGB-D cross-attention with a depth-affinity bias, and an occlusion head. DG-Track matches tracks to detections by
Hungarian assignment on a cost made of 2D box-centre distance, appearance cosine distance, centre-depth
difference and an IoU gate (their Eq. 8), then keeps tracks by an occlusion-adaptive temporal vote.

**Association signal.** Per-detection depth is part of the cost, together with image-plane position and
appearance. The cost as written has no camera pose or world-coordinate term.

**Sensor depth is not the only mode.** Rows marked † in their Table 2 use *estimated* depth. Test MAE, text +
exemplar, as reported: CountVid 20.98, DG-Net with estimated depth 13.63, DG-Net with sensor depth (RealSense
D455) 7.97. Text only: 23.69, 13.75, 12.68. Their robustness study shows added depth noise raising error, and
their failure analysis says both sensor and predicted depth break down on glass and reflective scenes. On
FSCD-147 with Depth Anything V2 pseudo-depth they report no substantial gain.

**Data.** RGBD-VideoCount: 195 clips, 6 categories, 2,032 annotated frames, 77,638 boxes (CC BY 4.0 per the
Hugging Face card, see DATASETS). DG-Det is trained on this dataset **[repo]**; the paper does not say that the
RGB baselines were fine-tuned on it, so part of the gap to CountVid may be training data.

**Consequences for us.**
- "CountOnce needs no depth sensor" does **not** by itself separate it from DG-Net. The same-input comparison is
  CountOnce (RGB) against DG-Net with estimated depth; DG-Net with sensor depth is an upper reference with an
  extra sensor.
- Their ROI is an input. CountOnce must apply the same ROI, or the comparison is not like for like.
- Code and checkpoints are released, MIT **[repo]**.

### 2.3 OneScene-Bench **[full]**
Suhas et al., *OneScene-Bench: Do Vision-Language Models Integrate Evidence Across Views?*, accepted at a
NeurIPS 2026 **workshop** (arXiv comment: LM4RWD; PDF header: VLM4RWD), arXiv 2610.38926v1.

**What it is.** A benchmark for vision-language models, built from DL3DV-10K scenes and manually verified:
1,300 two-view questions, of which 700 are unique-object counting (200 complete overlap, 300 subset overlap,
200 complementary views), 300 cross-view consistency and 300 spatial reasoning. A multi-view extension (2-6
views) covers only the complete-overlap counting questions.

**Reported results.** 12 VLMs. Counting accuracy is 45.5-81.5% on complete overlap, 26.7-76.7% on subset overlap
and 4.5-50.0% on complementary views. Joint predictions agree more often with the maximum of the per-view counts
than with their sum. Free-form answers are scored by a GPT-4o judge.

**What it does and does not tell us.**
- It evaluates VLMs. It runs no video tracker or geometric associator, so it is **not evidence for H-a or H-b**.
- In complete and subset overlap, the per-view Max already gives the answer. Only the **complementary-view**
  split tests association, so report the three splits separately and always show our Max baseline beside them.
- Core questions have two views, so it says nothing about how error changes with view count under partial overlap.
- Our integer outputs would be scored by exact match; their scores come from an LLM judge. State this when the
  numbers are put side by side.

**Availability [repo].** `swathisuhas/OneScene-Bench` has parquet files with `image1`, `image2`, `object_name`,
`answer`. In the three files inspected (complete, complementary, consistency; first row), the image fields held a
relative path and no bytes, and the repository file tree has no image files. GitHub reports no licence; the README
defers to the DL3DV-10K terms. Until the images are obtained, this evaluation cannot run.

## 3. Open-vocabulary and multi-view 3D detection, and the matcher and grouping precedents **[full]**

These are the works a reviewer is most likely to raise: their output boxes can be counted, and some of them
merge boxes across views. Section, table and equation numbers refer to the arXiv HTML version (v1 unless
stated). Numbers are copied, not reproduced. None of these works reports a unique-object counting metric
(every one evaluates boxes by mAP/AP/AR, or matches and groups by F1), so none is evidence for or against H-a
or H-b.

Summary of inputs and outputs:

| Work | Poses needed? | Depth needed? | Ordered video? | Output | Per-instance identity / count | Data |
|---|---|---|---|---|---|---|
| See and Detect | Yes, posed RGB at inference (§4.2) | Training only, for pseudo-boxes (§3.2) | No (graph over unordered boxes) | Class-agnostic 3D boxes + open-vocab labels | No identities; no count metric | ScanNet, ARKitScenes |
| SMOV3D | Yes (§3, §5) | Predicted by MoGe (§3.1) | No ("collection" of views) | 3D boxes per prompt | No identities; no count metric | ScanNet, Replica, OpenSUN3D |
| OpenM3D | Yes at inference (§4); RGB-D + mesh at training | Training only | Chronological frame selection in preprocessing (App. B.1) | Axis-aligned 3D boxes + labels | No identities; no count metric | ScanNet200, ARKitScenes |
| VGGT-Det | **No** (Abstract, §3.1) | No | Not stated | 256 query boxes + class (Eq. 5, §4.1) | No identities; no count metric | ScanNet, ARKitScenes |
| ROM | No (§III) | No | Pairs of images | Pairwise object matches with dustbin (§III-B) | Pairwise only; no count | Hypersim (synthetic) |
| C-DOG | Yes, calibrated K, R, T (§IV-A) | No | No (all view pairs, Alg. 1) | Groups of 2D points = one 3D point each (§IV-A) | Groups, count implicit in group number; no count metric | Synthetic only (§IV-G, §V-A) |

### 3.1 See and Detect **[full]**
Xu et al., *See and Detect: Image-Driven Open-Vocabulary Multi-View 3D Object Detection*, arXiv 2611.17897v1
(13 Nov 2026); no venue stated in the text. **Coverage:** HTML read to the end including references; no
limitations section appears.

- **Input.** Inference uses "only the feed-forward multi-view 3D detector with RGB images and camera
  parameters" (§4.2). Pseudo-boxes for training come from Cubify Anything run per view on RGB plus a depth map
  (§3.2), with SAM and Qwen3-VL used offline (§4.2). The text does not say whether the depth is ground truth or
  estimated.
- **Output.** Class-agnostic 3D boxes labelled by cosine similarity to CLIP text embeddings (§4.2). No instance
  IDs and no counting metric.
- **Cross-view step (§3.2, Eq. 1).** Training-time only: one graph node per single-view 3D box, an edge when 3D
  IoU > θ = 0.4, connected components are candidate objects, components seen in one view are dropped, and the
  rest are fused. Table 4 (ScanNet200) shows dropping singletons raises pseudo-box precision at 0.25 IoU from
  34.1 to 50.5 at some recall cost. This is association by 3D IoU over unordered posed views, the same kind of
  signal as our v1.
- **Data.** ScanNet (1,201 train / 312 test scenes) and ARKitScenes; mAP25/mAP50 (§4.1, Tables 1-3). 100 views at
  test, 40 in training (§4.2).
- **Closest overlap with H-b.** Posed multi-view geometric association with view-count support. It is a
  training-label generator, not a counting method.
- **Not in the paper.** Any counting or duplicate-rate metric; any sparse-view sweep (the only view-related
  ablation is the minimum-view filter, Table 4); any comparison with video tracking or with a simple matcher on
  association quality. Code: none stated.

### 3.2 Sparse Multiview Open-Vocabulary 3D Detection (SMOV3D) **[full]**
Moliner, Larsson and Åström, arXiv 2509.15924v1 (19 Sep 2025); no venue stated. **Coverage:** HTML read verbatim
through the start of §6 (abstract, §1-§5, Tables 1-5); the end of §6 and the references came back only as a
tool-written summary and are not relied on. A direct PDF fetch failed, so this is not a page-checked read.

- **Input.** Posed RGB images with intrinsics (§3) and a text prompt. Depth is predicted by MoGe (§3.1). Up to 32
  random views per ScanNet scene, chosen so each differs by more than 0.5 m or 15° from earlier ones (§4.1).
  The method "relies on accurate camera poses"; noisy poses are "a factor we have not explored" (§5).
- **Output.** "A collection of 3D bounding boxes"; several boxes per prompt for several instances (§3). No IDs.
  Metric is mAP25 only (§4.1).
- **Cross-view step.** Per-view OWLv2 + SAM2 + depth lifting + DBSCAN (§3.1); photometric and CLIP-feature
  consistency refine depth (§3.2, Eqs. 1-8); finally axis-aligned boxes are "greedily merged based on" IoU
  (§3.3). Table 5 ablates only the depth-refinement losses.
- **Sparse views.** Replica with 4, 8 and 16 views (§4.4): 15.6, 26.7 and 50.5 s on an RTX 4090. The text says 4
  views already beat the point-cloud baseline OpenIns3D.
- **Closest overlap.** Training-free open-vocabulary detection from a few posed, unordered photos, then merging
  by 3D IoU. An open-vocabulary detector of this kind could be run on our views to give a box count. We have not
  tried, and no code link appears in the text read.
- **Not in the paper.** Unique-object counting; identity association; an association ablation; video-tracking
  baselines. §5 itself says objects "classified differently across views" can give overlapping detections and
  that fusion logic is future work.

### 3.3 OpenM3D **[full]**
Hsu et al., *OpenM3D: Open Vocabulary Multi-view Indoor 3D Object Detection without Human Annotations*, arXiv
2508.20063v1 (27 Aug 2025). The earlier version of this file said ICCV 2025; the arXiv text read here states no
venue, so that is **unverified**. **Coverage:** HTML read verbatim to about 80k of 90.5k characters; the tail
(about the last 10 %) was a tool summary. The PDF was not fetched.

- **Input.** Training needs posed RGB-D and a ground-truth mesh for segmentation refinement (§3, §5.2, App. B.1);
  frames are chosen by chronological interval and sharpness (App. B.1). Inference needs "only RGB images and
  their corresponding camera poses" (§4, Conclusion), 20 uniformly sampled views per scene (§5.1).
- **Output.** Axis-aligned 3D boxes with labels from a voxel feature volume (App. B.1). No identities and no
  counting metric; evaluation is AP/AR and pseudo-box precision/recall (Tables 1-3, 5, 6).
- **Cross-view step.** None at detection time is described. Cross-view association exists only when building
  pseudo-labels: 2D segments become graph nodes, edges by overlap > θ, K-means over node embeddings (K = 100),
  majority voting (§3, Eqs. 2-3, Alg. 1, §5.2). Table 1 compares pseudo-box precision with SAM3D, a
  multi-view method, on ScanNet200 (32.07 vs 11.62 at IoU 0.25) as **box quality, not counting**.
- **Sparse views.** Not studied; 20 views at test.
- **Closest overlap.** Geometry-aware grouping of segments across views, used for training targets only.
- **Not in the paper.** Test-time deduplication, counting, duplicate-rate evaluation, sparse or unordered view
  studies. Code and licence: not stated.

### 3.4 VGGT-Det **[full]**
Cao et al., *VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object
Detection*, arXiv 2603.00912v1 (1 Mar 2026). The abstract links a `VGGT-Det-CVPR2026` repository, which suggests
CVPR 2026, but the text read states no venue. **Coverage:** HTML verbatim to about 80k of 92k characters; Sec.
6-9 (supplementary and discussion) were a tool summary. Repository and licence not opened.

- **Input.** Multi-view images only: "no sensor-provided geometric inputs (multi-view poses or depth)"
  (Abstract, §3.1). VGGT's own predicted points guide query placement (§3.2, §3.3). 40 frames by default; mAP
  rises to 46.9 at 80 frames and saturates around there (§4.1, §4.2, Table 2b). Whether frames must be ordered
  is not stated.
- **Output.** 256 object queries each with a class and a 7-parameter box (§3.2, Eq. 5, §4.1). No IDs, no count
  metric. mAP@0.25 on ScanNet (46.9, Table 1) and ARKitScenes (28.0, Table 3).
- **How VGGT is used.** The VGGT encoder is frozen (§4.1); per-view tokens are concatenated (Eq. 1) and queries
  cross-attend to them (Eq. 4), with attention-guided query generation (§3.3) and layer-wise aggregation of
  VGGT layers 4, 11, 17 and 23 (§3.4, §4.1). Cross-view merging is implicit joint attention; no explicit
  matching or deduplication step is described.
- **Closest overlap.** The same foundation model (VGGT) and the same no-poses setting as CountOnce, applied to
  closed-set detection with a trained decoder (18 ScanNet and 17 ARKitScenes classes).
- **Not in the paper.** Open vocabulary, counting, identity association, sparse-view sweeps below 20 frames
  (as far as the verbatim text shows), video-tracking baselines.

### 3.5 ROM: Learning-based Relational Object Matching Across Views **[full]**
Elich, Armeni, Oswald, Pollefeys and Stückler, arXiv 2305.02398v1, ICRA 2023 (stated on the arXiv page). Earlier
versions of this file called it "ROM"; the paper's title is above and ROM is the name of its features.
**Coverage:** HTML verbatim through §IV-A, Tables I-V, Eqs. 1-4; §IV-B (limitations), §V and the references were
a tool summary and are not relied on.

- **Input.** Two RGB images plus per-image 2D detections (§III); no poses, depth or intrinsics at test time.
  Ground-truth 3D positions and distances supervise training only (§III-C, Eqs. 3-4).
- **Output.** Pairwise matches with a dustbin for unmatched objects, from Sinkhorn assignment (§III-B, Eq. 1;
  10 iterations, §IV). Strictly pairwise; no global identities across more than two views, no counting.
- **Architecture.** ResNet34 crop features plus box MLP, attentional GNN with 2 self/cross rounds (§III-A,
  §III-B, Table I), a relative-distance term (Eq. 4), optional SuperGlue keypoint fusion with α = 100 (§III-D).
  No epipolar constraint or triangulation in the sections read.
- **Data.** Hypersim, synthetic indoor, 302/43/40 rooms (§IV). Difficulty bins by distance and angle; on "very
  hard" pairs F1 is 0.332 vs SuperGlue's 0.169 (Table II, ground-truth boxes). Metrics are precision, recall,
  F1.
- **Closest overlap.** Learned object-level cross-view association with a dustbin: the architecture family of
  CountOnce-Match, so that architecture is not ours. Its geometry is implicit, through auxiliary losses.
- **Not in the paper.** Sets of more than two views; unique-object counting; explicit multi-view geometry; real
  data. Code: no official repository link found in the sections read (a repository was noted earlier from its
  GitHub page, not rechecked here).

### 3.6 C-DOG **[full]**
Sun, Lin, Chen, Jiang and Hu, *C-DOG: Multi-View Multi-instance Feature Association Using Connected δ-Overlap
Graphs*, arXiv 2507.14095v2 (1 Aug 2025), CC BY 4.0 paper; no venue found. **Coverage:** HTML read verbatim to
§IV-C, then PDF pages 5-10 read directly (§IV-B to §V-C and Table I). The conclusion, Tables II-V and the
baseline implementation details were not read verbatim.

- **Input.** Calibrated cameras, "intrinsic and extrinsic camera poses are known", and 2D point detections with
  no descriptors (§IV-A, Alg. 1). Not every 3D point needs to be seen in every view (§IV-A), although the
  synthetic benchmark does observe each point in all views (§IV-G). No pose-noise study is described in the
  parts read.
- **Algorithm.** Per view pair and point, take the nearest point by epipolar distance, keep it if below
  τ = α√2σ (Eqs. 6, 14-16, Alg. 1); prune edges whose neighbourhood-overlap score θ is at most δ, with
  δ in [0.5, 0.55] found by grid search on the authors' benchmark (Eq. 17, Alg. 2, §IV-D); remove outliers by
  triangulation back-projection error with an IQR filter (Eqs. 18-19, Alg. 3, §IV-E); discard groups after a
  "sudden increase" in sorted error (§IV-F).
- **Output.** Association groups, each one 3D point, at least 2 members, at most one per view (§IV-A). The count
  is the number of groups and is not scored separately; metrics are group and point F1/IoU, 3D error and
  back-projection error (§V-A).
- **Data.** Synthetic only: random 3D points, 1-130 per scene, projected into 10 calibrated views with Gaussian
  pixel noise of 0-5 px; no missing detections or false positives (§IV-G, §V-A). Baselines include greedy,
  CAO, ST-cut, bundle adjustment, CCA, spectral, permutation synchronization and factorized graph matching
  (§V-C).
- **Sparse views.** Table I (σ = 3): G-F1 is 0.836 at 2 views, 0.861 at 3, 0.743 at 4, 0.775 at 5, then 0.855,
  0.875, 0.890, 0.880, 0.881 at 6-10. Not monotonic; the authors attribute the dip to noisier observations
  (§V-B).
- **Closest overlap.** The simpler geometric matching in H-b: epipolar grouping of per-view points with known
  poses. CountOnce lifts detections with estimated poses, so its input is noisier than C-DOG's.
- **Not in the paper.** Real images or real detections, estimated poses, counting as a task, appearance or
  video baselines. Code is linked from the paper header (`Yunghong/C-DOG`); its code licence was not checked
  (an earlier note found none on the repository).

### 3.7 What this changes for H-b
- No work above tests duplicate suppression as a counting error, so no reviewer can point to a paper that has
  already shown, or refuted, the §2.4 hypothesis. The honest statement is narrower: **several works associate
  across views by geometry (See and Detect, SMOV3D, OpenM3D, C-DOG), and one does so without poses (VGGT-Det),
  but each is evaluated by detection or grouping quality, not unique-object count.**
- The nearest *runnable* comparators are C-DOG (needs poses and points; we have both from VGGT, with noise) and,
  if code exists, SMOV3D (posed images plus a text prompt, boxes counted). See and Detect, OpenM3D and
  VGGT-Det need trained detectors on ScanNet-style indoor scenes and are not usable on orchard photos without
  retraining; they are not added to §4.

## 3b. Other related work, peripheral and not read in full

| Work | Evidence | What it states |
|---|---|---|
| MVDG, arXiv 2610.01098 | [abs] | VGGT-based multi-view disambiguation of visually similar surfaces for reconstruction |
| CrossView, ECCV 2026, arXiv 2608.15539 | [abs] | Multi-camera video question-answering benchmark for VLMs |
| MultiView-Bench, arXiv 2607.08970 | [abs] | Diagnostic benchmark of multi-view spatial integration in VLMs |
| MixCount, arXiv 2605.18063 | [abs] | Synthetic dataset and benchmark for mixed-object open-vocabulary counting |

These benchmark VLMs or reconstruction rather than associating detections across views, so they are peripheral.

## 4. Experiments that test the hypothesis (none run yet)

Rules for all rows: same images, same prompt, same ROI where there is one; count MAE and signed bias with paired
bootstrap CIs on a frozen test split; settings chosen on validation only; failed runs kept as rows.

| ID | Tests | Comparison | Hypothesis not supported if |
|---|---|---|---|
| E1 | H-a, association only | Identical detections into CountOnce association vs SAM 2.1 propagation and ByteTrack, over 3-40 views and increasing viewpoint gap | The tracker's error is equal or lower in the sparse regime |
| E2 | H-a, end to end | CountOnce vs CountVid (with and without the temporal filter) on video versions of own captures at matched frame budgets; a VideoCount static-scene subset if one can be identified | Same |
| E3 | H-a, depth-aided tracking | CountOnce (RGB) vs DG-Net with estimated depth on RGBD-VideoCount; DG-Net with sensor depth shown as a reference | Same |
| E4 | H-b | v1 vs v2 vs sync vs v3 vs C-DOG on identical detections and cameras, on Fuji-SfM, own captures and synthetic data with pose noise | No geometry-aware method beats v1 and C-DOG on real data |
| E5 | Cross-view dedup in general | CountOnce on OneScene-Bench counting, by split, beside Max and a frontier VLM on identical images | No gain over Max on complementary views |

No row is added for See and Detect, OpenM3D or VGGT-Det (§3.7). E4 is unchanged: C-DOG stays the geometric
comparator, and an SMOV3D row is added only if its code is found and runs on our views.

For each experiment, also report where the comparator wins. Likely places, not yet measured: dense continuous
video and moving objects (our pipeline assumes a static scene), and clean sensor depth in crowded RGB-D scenes.
E3 and E2 need a static-scene check per video, because RGBD-VideoCount and VideoCount were not built around
that assumption (VideoCount has moving objects by design).

## 5. What can be said today

- Counting unique instances across video frames exists (CountVid; DG-Net), including from estimated depth
  (DG-Net †). CountOnce is not the first to de-duplicate counts.
- CountOnce differs in design from CountVid and DG-Net by associating detections through camera geometry
  estimated from unordered photos. Whether that **counts better** is the untested part.
- The matcher architecture follows ROM (§3.5); use of VGGT for multi-view tasks has precedent (VGGT-Det §3.1/§4.1, MVDG).
- Cross-view merging of boxes by 3D geometry exists in open-vocabulary multi-view detectors (See and Detect §3.2 Eq. 1; SMOV3D §3.3; OpenM3D §3 for training labels). None of them reports a unique-object count or a duplicate rate (§3).
- No "first to" or "only" wording anywhere until E1-E4 have results. The detection papers have now been read (§3), and they do not remove that condition.

## Sources

- CountVid: https://arxiv.org/abs/2506.15368 , https://github.com/niki-amini-naieni/CountVid
- DG-Net: https://arxiv.org/abs/2608.06236 , https://github.com/streamer-AP/DG-Net , https://huggingface.co/aerospace123/DG-Net , https://huggingface.co/datasets/aerospace123/RGBD-VideoCount
- OneScene-Bench: https://arxiv.org/abs/2610.38926 , https://github.com/swathisuhas/OneScene-Bench
- ROM: https://arxiv.org/abs/2305.02398 , https://arxiv.org/html/2305.02398v1 , https://github.com/EmbodiedVision/rom
- C-DOG: https://arxiv.org/abs/2507.14095 , https://github.com/Yunghong/C-DOG
- VGGT-Det: https://arxiv.org/abs/2603.00912 , https://arxiv.org/html/2603.00912 , https://github.com/yangcaoai/VGGT-Det-CVPR2026
- See and Detect: https://arxiv.org/abs/2611.17897 , OpenM3D: https://arxiv.org/abs/2508.20063 , SMOV3D: https://arxiv.org/abs/2509.15924 (HTML versions read)
- Abstract-only: https://arxiv.org/abs/2610.01098 , https://arxiv.org/abs/2608.15539 , https://arxiv.org/abs/2607.08970 , https://arxiv.org/abs/2605.18063
- See also `MODEL_SURVEY.md`, `DATASETS.md`, `PLAN.md` §2.3-2.4.
