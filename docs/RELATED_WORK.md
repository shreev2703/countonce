# Related work, organised around the §2.4 hypothesis

Last updated 2026-11-25 (task M10). Replaces the 2026-11-24 abstract-level version.

## 0. How to read this file

| Tag | Meaning |
|---|---|
| **[full]** | Paper read end to end, including appendices (CountVid, DG-Net, OneScene-Bench only) |
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
  GPU has less, so batch sizes must be reduced; whether it fits is `TODO: measure`.
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

## 3. Other related work (not read in full)

| Work | Evidence | What it states |
|---|---|---|
| ROM, Elich et al., ICRA 2023, arXiv 2305.02398 | [grep] | Matches object detections between pairs of images using keypoints and object-level features in an attentional graph neural network, with Sinkhorn assignment and a dustbin; trained on rendered synthetic images (Hypersim). The matcher family CountOnce-Match starts from, so the architecture idea is not ours |
| C-DOG, arXiv 2507.14095 | [grep] | Training-free multi-view multi-instance association from epipolar geometry, given calibrated cameras (K, R, T) and 2D points. No licence on its repository **[repo]**, so re-implement from the paper |
| VGGT-Det, CVPR 2026, arXiv 2603.00912 | [abs] | Multi-view indoor 3D detection without sensor-provided poses or depth, built on the VGGT encoder; evaluated on ScanNet and ARKitScenes |
| OpenM3D, ICCV 2025, arXiv 2508.20063 | [abs] | Open-vocabulary multi-view indoor 3D detector trained without human annotations (posed RGB-D in training, multi-view images at inference) |
| See and Detect, arXiv 2611.17897 | [abs] | Open-vocabulary multi-view indoor 3D detection that fuses spatially consistent per-view predictions ("multi-view consensus box consolidation") |
| Sparse Multiview Open-Vocabulary 3D Detection, arXiv 2509.15924 | [abs] | Training-free open-vocabulary 3D detection from a few **posed** RGB images, lifting 2D detections |
| MVDG, arXiv 2610.01098 | [abs] | VGGT-based multi-view disambiguation of visually similar surfaces for reconstruction |
| CrossView, ECCV 2026, arXiv 2608.15539 | [abs] | Multi-camera video question-answering benchmark for VLMs |
| MultiView-Bench, arXiv 2607.08970 | [abs] | Diagnostic benchmark of multi-view spatial integration in VLMs |
| MixCount, arXiv 2605.18063 | [abs] | Synthetic dataset and benchmark for mixed-object open-vocabulary counting |

The open-vocabulary multi-view 3D detection papers are the likeliest reviewer objection: their boxes can be
counted, and at least See and Detect consolidates predictions across views. Read them in full before making any
novelty claim.

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

For each experiment, also report where the comparator wins. Likely places, not yet measured: dense continuous
video and moving objects (our pipeline assumes a static scene), and clean sensor depth in crowded RGB-D scenes.
E3 and E2 need a static-scene check per video, because RGBD-VideoCount and VideoCount were not built around
that assumption (VideoCount has moving objects by design).

## 5. What can be said today

- Counting unique instances across video frames exists (CountVid; DG-Net), including from estimated depth
  (DG-Net †). CountOnce is not the first to de-duplicate counts.
- CountOnce differs in design from CountVid and DG-Net by associating detections through camera geometry
  estimated from unordered photos. Whether that **counts better** is the untested part.
- The matcher architecture follows ROM; use of VGGT for multi-view tasks has precedent (VGGT-Det, MVDG).
- No "first to" or "only" wording anywhere until E1-E4 have results and §3's detection papers have been read.

## Sources

- CountVid: https://arxiv.org/abs/2506.15368 , https://github.com/niki-amini-naieni/CountVid
- DG-Net: https://arxiv.org/abs/2608.06236 , https://github.com/streamer-AP/DG-Net , https://huggingface.co/aerospace123/DG-Net , https://huggingface.co/datasets/aerospace123/RGBD-VideoCount
- OneScene-Bench: https://arxiv.org/abs/2610.38926 , https://github.com/swathisuhas/OneScene-Bench
- ROM: https://arxiv.org/abs/2305.02398 , https://github.com/EmbodiedVision/rom
- C-DOG: https://arxiv.org/abs/2507.14095 , https://github.com/Yunghong/C-DOG
- VGGT-Det: https://arxiv.org/abs/2603.00912 , https://github.com/yangcaoai/VGGT-Det-CVPR2026
- Abstract-only: https://arxiv.org/abs/2508.20063 , https://arxiv.org/abs/2611.17897 , https://arxiv.org/abs/2509.15924 , https://arxiv.org/abs/2610.01098 , https://arxiv.org/abs/2608.15539 , https://arxiv.org/abs/2607.08970 , https://arxiv.org/abs/2605.18063
- See also `MODEL_SURVEY.md`, `DATASETS.md`, `PLAN.md` §2.3-2.4.
