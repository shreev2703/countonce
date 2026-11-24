# Related work vs. the §2.4 novelty claim

Scope: CountVid, DG-Net, VGGT-Det, ROM, C-DOG, OneScene-Bench, plus adjacent work found during the search.

**Evidence level.** Written overnight from search results, abstracts, repo/model cards and the earlier
`MODEL_SURVEY.md` / `DATASETS.md`. **No paper was read in full.** Every "does not do X" statement below is
a claim about the abstract-level description, tagged `TODO: verify in full text` where it matters.
No metrics from any paper are reproduced here.

## 1. The claim under test (PLAN §2.4)

> Open-vocabulary object counting from a few unordered RGB photos, with de-duplication in a 3D frame
> recovered by feed-forward reconstruction (no depth sensor, no SfM), and a multi-view learned association
> model trained on synthetic data that transfers to real orchards.

Split into five testable parts:

| ID | Component |
|---|---|
| C1 | Open-vocabulary (text/exemplar) counting |
| C2 | Few, **unordered** RGB photos (not continuous video) |
| C3 | De-duplication in a **3D frame from feed-forward reconstruction** (no depth sensor, no SfM) |
| C4 | **Multi-view** learned association (not pairwise) |
| C5 | Trained on **synthetic** data, transfers to real orchards |

## 2. Coverage matrix

Y = does it, P = partly, N = no (per abstract-level description), n/a = not that kind of work.

| Work | C1 open-vocab | C2 unordered sparse RGB | C3 feed-forward 3D dedup | C4 multi-view learned assoc. | C5 synthetic→real | Task |
|---|---|---|---|---|---|---|
| CountVid | Y | N (video) | N (2D tracking) | N | N | counting |
| DG-Net | Y (CountGD-based) | N (video, RGB-D) | N (sensor depth) | P (depth-aided association) | N | counting |
| VGGT-Det | N (fixed categories) | Y | P (VGGT internals, no explicit dedup) | N | N | 3D detection |
| ROM | N | Y (pairs) | N (2D relations) | N (pairs only) | Y (Hypersim) | matching |
| C-DOG | n/a | Y | N (epipolar, needs known cameras) | N (training-free) | n/a | association |
| OneScene-Bench | n/a | Y | n/a | n/a | n/a | VLM evaluation |

No single work covers C1 + C2 + C3 together. The combination, not any single part, is the novelty.

## 3. Per-work comparison

### CountVid (Oxford VGG, AAAI-26, arXiv 2506.15368)
- **What:** CountGD-Box detections + SAM 2.1 promptable video tracking to count unique instances in a video from text or exemplars. Releases VideoCount (370 videos).
- **Overlap:** C1 and the de-duplication goal.
- **Gap:** Needs temporal continuity; de-duplication is 2D track identity. It has no camera geometry, so it should degrade when frames are sparse or viewpoints jump and tracks cannot be linked.
- **Where it likely wins:** dense continuous video, small motion between frames, moving objects (3D methods assume a static scene).
- **Experiment:** run on video versions of own captures at matched frame budgets; sweep frames-per-scene (3→40) and viewpoint gap; run on a VideoCount subset (Strong tier). Report where CountVid wins.

### DG-Net / Depth-VIC (ACM MM 2026, arXiv 2608.06236)
- **What:** Depth-guided detector (CountGD-initialised) plus post-processing for occlusion, cross-frame association and de-duplication in crowded video. Needs RGB-D. Releases RGBD-VideoCount (195 clips, CC BY 4.0 per HF card).
- **Overlap:** C1, geometry-aided de-duplication.
- **Gap:** Depends on sensor depth (its model card warns of degradation when depth is inaccurate or missing) and on video. CountOnce uses RGB only, with estimated geometry in a global frame.
- **Where it likely wins:** RGB-D crowded video where depth is clean.
- **Experiment:** RGBD-VideoCount, RGB-only CountOnce vs DG-Net, stating the sensor difference. Also try CountOnce with sensor depth substituted, to separate "better geometry" from "better method" `TODO`.
- **Naming:** the paper says DG-Det; the repo and model card say DG-Net. Same work as far as the sources show.

### VGGT-Det (CVPR 2026, arXiv 2603.00912)
- **What:** Sensor-geometry-free multi-view indoor 3D object detection. VGGT encoder as backbone; attention-guided query generation uses VGGT attention maps as semantic priors; a query-driven aggregator pools multi-level features. Evaluated on ScanNet and ARKitScenes.
- **Overlap:** C3-adjacent: uses VGGT for pose-free multi-view reasoning, which supports our backbone choice.
- **Gap:** Fixed indoor categories, trained detector, outputs boxes not counts, no open vocabulary and no de-duplication objective. Needs training on indoor 3D annotations. Code: `yangcaoai/VGGT-Det-CVPR2026`.
- **Position:** supporting evidence that VGGT is a viable pose-free multi-view backbone. Not a competitor on counting.
- **Experiment:** none required. Optionally a qualitative note that a closed-vocabulary detector cannot take a text prompt.

### ROM (ICRA 2023, arXiv 2305.02398, "Learning-based Relational Object Matching Across Views")
- **What:** Matches object detections between **image pairs** using local keypoints plus object-level features and an attentional GNN over appearance and spatial relations. Scores via dot products, Sinkhorn with a dustbin. Trained on rendered Hypersim.
- **Overlap:** C5 (synthetic training) and the matcher architecture family. CountOnce-Match builds on this idea (PLAN §10).
- **Gap:** Pairwise, no 3D geometry, indoor and not counting. CountOnce adds many-view association, 3D features from reconstruction, cannot-link constraints and cycle-consistent synchronization.
- **Honesty point:** the matcher's architectural idea is not novel; novelty is the multi-view geometry-conditioned version, its use for counting, and the synthetic→real orchard transfer.
- **Experiment:** ablate pairwise-only vs multi-view; ROM-style pairwise baseline `TODO: verify whether code is usable` (EmbodiedVision/rom exists; license `TODO: verify`).

### C-DOG (arXiv 2507.14095)
- **What:** Training-free, feature-free multi-view multi-object association via epipolar-distance graphs, delta-neighbour overlap clustering and IQR filtering. Input is per-camera K, R, T and 2D points.
- **Overlap:** The association step (C4) as a geometric method.
- **Gap:** Requires known cameras (we get them from VGGT, with noise), uses no appearance, no learning. It is a baseline for our geometric v1/v2 merge, not for counting end to end.
- **Licence:** repo has none, so re-implement from the paper (see MODEL_SURVEY).
- **Experiment:** same detections + same VGGT cameras into C-DOG vs 3D clustering vs learned matcher, on Fuji-SfM and synthetic data; stress with pose noise.

### OneScene-Bench (NeurIPS 2026, arXiv 2610.38926)
- **What:** Benchmark of whether VLMs integrate evidence across views, with counting under controlled overlap and spatial reasoning. 12 models. Per the abstract, counting accuracy is 4.5–50.0% on complementary views vs 45.5–81.5% on complete-overlap views.
- **Role:** Motivation and evaluation set, not a method. It supports the claim that unique counting across views is unsolved for VLMs, and that the hard case is complementary views, which is where explicit correspondence matters.
- **Caution:** data/licence/format are `TODO: verify` (DATASETS.md). Questions are indoor DL3DV scenes, so it tests the dedup logic but not orchard transfer.
- **Experiment:** CountOnce on its counting questions, compared with the reported VLM numbers; also run our own frontier-VLM baseline on identical images.

## 4. Adjacent work that could weaken the claim (not yet read)

Surfaced by a single extended search; none read beyond title/snippet. **Read these in full before the professor meeting.**

| Work | Why it matters |
|---|---|
| OpenM3D (arXiv 2508.20063) | Open-vocab multi-view indoor 3D detection from RGB; reportedly uses VGGT in a baseline pipeline. Closest on C1+C3 for detection. |
| See and Detect (arXiv 2611.17897) | Image-driven open-vocab multi-view 3D detection. |
| Sparse Multiview Open-Vocabulary 3D Detection (arXiv 2509.15924) | Training-free, sparse multi-view RGB, open-vocab. Close to C1+C2. |
| MVDG (arXiv 2610.01098) | Multi-view 3D disambiguation on unconstrained real-world images; possibly relevant to association. |
| CrossView (2608.15539), MultiView-Bench (2607.08970) | Further evidence of the VLM cross-view gap; candidates for extra evaluation. |
| MixCount (arXiv 2605.18063) | Open-vocab counting dataset; whether it has any multi-view part is unknown. |

Open-vocab multi-view 3D **detection** can be turned into counting by counting boxes, so a reviewer may ask for it as a baseline. The distinguishing factors would be explicit de-duplication and counting-specific evaluation, plus the learned association trained on synthetic data. `TODO: verify.`

## 5. Verdict on §2.4

1. **Survives, with one qualification.** No surveyed work provides open-vocabulary counting from sparse unordered RGB photos with feed-forward-3D de-duplication. The search found no paper combining open-vocab counting, cross-view 3D dedup and VGGT-style reconstruction (search negatives are weak evidence).
2. **Narrowest defensible wording:** the claim should lean on the *combination* and on C5 (synthetic→real transfer in agriculture), and should not claim novelty for the matcher architecture (ROM) or for use of VGGT in multi-view tasks (VGGT-Det, OpenM3D).
3. **Weakest component:** C4+C5 are only novel if the learned matcher beats geometric merge on real data (PLAN goal 3). If it does not, the contribution is C1+C2+C3 as a pipeline plus the benchmark, which is still defensible but smaller.
4. **Most exposed to a reviewer:** open-vocab multi-view 3D detection papers (§4), which may be cast as "count the boxes".
5. **Must be reported honestly:** video tracking (CountVid) on continuous video, sensor depth (DG-Net) on RGB-D, and static-scene assumption failures (moving objects).

## 6. Follow-ups (feed Phase 0)

- [ ] Read in full: CountVid, DG-Net, OneScene-Bench, and the §4 papers; replace `TODO: verify` tags.
- [ ] Check ROM code licence and whether C-DOG's author will grant one.
- [ ] Add OpenM3D / training-free sparse multi-view detector as a "count the boxes" baseline if code exists.
- [ ] Bring §2 matrix and §5 verdict to the professor meeting.

## Sources
- CountVid: https://arxiv.org/abs/2506.15368 , https://github.com/niki-amini-naieni/CountVid
- DG-Net: https://arxiv.org/pdf/2608.06236 , https://github.com/streamer-AP/DG-Net , https://huggingface.co/aerospace123/DG-Net , https://huggingface.co/datasets/aerospace123/RGBD-VideoCount
- VGGT-Det: https://arxiv.org/abs/2603.00912 , https://cvpr.thecvf.com/virtual/2026/poster/39884
- ROM: https://arxiv.org/abs/2305.02398 , https://github.com/EmbodiedVision/rom
- C-DOG: https://arxiv.org/abs/2507.14095 , https://github.com/Yunghong/C-DOG
- OneScene-Bench: https://arxiv.org/pdf/2610.38926 , https://neurips.cc/virtual/2026/165035
- Adjacent: https://arxiv.org/html/2508.20063 , https://arxiv.org/pdf/2611.17897 , https://arxiv.org/pdf/2509.15924 , https://arxiv.org/html/2610.01098 , https://arxiv.org/pdf/2608.15539 , https://arxiv.org/pdf/2607.08970 , https://arxiv.org/pdf/2605.18063
- See also `MODEL_SURVEY.md`, `DATASETS.md`, `PLAN.md` §2.3–2.4.
