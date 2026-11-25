# Model survey (Phase 0)

Written 2026-11-24 from the upstream READMEs, GitHub/Hugging Face metadata and arXiv pages. Nothing here was run on our hardware
unless stated. Where upstream does not publish a figure the cell says `TODO: measure` (PLAN §4 rule 3: no invented numbers).
Licenses are what GitHub/HF report plus what the READMEs say; re-read the license text before any commercial step (PLAN §24).

Related: [[PLAN]] §24, [[DEVIATIONS]].

## Summary

| Model | Role in CountOnce | Repo | Code license | Weights | Weights license | Output | VRAM |
|---|---|---|---|---|---|---|---|
| CountGD | 2D detector (candidate) | [niki-amini-naieni/CountGD](https://github.com/niki-amini-naieni/CountGD) | MIT | Google Drive, 1.2 GB (+ GroundingDINO Swin-B, SAM ViT-H) | MIT (HF card); SAM/GDINO keep own | per-image count + boxes/points | TODO: measure |
| CountGD++ | 2D detector (preferred) | [niki-amini-naieni/CountGDPlusPlus](https://github.com/niki-amini-naieni/CountGDPlusPlus) | MIT | [HF `nikigoli/CountGDPlusPlus`](https://huggingface.co/nikigoli/CountGDPlusPlus) (safetensors), also Drive 1.25 GB | MIT (HF card) | per-image count + boxes | TODO: measure |
| GroundingDINO | detector backbone / text-only fallback | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 | GitHub releases (`groundingdino_swinb_cogcoor.pth` etc.) | Apache-2.0 (code license; verify per checkpoint) | boxes + phrase logits | TODO: measure |
| CountVid | video-counting baseline | [niki-amini-naieni/CountVid](https://github.com/niki-amini-naieni/CountVid) | MIT | CountGD-Box (Drive) + SAM 2.1 Hiera-L | MIT / SAM 2.1 own license | unique-instance count, tracked masks, videos | TODO: measure |
| DG-Net | video-counting baseline (RGB-D) | [streamer-AP/DG-Net](https://github.com/streamer-AP/DG-Net) | MIT (third-party parts keep own) | [HF `aerospace123/DG-Net`](https://huggingface.co/aerospace123/DG-Net) | MIT (HF tag) | JSON: global metrics, per-video counts, frame-wise boxes | TODO: measure |
| C-DOG | association baseline | [Yunghong/C-DOG](https://github.com/Yunghong/C-DOG) | **none declared** | none (training-free) | n/a | groups of 2D points, one per camera, `[-1,-1]` = missing | CPU only |
| VGGT | reconstructor (current) | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | VGGT License (custom; commercial OK excl. military) | HF `facebook/VGGT-1B` (non-commercial), `facebook/VGGT-1B-Commercial` (gated form) | see left | extrinsics/intrinsics (OpenCV, cam-from-world), depth + conf, point maps, tracks | 3.1 GB @1 frame to 6.9 GB @70 (own 8 GB run) |
| VGGT-Ω | reconstructor (candidate) | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) | FAIR Noncommercial Research License v1 | HF `facebook/VGGT-Omega`, gated | non-commercial research | cameras, depth + conf (point cloud by unprojection) | 6.0 GB @1 to 43 GB @500 (A100, 624x416) |
| VGG-T³ | long-sequence reconstructor (watch) | none found; [paper](https://arxiv.org/abs/2602.23361), [project](https://research.nvidia.com/labs/dvl/projects/vgg-ttt) | unknown | unknown | unknown | VGGT-style geometry (paper) | not published in sources read |

## Per-model notes

### CountGD (NeurIPS 2024, arXiv 2407.04619)
- Open-world counting from text, visual exemplars, or both. Built on Open-GroundingDINO; python 3.9, needs a compiled
  MultiScaleDeformableAttention CUDA extension.
- Weights: paper checkpoint 1.2 GB (Google Drive); inference also needs GroundingDINO Swin-B and SAM ViT-H checkpoints
  (SAM only for the test-time normalisation; README says skipping it is faster but slightly less accurate).
- Output: count plus boxes per image via `main_inference.py` (COCO-style eval). Reference runtime: FSC-147 val ~26 min on one RTX 3090 (~6 min without SAM).
- Risk for us: Google Drive weights links; pinned old env vs. sm_120 (RTX 5070) needs a CUDA 12.8 rebuild of the extension.

### CountGD++ (CVPR 2026, arXiv 2512.23351)
- Adds negative prompts (text/visual), pseudo-exemplars from text-only input, external exemplar images.
- Python 3.10, GCC 11, CUDA toolkit + detectron2 needed. Weights on HF as safetensors (MIT); repo also lists a 1.25 GB Drive file.
- **Recommendation:** primary detector (strictly more capable than CountGD, same license, HF-hosted weights).
  Negative prompts are directly useful for foliage look-alikes. Last upstream push 2026-10-27.

### GroundingDINO
- Apache-2.0 code, weights on GitHub releases. Text-prompted boxes only, no counting head. Used by CountGD/++ as the
  backbone; keep as a plain-text fallback detector and for ablation. Last push 2024-08 (stale; expect build friction on new CUDA).

### CountVid (AAAI-26, arXiv 2506.15368)
- CountGD-Box detector + SAM 2.1 tracker to count unique instances in a video from text or exemplar. Also releases the VideoCount dataset.
- Weights: CountGD-Box (Drive) and `sam2.1_hiera_large.pt`. CLI `count_in_videos.py` with `--obj_batch_size` / `--img_batch_size` / downsample knobs, which suggests memory is tunable.
- Role: the main single-camera-video baseline CountOnce must beat or complement (see RELATED_WORK task).

### DG-Net (Depth-Guided Video Object Counting, repo `streamer-AP/DG-Net`)
- Extends CountVid with depth; trained and evaluated on RGBD-VideoCount (HF dataset). Tested on RTX 5090, CUDA 12.8, Python 3.10.
- Takes a depth map per frame: sensor depth, or estimated depth (paper Table 2 † rows). So an RGB-only comparison is possible by estimating depth; needs `countgd_box.pth` to train, `DG-Net.pth` for inference.
- Output: JSON with global metrics, per-entry predicted counts + GT, frame-wise boxes; `scripts/demo.py` renders GIFs.
- Accepted at ACM Multimedia 2026 (arXiv comment and repo README, checked 2026-11-25). Do not confuse with unrelated "DG-Net" papers.

### C-DOG (arXiv 2507.14095)
- Training-free, feature-free multi-view association via epipolar-distance graphs, delta-neighbour-overlap clustering, IQR filtering.
- Input: per-camera K, R, T and 2D points. Output: groups of 2D points, one per camera. Repo is a research script (`main.py`), demo on a 130-instance synthetic case.
- **No license in the repo**: legally all rights reserved. Treat as reference only; re-implement from the paper rather than vendoring code, or ask the author. Logged as a gate-style caveat, not a blocker.
- Relevant to our cannot-link clustering in `countonce/merge/cluster.py` as a comparison baseline.

### VGGT (CVPR 2025)
- Already ported (`countonce/recon/vggt.py`). Predicts cameras, depth, point maps, tracks from 1..hundreds of views.
- License is a custom "VGGT License" (2025-07-29): commercial use allowed excluding military, but **only** the `VGGT-1B-Commercial` checkpoint (gated, form-based) is commercial; the original `VGGT-1B` stays non-commercial.
- README (2026-05-15) notes a fix that lets the same GPU memory run roughly 2-3x more frames; our earlier VGGT-project numbers
  predate/may not include that. VRAM figures from our own run (RTX 5070 Laptop 8 GB, WSL2, `~/vggtProject/README.md`): 3.1 GB at 1 frame, 6.9 GB at 70 frames; 3.4 to 5.6 GB for 3 to 40 views. **TODO: re-measure on current upstream.**

### VGGT-Ω (CVPR 2026 oral, arXiv 2605.15195)
- DINOv3 (patch 16) backbone. Checkpoints (gated on HF, automated approval): `VGGT-Omega-1B-512` (recommended in-the-wild), `-1B-416-Reproduction`, `-1B-256-Text-Alignment`.
- Output: `predictions["depth"]`, `depth_conf`, cameras via `encoding_to_camera`; point cloud by unprojecting depth.
- License: FAIR Noncommercial Research License v1, so research only (PLAN §24 already assumes this).
- VRAM (A100, `VGGT-Omega-1B-512`, 624x416, includes weights): 1 frame 6.02 GB, 10: 6.67, 25: 7.80, 50: 9.66, 100: 13.37, 200: 20.82, 500: 43.15 GB.
  On 8 GB that is about 25 frames; on 12 GB about 75 (interpolated from upstream table, `TODO: verify on RTX 5070`).
  `mode="max_size"` (512x336) reduces memory.

### VGG-T³ (CVPR 2026, arXiv 2602.23361, NVIDIA / U. Toronto)
- Replaces VGGT global attention with test-time-training blocks, making cost linear in views. Reports 1k images in 54 s (11.6x faster than baseline; project page: 58 s vs 11 min for VGGT).
- **Could not find a code or weights release** in the arXiv page, project page or web search. License and VRAM unknown.
  Status: watch-only; revisit if code appears. Related public alternative with code: TTT3R.

## Decisions

1. **Detector:** CountGD++ primary, CountGD as ablation, GroundingDINO text-only fallback. All MIT/Apache-compatible with a commercial path (weights' third-party components, e.g. SAM, to be checked at deploy).
2. **Reconstructor:** keep VGGT (commercial checkpoint available) as default; VGGT-Ω as research-only upgrade behind the `Reconstructor` interface. VGG-T³ deferred (no code).
3. **Baselines:** CountVid (RGB video), DG-Net (RGB-D only), C-DOG (re-implement from paper because of missing license).
4. **Commercial gate:** any deployed path must avoid VGGT-Ω, original VGGT-1B, and C-DOG code.

## Open items
- `TODO: measure` detector VRAM/latency for CountGD++ / CountVid / DG-Net on the RTX 5070 (needs sm_120 extension build).
- Confirm checkpoint-level licenses for GroundingDINO weights and SAM/SAM 2.1.
- Check weekly for a VGG-T³ release.
