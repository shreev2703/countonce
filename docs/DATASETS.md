# DATASETS

Availability, license, format and size for every dataset in `docs/PLAN.md` §9.1.
Researched 2026-11-24 from web search snippets only (no files downloaded, no landing pages read in full).
Anything not confirmed from a primary source is marked `TODO: verify`. Licenses that conflict between
sources are listed as conflicts; resolve against the primary record before any release (PLAN §Track I).

## Summary

| Dataset | Role | Available | License (best evidence) | Size | Tier |
|---|---|---|---|---|---|
| Fuji-SfM | Exact dup/miss measurement; matcher fine-tuning | Zenodo, DOI 10.5281/zenodo.3712808 | **Conflict**: CC BY 4.0 (MaRDI, recercat) vs CC BY-NC-SA 4.0 (Roboflow) vs "research/education only" (Lleida) | ~6.4-6.9 GB | Core |
| RGBD-VideoCount | RGB-D video counting eval | Hugging Face `aerospace123/RGBD-VideoCount` | CC BY 4.0 (HF card) | 2,032 frames, 77,638 boxes; GB `TODO: verify` | Strong |
| VideoCount | Video counting eval (CountVid benchmark) | github.com/niki-amini-naieni/CountVid | `TODO: verify` (repo is MIT; source-video licenses TAO/MOT20 apply) | `TODO: verify` | Strong |
| OneScene-Bench | Multi-view evidence integration (VLM) eval | arXiv 2610.38926; release `TODO: verify` | `TODO: verify` | `TODO: verify` | Strong |
| WGISD | Grape-cluster detection; secondary orchard data | Zenodo 3361736, GitHub, HF | **Conflict**: CC BY-NC-SA 4.0 (Dataset Ninja) vs "no restrictions, contact Embrapa for commercial" (README) | ~1.3 GB | Optional |
| MinneApple | Apple detection/counting, detector sanity checks | UMN Conservancy handle 11299/206575 | UMN Terms of Use; CC BY-NC-SA 3.0 US per Dataset Ninja (unconfirmed) | ~1-4 GB across archives | Optional |
| AppleGrowthVision | Stereo apple orchard, detection, 3D | Fraunhofer HHI project page; HF mirror `Voxel51/AppleGrowthVision` | Original `TODO: verify`; HF FiftyOne copy says CC BY 4.0 | 9,317 stereo + 1,125 annotated images; GB `TODO: verify` | Optional |
| FSC-147 | Class-agnostic counting reference / detector sanity | Official release (Ranjan et al., CVPR 2021); HF mirrors | Official `TODO: verify`; a HF mirror tags MIT (uploader label, not authoritative) | 6,135 images, ~5.2 GB repo on one mirror | Optional |

## Per-dataset notes

### Fuji-SfM
- **What:** SfM image sequence of 11 Fuji apple trees (582 images), 3D point cloud of the scanned scene with
  1,455 apple positions as ground truth in global coordinates, and 2D instance masks for 288 images (per PLAN §9.1).
- **Where:** Zenodo DOI 10.5281/zenodo.3712808 (a sibling record 3712807 also exists, `TODO: verify` which part it holds);
  Lleida GRAP page `grap.udl.cat/en/publications/fuji-sfm-dataset/`.
- **Format:** images + masks + point cloud + apple-position GT; exact file layout and camera/pose convention `TODO: verify after download`.
  Needed for the GT-projection matching in PLAN §9.1 (cameras either shipped or recovered by COLMAP/VGGT + Sim(3) alignment).
- **Size:** ~6.9 GB (Zenodo) / 6.43 GB (Lleida).
- **License:** conflicting (see table). Treat as **non-commercial research use** and do not redistribute the data until `LICENSE.txt` in the
  Zenodo record is read. Cite Data in Brief (10.1016/j.dib.2020.105591) and Comput. Electron. Agric. (10.1016/j.compag.2019.105165).
- **Decision:** Core benchmark dataset; download script only, no data committed. Public CountOnce-Bench release must not include it unless the license allows.

### RGBD-VideoCount
- **What:** RGB-D video counting from "Depth-Guided Video Object Counting in Crowded Scenes" (arXiv 2608.06236); Intel RealSense D455;
  2,032 annotated frames, 77,638 instance boxes, multiple categories per video, depth, exemplar boxes, video-level ROIs.
- **Where:** `huggingface.co/datasets/aerospace123/RGBD-VideoCount`; code at `github.com/streamer-AP/DG-Net`.
- **Format:** per the card: RGB + depth + box annotations. HF viewer currently fails, so expect raw-file download; layout `TODO: verify`.
- **License:** CC BY 4.0 (HF card).
- **Size:** `TODO: verify`.

### VideoCount
- **What:** Benchmark from the CountVid paper, built from TAO and MOT20 tracking videos plus penguin videos and X-ray alloy crystallization videos.
- **Where:** `github.com/niki-amini-naieni/CountVid`.
- **License:** `TODO: verify`; derived videos inherit TAO/MOT20 terms. MOT20 is research-use only, `TODO: verify`.
- **Format / size:** `TODO: verify`.
- **Fit caveat:** crowds, penguins, crystals; not orchard-like. Used only as the Strong-tier external video benchmark.

### OneScene-Bench
- **What:** arXiv 2610.38926, "Do Vision-Language Models Integrate Evidence Across Views?". **Not a counting-only dataset**; its counting part draws on
  DL3DV-10K scenes (object category chosen by Qwen3-VL-32B-Instruct). It is the right probe for the frontier-VLM baseline.
- **Availability, license, format, size:** `TODO: verify` (not found in search results; check the paper's project/code link).
- **Note:** DL3DV-10K has its own terms (`TODO: verify`) that bind any derived images.

### WGISD (Embrapa Wine Grape Instance Segmentation Dataset)
- **What:** 300 images, 4,432 grape-cluster boxes; 137 images with 2,020 binary masks; 5 varieties (Chardonnay, Cabernet Franc,
  Cabernet Sauvignon, Sauvignon Blanc, Syrah); Guaspari Winery, Brazil; 2019.
- **Where:** Zenodo 3361736 (v1.0.0), GitHub (Embrapa / mirrors), HF `thsant/wgisd`.
- **Format:** JPG images, YOLO-style text box labels, PNG masks, point-cloud-free (single views; ~1.3 GB).
- **License:** conflicting. Prefer the official README (free use, commercial users contact Embrapa) but treat as non-commercial in the meantime.
- **Fit caveat:** no multi-view identity GT in the base release; useful for detector prompts on clusters, not dedup measurement.

### MinneApple
- **What:** Apple detection, segmentation and counting benchmark (Häni, Roy, Isler, Univ. of Minnesota); ~1,000 images, 41k+ instances per project page.
- **Where:** `conservancy.umn.edu/handle/11299/206575`; paper arXiv 1909.06441.
- **Format:** polygon masks; separate archives: counting (2,875 JPG + 67,990 PNG, ~977 MB), detection/segmentation (1,671 PNG, 1.7 GB),
  test labels (1.17 GB).
- **License:** UMN Digital Conservancy Terms of Use apply, possibly depositor restrictions. CC BY-NC-SA 3.0 US per a third-party site (unconfirmed).
- **Conflict:** Dataset Ninja reports 69,982 images / 23,065 objects, which differs from the primary 41k-in-1000 figure; trust the UMN record.
- **Fit caveat:** single images, no cross-view identity; per-image counting only.

### AppleGrowthVision
- **What:** Stereo orchard dataset (CVPRW 2025, von Hirschhausen et al., Fraunhofer HHI; arXiv 2505.14029): 9,317 stereo images across six BBCH
  growth stages (Brandenburg farm) plus 1,125 densely annotated images with 31,084 apple labels.
- **Where:** `fraunhoferhhi.github.io/AppleGrowthVision` (authors' page); HF mirrors `Voxel51/AppleGrowthVision`, `harpreetsahota/AppleGrowthVision`.
- **Format:** stereo image pairs + detection boxes; the FiftyOne copy is FiftyOne-format. Exact original layout `TODO: verify`.
- **License:** original `TODO: verify`; HF conversion states CC BY 4.0 (does not prove the original's terms).
- **Size:** `TODO: verify`.
- **Fit caveat:** stereo gives a second geometric signal for VGGT depth checks; no multi-view identity GT stated.

### FSC-147
- **What:** 6,135 images, 147 object categories, class-agnostic counting (Ranjan et al., CVPR 2021, "Learning to Count Everything").
- **Where:** official project/GitHub release (link `TODO: verify`); HF mirrors e.g. `isentropic/FSC147`, `Hzzone/PseCo`.
- **Format:** images (384-px-height variant widely used), `annotation_FSC147_384.json` (~45.6 MB) with points/exemplar boxes, split file.
- **Size:** ~5.19 GB for one mirror repo (not broken out by folder).
- **License:** official terms `TODO: verify`; images are scraped web images so redistribution is ambiguous. Use for evaluation of the detector only; never ship.
- **Fit caveat:** single-view counting; supports detector/CountGD sanity checks, not dedup.

## Decisions
1. **Only Fuji-SfM and own captures provide cross-view identity GT**; the rest are single-view or video counting references. PLAN §3.1 Core therefore stays Fuji-SfM + own captures.
2. **Download scripts only.** No dataset bytes are committed or redistributed; each license is resolved before CountOnce-Bench release.
3. **License policy:** any conflicting or unverified license is treated as non-commercial, research-only until `TODO: verify` items are closed.
4. **Open items for the first download session:** Fuji-SfM `LICENSE.txt` and file layout; RGBD-VideoCount size/layout; VideoCount and OneScene-Bench terms; FSC-147 official terms; AppleGrowthVision original license.

## Sources
- Fuji-SfM: https://zenodo.org/record/3712808 , https://grap.udl.cat/en/publications/fuji-sfm-dataset/ , https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7184157/
- RGBD-VideoCount: https://huggingface.co/datasets/aerospace123/RGBD-VideoCount , https://arxiv.org/pdf/2608.06236 , https://github.com/streamer-AP/DG-Net
- VideoCount: https://github.com/niki-amini-naieni/CountVid
- OneScene-Bench: https://arxiv.org/pdf/2610.38926
- WGISD: https://zenodo.org/record/3361736 , https://huggingface.co/datasets/thsant/wgisd , https://datasetninja.com/wgisd
- MinneApple: https://conservancy.umn.edu/handle/11299/206575 , https://arxiv.org/pdf/1909.06441
- AppleGrowthVision: https://arxiv.org/abs/2505.14029 , https://huggingface.co/datasets/Voxel51/AppleGrowthVision
- FSC-147: https://huggingface.co/datasets/isentropic/FSC147 , https://service.tib.eu/ldmservice/dataset/fsc147
