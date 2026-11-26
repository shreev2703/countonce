# DATASETS

Availability, license, format and size for every dataset in `docs/PLAN.md` §9.1.
Researched 2026-11-24 from web search snippets only (no files downloaded, no landing pages read in full).
Anything not confirmed from a primary source is marked `TODO: verify`. Licenses that conflict between
sources are listed as conflicts; resolve against the primary record before any release (PLAN §Track I).

## Summary

| Dataset | Role | Available | License (best evidence) | Size | Tier |
|---|---|---|---|---|---|
| Fuji-SfM | Exact dup/miss measurement; matcher fine-tuning | Zenodo, DOI 10.5281/zenodo.3712808 | **CC BY-NC-SA 4.0** (`LICENSE.txt` and `readme.txt` in the download; research/education only, no commercial use) | ~6.4-6.9 GB | Core |
| RGBD-VideoCount | RGB-D video counting eval | Hugging Face `aerospace123/RGBD-VideoCount` | CC BY 4.0 (HF card) | 2,032 frames, 77,638 boxes; GB `TODO: verify` | Strong |
| VideoCount | Video counting eval (CountVid benchmark) | github.com/niki-amini-naieni/CountVid | `TODO: verify` (repo is MIT; source-video licenses TAO/MOT20 apply) | `TODO: verify` | Strong |
| OneScene-Bench | Multi-view evidence integration (VLM) eval | github.com/swathisuhas/OneScene-Bench (images not in repo) | none reported by GitHub; DL3DV-10K terms | images: `TODO: verify` | Strong |
| WGISD | Grape-cluster detection; secondary orchard data | Zenodo 3361736, GitHub, HF | **Conflict**: CC BY-NC-SA 4.0 (Dataset Ninja) vs "no restrictions, contact Embrapa for commercial" (README) | ~1.3 GB | Optional |
| MinneApple | Apple detection/counting, detector sanity checks | UMN Conservancy handle 11299/206575 | UMN Terms of Use; CC BY-NC-SA 3.0 US per Dataset Ninja (unconfirmed) | ~1-4 GB across archives | Optional |
| AppleGrowthVision | Stereo apple orchard, detection, 3D | Fraunhofer HHI project page; HF mirror `Voxel51/AppleGrowthVision` | Original `TODO: verify`; HF FiftyOne copy says CC BY 4.0 | 9,317 stereo + 1,125 annotated images; GB `TODO: verify` | Optional |
| FSC-147 | Class-agnostic counting reference / detector sanity | Official release (Ranjan et al., CVPR 2021); HF mirrors | Official `TODO: verify`; a HF mirror tags MIT (uploader label, not authoritative) | 6,135 images, ~5.2 GB repo on one mirror | Optional |

## Per-dataset notes

### Fuji-SfM
- **What:** SfM image sequence of 11 Fuji apple trees (582 images), a 3D point cloud of the scanned scene with
  1,455 apple positions as ground truth in global coordinates, and 2D instance masks for 288 images (per PLAN §9.1).
- **Where:** Zenodo DOI 10.5281/zenodo.3712808; Lleida GRAP page `grap.udl.cat/en/publications/fuji-sfm-dataset/`.
  (Verified from the download: record 3712808 holds the whole dataset, a single ~6.9 GB `Fuji-SfM_dataset.rar`.)
- **Layout (verified against the real download, `bench/datasets/fuji_sfm.py`):**
  - `2-SfM-set/raw_images_{east,west}_side/*.JPG`: 291 + 291 frames, 5184x3456. **No camera poses ship**; recover them
    with COLMAP/VGGT and align with Sim(3) (`align_recon_to_gt`). No file maps frames to trees.
  - `3-3D_data/GroundTruth-3D_apples_locations/{training_set,test_set}/TreeNN-annotations_NNNNNN.txt`: one file per apple,
    whitespace-separated xyz rows (its fruit points, on a 5 cm grid). The loader takes the centroid as the apple position and the
    per-axis range as its extent (mean 0.100 m). `NNNNNN` restarts at 0 per tree, so an apple id is `TreeNN/NNNNNN`.
  - `3-3D_data/Fuji_apple_trees_point_cloud.txt` (515 MB scene cloud, not read by the loader).
  - `1-Mask-set/`: 288 image crops (`*.jpg` + per-image `*.csv`) with `training_set_masks.json` / `validation_set_masks.json`; the 12
    full frames they were cut from are in `raw_images/`. They are **not** the SfM frames, so the masks do not align with the 582 images.
- **GT units and frame:** metres, in the global UTM-like frame shifted by T=[-336300, -4623400, 0] (saved = global + T, `3-3D_data/readme.txt`).
  The loader keeps the saved (shifted) frame, which keeps float precision, and exposes `gt_shift`.
- **Per-tree grouping:** yes, from the file name. Dataset `test_set` = Tree01-08 (147/105/135/137/92/133/122/151 = 1,022 apples);
  `training_set` = Tree09-11 (143/128/162 = 433). Total 1,455.
- **Frozen real split:** the dataset's own split. `test_set` trees are `test`, `training_set` trees are `val` (the dev split for any
  threshold), pinned by a sha256 over each tree's GT files in `bench/splits_fuji_sfm.json` (`bench.splits.write_fuji_splits`, `verify_fuji_splits`).
  Only hashes are committed, not data.
- **Size:** ~6.9 GB (Zenodo) / 6.43 GB (Lleida).
- **License: CC BY-NC-SA 4.0** (read from `LICENSE.txt`; research and education only, no commercial use, share-alike). The earlier
  CC BY 4.0 statements on aggregator pages are superseded. Consequences: do **not** redistribute the images or any GT derived from them,
  do not upload them (or derived GT) to a Hugging Face dataset, and do not ship them in CountOnce-Bench; keep `data/fuji_sfm` gitignored.
  Cite Data in Brief (10.1016/j.dib.2020.105591) and Comput. Electron. Agric. (10.1016/j.compag.2019.105165).
- **Decision:** Core benchmark dataset; download script only, no data committed.

### RGBD-VideoCount
- **What:** RGB-D video counting from "Depth-Guided Video Object Counting in Crowded Scenes" (arXiv 2608.06236); Intel RealSense D455;
  2,032 annotated frames, 77,638 instance boxes, multiple categories per video, depth, exemplar boxes, video-level ROIs.
- **Where:** `huggingface.co/datasets/aerospace123/RGBD-VideoCount`; code at `github.com/streamer-AP/DG-Net`.
- **Format:** HF repo is a single ~2.7 GB `RGBD-VideoCount.zip` (file tree listed, nothing downloaded). README gives only the top level: `images/`, `Depth_Data/`, `object_annotations/`, `count_annotations/`, `dataset_split.json`, `video_class.txt`, `exemplars_*.json`; 195 clips, 6 categories. Sub-folder and JSON schemas `TODO: verify`.
- **Loader:** `bench/datasets/rgbd_videocount.py` (scaffold, synthetic-fixture tests only; layout below the top level is assumed). `scripts/download_rgbd_videocount.sh` (not run).
- **License:** CC BY 4.0 (HF card).
- **Size:** `TODO: verify`.

### VideoCount
- **What:** Benchmark from the CountVid paper, built from TAO and MOT20 tracking videos plus penguin videos and X-ray alloy crystallization videos.
- **Where:** `github.com/niki-amini-naieni/CountVid`.
- **License:** `TODO: verify`; derived videos inherit TAO/MOT20 terms. MOT20 is research-use only, `TODO: verify`.
- **Format / size:** `TODO: verify`.
- **Loader:** `bench/datasets/videocount.py` (scaffold; whole layout assumed: `annotations.json` + `frames/<video>/` or `videos/<video>.mp4`). `scripts/download_videocount.sh` needs its `DATA_URLS` filled first (not run).
- **Fit caveat:** crowds, penguins, crystals; not orchard-like. Used only as the Strong-tier external video benchmark.

### OneScene-Bench
- **What:** arXiv 2610.38926, "Do Vision-Language Models Integrate Evidence Across Views?". **Not a counting-only dataset**; its counting part draws on
  DL3DV-10K scenes (object category chosen by Qwen3-VL-32B-Instruct). It is the right probe for the frontier-VLM baseline.
- **Release (checked 2026-11-25):** `github.com/swathisuhas/OneScene-Bench`. Parquet files per split (`counting_{complete,subset,complementary}`, `counting_multi_image`, `consistency`, `spatial_reasoning`); counting columns `id, image1, image2, object_name, answer`.
- **Images are missing:** in the three files inspected, first-row image fields hold a relative path and no bytes, and the repo tree has no image files. Ask the authors or rebuild from DL3DV-10K (`TODO: verify` whether the path-to-frame mapping is recoverable).
- **License / size:** GitHub reports no license; README defers to the DL3DV-10K terms. Image size `TODO: verify`.
- **Loader:** `bench/datasets/onescene.py` (scaffold; its assumed `counting.json` + `scenes/<id>/images/` layout does **not** match the parquet release and needs rewriting once images are available). `scripts/download_onescene.sh` needs its `DATA_URLS` filled first (not run).
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
4. **Open items for the first download session:** (Fuji-SfM is closed: license and layout verified); RGBD-VideoCount size/layout; VideoCount terms; OneScene-Bench images; FSC-147 official terms; AppleGrowthVision original license.

## Sources
- Fuji-SfM: https://zenodo.org/record/3712808 , https://grap.udl.cat/en/publications/fuji-sfm-dataset/ , https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7184157/
- RGBD-VideoCount: https://huggingface.co/datasets/aerospace123/RGBD-VideoCount , https://arxiv.org/pdf/2608.06236 , https://github.com/streamer-AP/DG-Net
- VideoCount: https://github.com/niki-amini-naieni/CountVid
- OneScene-Bench: https://arxiv.org/pdf/2610.38926 , https://github.com/swathisuhas/OneScene-Bench
- WGISD: https://zenodo.org/record/3361736 , https://huggingface.co/datasets/thsant/wgisd , https://datasetninja.com/wgisd
- MinneApple: https://conservancy.umn.edu/handle/11299/206575 , https://arxiv.org/pdf/1909.06441
- AppleGrowthVision: https://arxiv.org/abs/2505.14029 , https://huggingface.co/datasets/Voxel51/AppleGrowthVision
- FSC-147: https://huggingface.co/datasets/isentropic/FSC147 , https://service.tib.eu/ldmservice/dataset/fsc147

## Loader status (video / multi-view)

| Dataset | Loader | Layout | Download script | Status |
|---|---|---|---|---|
| RGBD-VideoCount | `bench/datasets/rgbd_videocount.py` | top level from README, rest assumed | `scripts/download_rgbd_videocount.sh` | scaffold, unverified against real data |
| VideoCount | `bench/datasets/videocount.py` | fully assumed | `scripts/download_videocount.sh` (URLs TODO) | scaffold, unverified |
| OneScene-Bench | `bench/datasets/onescene.py` | fully assumed | `scripts/download_onescene.sh` (URLs TODO) | scaffold, unverified |
| AppleGrowthVision (10 counted trees) | `bench/datasets/applegrowthvision.py::load_applegrowthvision` | fully assumed (`trees/counts.json`, `trees/<id>/images/`) | `scripts/download_applegrowthvision.sh` (mirror `TODO: verify`, not run) | scaffold, unverified; opt-in `datasets:` entry `applegrowthvision` in `configs/bench.yaml` (`enabled: false`), feeds `bench.sweep` / `bench.evaluate` like the other scenes |

## Loader status (single-image)

| Dataset | Loader | Layout | Download script | Status |
|---|---|---|---|---|
| FSC-147 | `bench/datasets/fsc147.py` | fully assumed (`images_384_VarV2/`, `annotation_FSC147_384.json`, `Train_Test_Val_FSC_147.json`, `ImageClasses_FSC147.txt`) | `scripts/download_fsc147.sh` (mirror `TODO: verify`, not run) | scaffold, unverified against real data |
| WGISD | `bench/datasets/wgisd.py` | fully assumed (`data/<stem>.{jpg,txt,npz}`, `train.txt`, `test.txt`) | `scripts/download_wgisd.sh` (HF mirror `TODO: verify`, not run) | scaffold, unverified; license CONFLICT, treated as non-commercial |
| MinneApple | `bench/datasets/minneapple.py` | fully assumed (`{detection,counting}/<split>/{images,masks}/`; count = mask instances) | `scripts/download_minneapple.sh` (bitstream URLs `TODO: verify`, not run) | scaffold, unverified; UMN ToU, non-commercial |
| AppleGrowthVision (annotated images) | `bench/datasets/applegrowthvision.py::load_applegrowthvision_images` | fully assumed (`images/`, `labels/`, `splits.json`) | `scripts/download_applegrowthvision.sh` (not run) | scaffold, unverified; boxes are sparse so counts are lower bounds; original license `TODO: verify` |

Detector sanity only (PLAN §9.1), not part of the dedup bench: `countonce bench detector-sanity --dataset fsc147|wgisd|minneapple|applegrowthvision
--split val --limit N --detector <name> [--root DIR]` yields image, class-name prompt, GT points and exemplar boxes,
counts detections above `--score-threshold`, and appends per-image rows plus one summary row (MAE/RMSE, image-level
bootstrap CI) to `results/detector_sanity.csv` (append-only schema, `bench/detector_sanity.py::SCHEMA`). Results:

TODO: fill from results

Layout assumptions (file names, JSON keys, box corner order, point order) are marked `TODO: verify` in `_read_layout`.
The WGISD / MinneApple / AppleGrowthVision loaders yield `CropImage` (`bench/datasets/crop_common.py`, duck-compatible with `FscImage`; points are box or mask-instance centres) and run the shared validation (`countonce bench validate --dataset wgisd|minneapple|applegrowthvision|applegrowthvision_images`). Each loader module records its license in its docstring and a `LICENSE` constant. These sets have no multi-view identity GT (except AppleGrowthVision's 10 counted trees); they are Track D data, not dedup measurements. MinneApple/WGISD have no `val` split: use `--split train|test`.

Each loader yields `BenchScene`s (frames, GT count, prompt, split). `configs/bench.yaml` `datasets:` entries are
opt-in (`enabled: false`); enabled ones are symlinked into `<csv dir>/datasets/<name>/` as scene dirs
(`images/` + `gt.json`) and swept like any scene, with `max_frames`/`stride` subsampling. `to_gt_doc` writes a GT
file for `bench.evaluate`. All format assumptions are marked `TODO: verify` in each `_read_layout`.
