# CountOnce: Master Project Plan (v6.0)

**Count anything in 3D from a few photos, without counting it twice.**

| | |
|---|---|
| **Owner** | Shreev |
| **Status** | Master plan v6.0. **Start at §0** (what happens next, in order, and the definition of done). §28 is the audit behind it; §17 is the product spec. **Read §28 too:** a brutal audit of the repo as of 2026-11-25 and the recovery order. §17 is now the full product and frontend spec. Earlier changes are in Appendix A |
| **Builds on** | `vggt-sparse-recon` (finished VGGT vs. COLMAP benchmark) and the CountGD++ paper project |
| **Repo** | New repo `countonce` (code ported from `vggt-sparse-recon`; see §4.1) |
| **Hardware** | Laptop with RTX 5070 GPU (8 or 12 GB VRAM, detected at runtime) and AMD Ryzen 9 CPU (core count and RAM detected at runtime); Windows + WSL2; optional pay-per-second cloud GPU |
| **Target roles** | ML engineering / computer vision, ML infrastructure, backend |

---

## Table of contents

0. Master execution plan (start here)
1. Summary
2. Background, related work and novelty
3. Goals, scope tiers and success criteria
4. Rules for Claude Code (read first)
5. Environment
6. System architecture
7. Core data model
8. Track A: Core pipeline
9. Track B: Benchmark and evaluation
10. Track C: Learned matching (CountOnce-Match) on synthetic data
11. Track D: Detector adaptation
12. Track E: Reconstruction backbones
13. Track F: Inference optimization
14. Track G: Web product
15. Track H: Agent and developer integrations
16. Track I: Open source and releases
16B. Additional tracks: J Data engineering, K Reliability/observability/MLOps, L Correction flywheel, M Showcase
17. Product and frontend spec
18. Testing strategy
19. Phases, tasks and acceptance criteria
20. Human tasks (things only Shreev can do)
21. Decision gates
22. Risks and mitigations
23. Documentation: README, EXPLAINER, paper
24. Licenses and data handling
25. Timeline and weekly time budget
26. Deliverables checklist and resume entries
27. Glossary
28. Brutal audit (2026-11-25) and recovery plan

Appendix A: What changed in v4, v4.1 and v4.2, and why

---

## 0. Master execution plan (v6.0): how this project gets finished

This section is the single source of truth for **what happens next, in what order, and when the project counts as done**. Every finding from the internal audit (§28) and the external review (§28.6b) maps to a task below (coverage matrix in §0.7). Agents run the `M` tasks from `TASKS_OVERNIGHT.md` in order; Shreev does the `H` tasks; the `P` tasks run after the photos exist.

### 0.1 Ground rules for finishing
1. **One reliable path first:** GroundingDINO (HF) → VGGT → v2 (or sync if it wins on validation). Everything else is labelled "planned" in `docs/STATUS.md`.
2. **No new subsystems.** Only the tasks in this section. §28.4's recovery order still applies.
3. **No placeholder numbers in the product.** Pages without real results show an honest empty state ("Benchmark in progress"), never `TODO` text or made-up examples.
4. **Fuji-SfM images are never shown publicly** (CC BY-NC-SA: no redistribution of images or derived GT). Fuji numbers can appear in the benchmark; Fuji photos can't appear in demo mode or the README.
5. **Design review moves after the build:** the agents build the frontend to §17 tonight; Shreev reviews it in the morning (H2) and requests changes as follow-up tasks. This replaces the "approve mockups before any frontend code" gate for this pass.
6. **Agent files stay local-only:** `TASKS_OVERNIGHT.md`, `CLAUDE.md` and the orchestrator scripts keep working on the laptop but are untracked from the public repo (task M21), never moved, so the orchestrator doesn't break.

### 0.2 Tonight: agent tasks (no photos needed), in order
| ID | Task | Done when |
|---|---|---|
| **M1** | Make GroundingDINO (HF) the `Config` default; CountGD/CountGD++ marked "planned" with a clear error; write `docs/STATUS.md` (component × implemented / synthetic-tested / real-model-tested / held-out-real-benchmarked) | The README quickstart command works from a clean venv; README and STATUS never contradict each other |
| **M2** | Debug the tiny-scene over-count (real detector + VGGT gives 7, truth 5): check per-frame duplicate boxes (add per-frame NMS), lifting region/background pixels, merge radius; check the "five views deduplicated to one" smoke-check note | Real path counts 5 on the tiny scene; GPU-marked regression test; root cause written in DEVIATIONS |
| **M3** | Count-first metrics: signed bias, zero-scene false count, and the error decomposition (missed + duplicates + false positives − wrong merges = predicted − true) in `bench/metrics.py`, wired into reports and figures (§9.4) | Unit tests on toy inputs; reports lead with count metrics; F1 reported only as an explanation |
| **M4** | True photo-order invariance test: permute photo-to-frame assignment and camera arrays (≥10 permutations); measure count and partition variation for v1/v2/sync/v3; fix v2 (global refinement) or narrow the claim | Test exists; result and decision in DEVIATIONS |
| **M5** | Synthetic eval upgrade: add a finite noisy-depth variant (visibility check active), count injected false positives, tune v1/v2/sync on the synthetic validation split, re-run the v3 comparison; label it "component test" in `docs/RESULTS_V3_SYNTH.md` | Table regenerated with tuned baselines and count-first metrics |
| **M6** | Controlled failure-source harness (§9.6): oracle detections + estimated geometry; real detections + oracle geometry; full pipeline; identical upstream inputs for every associator | Runs on the tiny scene and a synthetic fixture; ready for Fuji and own captures |
| **M7** | Download Fuji-SfM (`scripts/download_fuji_sfm.sh`, research use under CC BY-NC-SA), confirm layout, close the loader TODOs; if the download fails, log it in GATES and continue | Loader passes on real files |
| **M8** | Fuji-SfM Core run: freeze settings on validation trees, then run the Core sweep on the frozen test trees; keep failed rows; then the existing report task (`bench.fuji_report`, figures, failure analysis) and the M6 experiments on Fuji | `docs/RESULTS_FUJI.md` and the README Fuji table filled from `results/bench.csv`; gate G3 decided in GATES |
| **M9** | Fix CI: the `db` extra (add it to `pyproject.toml` or drop it from `ci.yml`); green on `main` locally with `act` or a dry run; add a Node job for `web/` | CI config consistent; badge expected green after push |
| **M10** | Read CountVid, DG-Net and OneScene-Bench in full (arXiv); rewrite `docs/RELATED_WORK.md` around the §2.4 hypothesis; remove unverified "does not do X" claims | No unverified claims left |
| **M11** | Capture kit: `docs/CAPTURE_GUIDE.md` (one printable page: what to shoot, how, the ground-truth definition, the hand-count procedure), a `scripts/ingest_captures.py` that turns a phone folder into `data/scenes/<id>/{images/, gt.json}` with EXIF stripping, and a scene list template covering ≥3 zero-target scenes | Running the script on a test folder produces a valid scene |
| **M12** | Frontend scaffold `web/`: Next.js App Router + TypeScript strict + Tailwind + shadcn/ui + react-three-fiber + TanStack Query; design tokens (§17.10); OpenAPI → TypeScript types and Zod `result.json` schemas by `schema_version`; Vitest + Playwright + axe set up | `npm run build` and tests pass |
| **M13** | Demo exporter `scripts/build_demo.py`: runs the real pipeline on demo scenes and writes static assets (result, compact binary point cloud, markers, overlays, thumbnails, match evidence, provenance); starts with the tiny scene, labelled "synthetic test scene" | Assets validate against the Zod schemas; a CI check fails if versions don't match |
| **M14** | Results experience (§17.5) on demo assets: count card, 3D viewer (LOD point cloud, numbered markers, camera frusta with fly-to, match lines, two-way highlighting, keyboard navigation), Photos / Explain / Warnings / Details tabs, phone layout | Works on the tiny-scene demo with no backend; Playwright test |
| **M15** | Landing page and demo gallery (§17.2–17.4), with proof points as honest empty states until real numbers exist | Lighthouse ≥ 90 on performance, accessibility and best practices for `/` |
| **M16** | Signed-in flow wired to the existing API: Google sign-in, `/new` (capture guide, camera input, browser pre-flight checks, resumable multipart upload, prompt + negative prompt + example box, idempotent submit), `/jobs/[id]` progress over SSE with polling fallback and the paused-worker banner, `/history`, share links `/s/[token]` with Open Graph images, corrections and feedback | Playwright end-to-end test against a mocked API and against a local API + fake worker |
| **M17** | `/benchmark` (reads `results/*.csv` at build time; empty state when missing), `/how-it-works`, `/status`, `/about`, `/privacy` | Pages build; no `TODO` strings in production output (build check) |
| **M18** | API fit for Vercel: move the API to its own Vercel project or path (`api.` or `/api`); make sure long work never runs in a serverless function (the worker does it); SSE works or falls back to polling; document env vars | Frontend talks to the deployed API; JSON root no longer the main link |
| **M19** | Worker hardening for the pilot: laptop worker as a service script, heartbeat, typed errors + dead-letter, retention job; one command to start it | Local end-to-end: upload from the frontend → worker on GPU → result page |
| **M20** | README front page rewrite (§28.6b): input photos, predicted vs. hand count, instance IDs across views, one measured table, one failure, one runnable command, link to the live demo and STATUS; Roadmap section below | No TODO tables on the front page |
| **M21** | Repo hygiene (§28.5): Apache-2.0 license for code (weights and datasets keep their own); untrack `TASKS_OVERNIGHT.md`, `CLAUDE.md`, orchestrator scripts and handoff paths (`git rm --cached` + `.gitignore`, files stay local); remove personal paths from tracked files | `git ls-files` shows no agent files or personal paths |
| **M22** | Final consistency pass: PLAN, STATUS, README, EXPLAINER and the site agree; EXPLAINER interview Q&A updated for M2–M8 findings | Checklist in §0.6 rows that don't need photos are ticked |

**Honest expectation:** M1–M9 and M12–M15 are realistic overnight; M16–M22 may spill into a second night. The product shell will be finished before the photos; the product is only *finished* once P1–P5 put real results in it.

### 0.3 Tomorrow: Shreev's tasks
| ID | Task | Details |
|---|---|---|
| **H1** | **Capture day** | Follow `docs/CAPTURE_GUIDE.md` (M11). Target 20–30 static scenes: bike racks, chairs, shelf items, cars in a small lot, potted plants, bins/cones, fruit on a tree or vine if available, plus **≥3 scenes with none of the target object**. Per scene: a 30–60 s video on an arc + ~20 stills. Hand-count from the photos, twice. Ask permission on private property; avoid faces. |
| **H2** | **Review the built frontend** | Open the deployed site on your phone and laptop; list changes; they become follow-up tasks |
| **H3** | **Interviews (Phase 0)** | 3–5 people for the ag niche + 2 backup; questions in §20 |
| **H4** | **Professor meeting** | Bring the §2.4 hypothesis, the Fuji results from M8 and the failure decomposition |
| **H5** | **Accounts** | Vercel project for `web/`, Google OAuth client, R2 (or the free fallback in §14.7), Neon; put secrets only in the hosts' env settings |
| **H6** | **Ownership study** | Read the core files in §28.4 Step 6, starting with whatever M2 changed |

### 0.4 After the photos: agent tasks
| ID | Task | Done when |
|---|---|---|
| **P1** | Ingest the captures (M11 script), validate every `gt.json`, split by physical scene into calibration/validation/test, freeze `bench/splits.json` | Splits frozen before any tuning |
| **P2** | Core benchmark on own captures + the M6 failure-source experiments + zero-target false counts + order-invariance on real data | Results tables and figures filled from `results/bench.csv` |
| **P3** | Fix the largest error source the decomposition shows (detection, geometry or association), re-run on validation only, then one final test run | Before/after on validation; one test-split run |
| **P4** | Replace the tiny scene in demo mode with 6–10 own captures (M13 exporter); regenerate landing proof points from real results | Demo shows real scenes and real numbers |
| **P5** | Fill README, EXPLAINER, `docs/RESULTS_*`, the LinkedIn/resume entries (§26) and the benchmark page from results | No `TODO: fill from results` left in Core docs |
| **P6** | Then Strong tier in order: rendered synthetic data (Kubric/BlenderProc) → v3 retrain → real-data v3 evaluation; VLM baseline (free tiers only); conformal intervals once ≥15 real calibration scenes exist | Per §19 |

### 0.5 Pilot (after P5)
Recruit 3–10 users from H3 interviews; run 2–4 weeks with the laptop worker; `docs/PILOT.md` filled from the job store (§19 Phase 5).

### 0.6 Definition of done ("project wrapped up")
Checked in the M22 consistency pass (2026-11-25). A box is ticked only where the evidence meets the row as written; notes say what is left. Component evidence is in `docs/STATUS.md`.

**Core done (required):**
- [x] One reliable path works from a clean install; STATUS table accurate — *a clean `uv` venv (not plain `pip`) with the README packages runs the real path, including all five views on the 8 GB WSL2 GPU*
- [x] Tiny scene counts correctly with the real detector + VGGT (single-frame path, `score_threshold=0.5`; see DEVIATIONS)
- [ ] Fuji-SfM held-out results with CIs, error decomposition and gate G3 decided — *results, CIs and a k=10 decomposition exist; G3 is recommended "not met" and awaits the owner*
- [ ] 20–30 own scenes (≥3 zero-target) benchmarked with frozen splits
- [ ] Failure-source experiments show where the remaining error comes from — *Fuji: detector, then duplicates; geometry is not separated from association (no oracle geometry)*
- [ ] Order-invariance measured on real data; the "unordered photos" claim matches the evidence — *measured on Fuji validation windows (`results/order_invariance_fuji.csv`): v1 invariant; v2 and sync invariant only vacuously (no merges at the frozen gate), so the claim is not yet supported for them; needs a scene where v2 merges*
- [x] README front page with a real result, a real failure and a runnable command — *the real result is the Fuji-SfM table; the picture example is synthetic*
- [ ] Related work read in full; one tested hypothesis instead of a novelty claim — *three closest works read in full, others not; the hypothesis is untested*
- [ ] Repo hygiene done (license, no agent files or personal paths, green CI) — *license done; `CLAUDE.md` still tracked (GATES, M21); hosted CI not run on this branch*

**Product done (required):**
- [ ] Live site: landing, demo gallery with real own-capture scenes, results experience with the 3D fly-to and "Explain this match", benchmark page with real numbers
- [ ] Signed-in flow works end to end from a phone: upload → laptop GPU → result
- [x] Playwright, axe and Lighthouse checks pass; no placeholder text in production — *local runs, not hosted CI or a real phone; Lighthouse median performance: `/` 0.95, `/demo/tiny` 0.93 (thin margin: an intermediate build had runs at 0.82-0.84, see DEVIATIONS), `/benchmark` 0.94; `/jobs/[id]` and `/s/[token]` not measured*
- [ ] 2-minute demo video

**Strong and Moon:** upgrades after the above, in tier order (§3.1, §19).

### 0.7 Coverage matrix (every finding → task)
| Finding (source) | Task |
|---|---|
| Default detector is the CountGD stub (review #1) | M1 |
| Real path counts 7 vs. 5 on the tiny scene (audit, review #5) | M2 |
| "Five views deduplicated to one" smoke check (review #5) | M2 |
| F1 isn't the product metric; count error and decomposition (review #2) | M3 |
| F1 ignores unlabelled false positives (review #2) | M3, M5 |
| NaN depth disables the visibility check in the synthetic test (review #3) | M5 |
| v1/v2 compared with untuned defaults (review #3) | M5, M8 |
| Failure-source experiments (review #3) | M6, P2 |
| Photo-order invariance not really tested (review #4) | M4, P2 |
| README vs. PROFILE contradiction; one status table (review #5) | M1, M22 |
| CI `db` extra missing (review #5) | M9 |
| Novelty ahead of the literature (review #6) | M10, §2.4 |
| README front page is placeholders (review #7, audit §28.5) | M20, P5 |
| Define the target count; 20–30 scenes incl. zero-target; split by scene (review "next steps") | M11, H1, P1 |
| Freeze settings before testing; report failures (review "next steps") | M8, P1–P3 |
| Publish evidence: configs, result rows, failure gallery, demo (review "next steps") | M8, M13, P4–P5 |
| No real-data results (audit §28.1) | M7–M8, P1–P2 |
| No frontend; JSON root URL (audit §28.2) | M12–M18 |
| Scaffolding before Core; scope discipline (audit §28.3) | §0.1 rule 2 |
| Code ownership in interviews (audit §28.4 Step 6) | H6, M22 |
| License, agent files, personal paths (audit §28.5) | M21 |
| Learned matcher only shown on synthetic geometry (audit, review #3) | M5 (honest label), P6 |
| Product spec §17 (frontend) | M12–M17 |
| $0 deployment (§14.7) | M18, H5 |

---

## 1. Summary

A user uploads a few photos or a short phone video and types what to count ("apples", "grape clusters", "bikes", "pallets"). CountOnce returns:

- **One deduplicated count** with a confidence range
- **A 3D view** of the scene with a marker on every counted object
- **Each photo** with its detections drawn on, colored **new**, **duplicate** (already counted from another photo) or **rejected**
- **Warnings** when the input is likely to give a bad result, with a suggested fix

**How it works:** an open-vocabulary counter finds the objects in each photo. A feed-forward 3D model (VGGT, or newer models) reconstructs the scene from the same photos in seconds, with no depth sensor and no slow optimization. Every detection is lifted into 3D, and detections of the same physical object are associated across photos, first with geometric rules and then with a **learned matching model trained on synthetic multi-view scenes**, so each object counts once.

**What makes it a "moon" project:** it combines two top-conference computer vision models, a new learned component trained on generated data, a real benchmark on real orchards, GPU inference optimization, and a deployed product with real users, plus open-source releases and a paper.

---

## 2. Background, related work and novelty

### 2.1 The problem
- When counting across several images, the same object appears in multiple frames and gets counted more than once. Fruit-counting research calls this the main challenge of counting from image sequences.
- Single-view counting **underestimates** (hidden objects), while naive multi-view counting **overestimates** (duplicates).
- Classic systems combine detection, tracking and Structure-from-Motion (COLMAP) to remove double counts. Examples: *Robust Fruit Counting* (Liu et al., 2018); SfM + bipartite matching for apples (CVPR 2024 Vision4Ag workshop); multi-view orchard yield mapping. These are typically **crop-specific** and depend on SfM, which is slow and fails with few views.

### 2.2 Building blocks
| Component | What it is | Notes |
|---|---|---|
| **VGGT** (CVPR 2025 Best Paper) | Feed-forward transformer: cameras, depth, point maps, tracks from 1–hundreds of images | github.com/facebookresearch/vggt. Original checkpoint non-commercial; separate commercial checkpoint by application |
| **VGGT-Ω** (CVPR 2026 oral) | Successor: about 30% of VGGT's memory, ~1.6× faster inference, better accuracy | github.com/facebookresearch/vggt-omega. Gated weights; FAIR non-commercial research license. Checkpoints include 512, 416-reproduction and 256-text-alignment |
| **VGG-T³** (CVPR 2026, NVIDIA) | Linear-time global attention via test-time training; scales to ~1k images | Code github.com/nv-dvl/vgg-ttt; model huggingface.co/nvidia/vgg-ttt; check license |
| **CountGD** (NeurIPS 2024) | Open-world counting from text and/or exemplar boxes, built on GroundingDINO | github.com/niki-amini-naieni/CountGD, MIT license, HF demo |
| **CountGD++** (CVPR 2026) | Adds negative prompts and pseudo-exemplars | github.com/niki-amini-naieni/CountGDPlusPlus. Check license and weights |
| **GroundingDINO** | Open-vocabulary detector | Fallback detector |
| **DINOv2** | Self-supervised image features | Appearance features for matching |
| **Kubric** (CVPR 2022, Google) | Open-source synthetic data pipeline (PyBullet + Blender) with instance masks, depth, camera poses | github.com/google-research/kubric; runs in Docker; uses Blender 2.93 (may not GPU-render on RTX 50-series; CPU rendering or BlenderProc as fallback) |
| **BlenderProc** (DLR) | Procedural Blender pipeline: RGB, depth, normals, instance segmentation, camera poses; physics; Python API | Plan B for synthetic data if Kubric's Blender 2.93 fights the RTX 5070; GPLv3 (fine for generating data) |

### 2.3 Closest work (must be compared or explained)
| Work | What it does | How CountOnce differs |
|---|---|---|
| **CountVid** (Oxford VGG, AAAI 2026) | Open-world counting in **video**: image counter + promptable video segmentation/tracking. Releases the **VideoCount** benchmark (370 videos: TAO-Count, MOT20-Count, Science-Count) | Associates through camera geometry estimated from **unordered sparse photos** instead of mask propagation through video; whether that counts better is the §2.4 hypothesis |
| **DG-Net / Depth-VIC** (ACM MM 2026) | Depth-guided **video** object counting in crowded scenes: CountGD-Box-initialised RGB-D detector plus depth-aided tracking, with **sensor depth** (RGB-D) or, in some reported rows, **estimated depth**. Takes a video-level ROI. Releases **RGBD-VideoCount** (195 clips, 2,032 annotated frames, 77,638 boxes) | Unordered photos; association in a **global 3D coordinate frame** with estimated camera poses; learned matcher trained on synthetic data. RGB-only input alone is not a difference (compare against DG-Net with estimated depth) |
| **VGGT-Det** (2026) | Indoor 3D object **detection** using VGGT internals, fixed categories | Counting, open-vocabulary, outdoor/agricultural scenes |
| **ROM** (ICRA 2023) | Learned object matching between image **pairs**: attentional GNN over appearance + spatial features, Sinkhorn with a dustbin, trained on synthetic renders | CountOnce's learned matcher extends this idea to **many views** with **3D geometry from feed-forward reconstruction**, applied to counting |
| **C-DOG** (2025) | Training-free multi-view multi-object association via connected δ-overlap graphs (epipolar geometry, featureless) | A strong geometric baseline for association to compare against |
| **Permutation synchronization / CLEAR** (NeurIPS 2013; CLEAR, MIT) | Turn noisy pairwise matches into cycle-consistent multi-view matches (spectral methods); CLEAR also enforces "distinctness" (items in the same view can't match), which is CountOnce's cannot-link rule | Used as an inference-time step on top of CountOnce-Match, and as a baseline |
| **OneScene-Bench** (NeurIPS 2026 workshop) | Tests whether VLMs integrate evidence across views, including **unique object counting** (two views); counting accuracy is 4.5–50.0% on complementary views vs. 45.5–81.5% on complete-overlap views | Shows VLMs struggle with cross-view counting; used as a VLM baseline reference and an extra evaluation set. Images are not in the public repo (see DATASETS) |
| **SfM fruit counting** | Detection + tracking + COLMAP | Open-vocabulary, feed-forward, few views |
| **3D counting of stacked objects** (EPFL) | Counts hidden stacked items from volume × occupancy | Complementary (counts visible distinct instances) |

### 2.4 Novelty claim (narrowed after research)
The related work shows that **video counting with de-duplication exists** (CountVid; DG-Net with sensor depth). CountOnce's claim is narrower and must be confirmed in Phase 0:

> **Open-vocabulary object counting from a few unordered RGB photos, with de-duplication in a 3D frame recovered by feed-forward reconstruction (no depth sensor, no SfM), and a multi-view learned association model trained on synthetic data that transfers to real orchards.**

**Primary hypothesis (v5.1, tested before any novelty claim is made):** *geometry-aware association improves unique-object counting from sparse views compared with video tracking and with simpler geometric matching.* CountVid, DG-Net and OneScene-Bench were read in full on 2026-11-25; `docs/RELATED_WORK.md` tags the evidence level of every other work and makes no "does not do X" claim about them.

The experiments must show where this beats the closest work: few views, unordered photos, large viewpoint changes, no depth sensor. They must also report honestly where video tracking (CountVid) or sensor depth (DG-Net) wins.

---

## 3. Goals, scope tiers and success criteria

### 3.1 Scope tiers
Everything is labeled with a tier. **Core must be finished before Strong starts; Strong before Moon.**

| Tier | Contents |
|---|---|
| **Core** | Pipeline with geometric merge (v1 + v2), benchmark on own captures + Fuji-SfM, baselines incl. CountVid, **frozen test split**, **bootstrap confidence intervals on every reported metric**, README + EXPLAINER. **Nothing else is added to Core.** |
| **Strong** | Learned matcher (synthetic data) **+ cycle-consistent synchronization**; **frontier VLM baseline**; **split conformal intervals**; **error slicing**; RGBD-VideoCount/VideoCount/OneScene-Bench evaluation; **sharded dataset + catalog + data-quality gates**; inference optimization report; deployed web app with **resumable uploads, idempotent submits, client-side pre-flight check and "Explain this match" inspector**; pilot users; **human baseline (time + accuracy)** |
| **Moon** | VGGT-Ω/VGG-T³ backbones; detector fine-tuning; **custom Triton kernel**; **cold-start-aware cloud serving**; **OpenTelemetry tracing**; **user-correction active-learning flywheel**; **Gaussian splat viewer toggle**; agent/MCP integration; releases (pip, HF Space, model, dataset, **public benchmark + leaderboard**); **harvest/yield validation with a partner**; workshop paper; CountGD++ swap-in |

### 3.2 Goals
1. Deduplicated counts from 3–40 photos or a ≤60 s video, for any text prompt
2. A rigorous benchmark against clear baselines, on real data with ground truth
3. A learned association model that beats the hand-tuned geometric merge on real data after training on synthetic data
4. Measured inference speedups on a consumer GPU
5. A polished deployed web app with real pilot users
6. Public releases and a paper-quality write-up

### 3.3 Non-goals
- Training VGGT or the counter from scratch (fine-tuning the detector is a Moon-tier option)
- Metric sizes in real-world units (VGGT scale is relative)
- Native mobile app, payments, multi-tenant scale
- Counting fully hidden objects

### 3.4 Success criteria
| Tier | Criterion |
|---|---|
| Core | 3D dedup beats "sum of per-image counts" and "best single image" on most scenes; honest comparison with CountVid; failure analysis |
| Strong | Learned matcher beats geometric merge on Fuji-SfM and own captures (sim-to-real transfer); app deployed with real pilot usage; measured speedup ≥1.5× end-to-end from optimization |
| Moon | Paper submitted; public releases with outside usage (downloads, stars, Space visits) |

---

## 4. Rules for Claude Code (read first)

1. **Read this whole file before writing code.** Work phase by phase (§19). Don't start a phase until the previous one meets its acceptance criteria, and finish a tier before starting the next.
2. **The real repos win.** When this plan and the real code (VGGT, VGGT-Ω, CountGD/CountGD++, GroundingDINO, CountVid, DG-Net, Kubric) disagree on names, formats or arguments, follow the source and log the difference in `docs/DEVIATIONS.md`.
3. **Never make up numbers.** Every number in the README, figures, EXPLAINER and paper comes from a script in this repo that was actually run. Use `TODO: fill from results` until then.
4. **Deterministic and resumable.** Seed everything; cache expensive outputs (detections, reconstructions, features) keyed by input hash + model version; resumable benchmarks and training.
5. **Small, typed, tested modules.** Type hints, `pytest` + `ruff` clean, no giant notebooks.
6. **Stop at gates.** If a gate (§21) fails, write it up in `docs/GATES.md` and wait for a decision.
7. **Track experiments.** Every training run and benchmark run logs config, git commit, data version and metrics (§10.6).
8. **Commits:** clear normal messages, one logical change per commit.
9. **Recovery mode (v5.0):** follow §28.4 in order. No new modules outside the current recovery step; bug fixes, evaluation and docs only.
10. **Real-data check required:** a feature tested only on synthetic or tmp fixtures stays open, not done.
11. **No scaffolding for a tier that hasn't started** (§3.1).

### 4.1 Repo decision
Use a **new repo `countonce`**. The VGGT benchmark is a finished project with its own results and README; keeping it separate means both projects stand on their own on GitHub and LinkedIn. Port the reusable code (env check, VGGT wrapper, memory guard, frame extraction, COLMAP wrapper, plotting style) into `countonce`, credit the source in comments, and link the two READMEs to each other. This plan file can live in the VGGT folder until the new repo exists, then moves to `countonce/docs/PLAN.md`.

---

## 5. Environment

- **OS:** Linux, or **WSL2 Ubuntu** on Windows
- **Python:** 3.11; dependencies pinned once working
- **PyTorch:** built for **CUDA 12.8+** (RTX 50-series needs `sm_120`); verify with `torch.cuda.get_arch_list()`
- **Precision:** bf16 autocast, fp16 fallback
- **VRAM:** detected at runtime; reuse the memory calibration from the VGGT project to cap frames per reconstruction call
- **GPU sharing:** detector and reconstructor run sequentially by default, freeing memory between stages
- **Docker:** required for Kubric (synthetic data); optional for the worker
- **CPU: AMD Ryzen 9 (laptop).** Detect core count and RAM at runtime (`os.cpu_count()`, `psutil`) and record them in `results/env.json`. Use the CPU for everything that isn't a neural network:
  - **Synthetic rendering:** if Kubric can't GPU-render on the RTX 5070, CPU rendering across many cores is a real option, not just a fallback. Render several scenes in parallel (one process per scene, threads per process tuned to the core count) and measure scenes per hour on CPU vs. GPU in `docs/SYNTH.md`.
  - **Overlap CPU and GPU work:** rendering, frame extraction and validation on CPU while the GPU runs detection/reconstruction on already-rendered scenes (a producer–consumer queue between the two).
  - **COLMAP baseline:** pycolmap feature extraction and matching use many CPU threads; set the thread count explicitly.
  - **Data pipeline:** ffmpeg decoding, blur/duplicate filtering, sharding, DuckDB queries, and PyTorch `DataLoader` workers (start with workers ≈ physical cores / 2).
  - **Statistics:** run bootstrap resampling and slice analysis in parallel across cores.
  - **Worker preprocessing:** decode, resize and sanitize uploads on CPU threads so the GPU never waits.
- **Laptop thermals and power:** running the GPU and all CPU cores at full load at the same time can make a laptop throttle. Run long jobs plugged in on the highest performance power mode, log CPU/GPU temperatures and clocks during long renders and training runs, and cap CPU render threads if the GPU slows down. Long unattended runs must checkpoint and resume (the plan already requires resumable runs).
- **WSL2 storage:** keep the repo, active synthetic scratch space, shards, caches and the Docker data on the **native WSL2 ext4 filesystem** (e.g., `~/countonce`, `~/data`), **not under `/mnt/c`**. Windows drives are reached through a slow 9P file-sharing layer, so heavy small-file I/O (rendering outputs, sharding, data loading) there is many times slower. Only copy finished artifacts to `/mnt/c` if Windows tools need them. Check free space inside the WSL2 virtual disk, not just on `C:`.
- **WSL2 RAM and CPU limits:** WSL2 only gets the memory and processors its config allows. Set them in `%UserProfile%\.wslconfig`, e.g.:
  ```ini
  [wsl2]
  memory=24GB        # leave a few GB for Windows; adjust to the laptop's RAM
  processors=16      # match the Ryzen 9's thread count, or a bit less
  swap=16GB
  ```
  Then run `wsl --shutdown` to apply. `00_env_check.py` reports the memory and CPUs WSL2 actually sees, so a misconfigured limit is caught early.
- **RAM and disk:** record total RAM and free disk space. Synthetic shards and caches can take hundreds of GB, so check free space before generating, and keep raw renders on the largest/fastest drive (or mirror to R2).
- `scripts/00_env_check.py` reports GPU, VRAM, CUDA arch, library versions and whether each model's weights load

---

## 6. System architecture

### 6.1 Pipeline
```
photos / video
     │
     ▼
[Ingest] frames, blur filter, near-duplicate removal, resize
     │
     ├───────────────────────────────┐
     ▼                               ▼
[Detect]                        [Reconstruct]
 open-vocab counter              VGGT / VGGT-Ω / VGG-T³
 → boxes/points per image         → cameras, depth, point maps, confidences
     │                               │
     └──────────────┬────────────────┘
                    ▼
              [Lift to 3D]
        detection → 3D point, size, confidence
                    ▼
              [Associate]
        v1: constrained 3D clustering
        v2: geometric cross-view matching
        v3: learned matcher (CountOnce-Match)
                    ▼
        [Confidence range]   [Quality warnings]
                    ▼
                [Export]
   count, range, instances, overlays, 3D markers, JSON/CSV
```

### 6.2 Training (Track C)
```
Kubric scene generator ──► synthetic multi-view scenes (images, masks, depth, cameras, instance IDs)
          │
          ▼
  run Detect + Reconstruct + Lift on synthetic images (same pipeline as real data)
          │
          ▼
  training examples: lifted detections + true instance IDs
          │
          ▼
  CountOnce-Match (multi-view attentional GNN + Sinkhorn)  ──► evaluate on Fuji-SfM + own captures
```

### 6.3 Product
```
Browser ──► API (FastAPI) ──► Postgres (users, jobs, results, feedback)
   │            │                   ▲
   │ signed URLs│                   │ claim jobs (SKIP LOCKED), heartbeats
   ▼            ▼                   │
Object storage (R2/S3) ◄──── GPU worker(s): laptop first, cloud GPU (scale to zero) later
```

---

## 7. Core data model

`countonce/types.py` (dataclasses or pydantic), used everywhere:

```python
@dataclass
class Frame:
    id: int
    path: Path
    width: int
    height: int
    sha256: str  # content hash (stage caching)
    privacy_mask: np.ndarray | None  # HxW bool, blurred faces/plates (§14.8)
    blur_score: float


@dataclass
class Detection:
    id: str  # f"{frame_id}:{k}"
    frame_id: int
    box_xyxy: np.ndarray | None  # (4,) pixels
    point_xy: np.ndarray | None  # (2,) pixels
    mask: np.ndarray | None  # HxW bool
    score: float
    feature: np.ndarray | None  # appearance embedding (DINOv2), filled on demand


@dataclass
class Recon:
    backbone: str  # "vggt" | "vggt_omega" | "vggt_t3" | "colmap"
    intrinsics: np.ndarray  # (N,3,3)
    extrinsics: np.ndarray  # (N,3,4) world->camera
    depth: np.ndarray  # (N,H,W)
    depth_conf: np.ndarray  # (N,H,W)
    world_points: np.ndarray  # (N,H,W,3)
    world_points_conf: np.ndarray  # (N,H,W)
    scene_scale: float
    runtime_s: float
    peak_vram_mb: float


@dataclass
class LiftedDetection:
    det: Detection
    xyz: np.ndarray | None
    size: float | None
    lift_conf: float
    motion_residual: float | None  # cross-view reprojection residual (§8.3b)
    fail_reason: str | None  # "low_conf" | "privacy_mask" | "too_few_pixels" | None


@dataclass
class MatchEvidence:  # powers "Explain this match" (§14.8)
    det_a: str
    det_b: str
    dist_3d_norm: float  # 3D distance / mean size
    reproj_error_px: float
    appearance_sim: float
    match_prob: float  # calibrated (temperature-scaled) probability
    merged: bool
    reason: str  # e.g. "same_frame_cannot_link", "below_threshold"


@dataclass
class Instance:
    id: int
    xyz: np.ndarray
    size: float
    members: list[str]
    n_views: int
    conf: float  # calibrated (§10.5)
    possibly_moved: bool  # dynamic-scene flag (§8.3b)


@dataclass
class Interval:
    low: int
    high: int
    level: float  # e.g. 0.90
    method: str  # "sensitivity" | "split_conformal" | "jackknife_plus"
    n_calibration: int | None


@dataclass
class ModelVersions:
    detector: str  # name + checkpoint hash
    backbone: str
    associator: str  # "v1" | "v2" | "v3@<checkpoint>"
    calibration: str | None  # temperature/conformal calibration version


@dataclass
class CountResult:
    schema_version: str  # "1.0", bumped on any change (§8.10)
    prompt: str
    negative_prompts: list[str]
    count: int
    interval: Interval
    zero_count: bool  # true when no target objects were found
    instances: list[Instance]
    per_frame_counts: dict[int, int]
    detection_status: dict[str, str]  # "new" | "duplicate" | "rejected"
    reject_reasons: dict[str, str]  # detection id -> reason
    match_evidence: list[
        MatchEvidence
    ]  # stored for the inspector (may be pruned to top-k per instance)
    warnings: list[dict]  # {code, message, fix}
    model_versions: ModelVersions
    input_hash: str  # content-addressable cache key
    recon_cache_hit: bool
    oom_fallback_rung: int  # 0 = no fallback used (§13 item 6c)
    timings: dict[str, float]
    peak_vram_mb: float
    config: dict
```

Match `Recon` to each backbone's real outputs; convert pose encodings with each repo's own utilities.

---

## 8. Track A: Core pipeline (Core tier)

### 8.1 Ingest (`countonce/ingest/`)
- Video → frames with ffmpeg at a configurable fps (default 2)
- Blur filter (variance of Laplacian), near-duplicate removal (downsampled image similarity)
- Frame cap from VRAM calibration; evenly spaced selection when over the cap
- Resize to the backbone's expected input; keep the mapping to original pixels for overlays

### 8.2 Detect (`countonce/detect/`)
- `base.py`: `Detector.detect(frames, prompt, exemplars=None) -> list[Detection]`
- `countgd.py` (default), `countgd_pp.py` (Moon tier), `groundingdino.py` (fallback)
- Handle the real output format: boxes directly; if points/density only, convert peaks to points and estimate size from exemplars or defaults
- Store **all raw detections with scores** so thresholds can be swept without rerunning
- Cache per (frame hash, prompt, detector, version)
- Optional `masks.py`: SAM-style masks from boxes (ablation)

### 8.3 Reconstruct (`countonce/recon/`)
- `base.py`: `Reconstructor.reconstruct(frames) -> Recon`
- `vggt_recon.py` (ported, with memory guard), `vggt_omega_recon.py` and `vggt_t3_recon.py` (Track E), `colmap_recon.py` (baseline)
- Chunking: if frames exceed the VRAM cap, reconstruct overlapping chunks and align them with Sim(3) on shared frames (test it; document as a limitation)
- **Pose-graph stitching (Moon):** for long captures (60–150+ frames, e.g., a whole orchard row), sequential chunk chaining accumulates scale and rotation drift. Build a Sim(3) pose graph (nodes = chunks, edges = Sim(3) alignments from shared frames, plus extra edges between non-adjacent chunks that share views, acting as loop closures) and optimize it jointly (e.g., with a small least-squares solver or GTSAM/theseus if practical). Measure drift with and without it on Fuji-SfM's full sequence.

### 8.3b Dynamic scenes and moving objects (Moon)
VGGT assumes a static scene. Wind-blown branches, people walking through, or a bike being moved between photos can corrupt poses or create "ghost" duplicates.
- **Outlier filter:** for each detection lifted into 3D, check how consistently it reprojects into the other views where it should be visible. Large reprojection or epipolar residuals across views mark it as possibly moving. Flag it, downweight it in association, and show it as "possibly moved" in the UI.
- **Benchmark:** add 3–5 own captures with deliberate motion (a person walks through; one object is moved) and measure ghost-duplicate rate with and without the filter.
- **VGGT-Ω comparison:** VGGT-Ω reports support for dynamic scenes; compare it against static VGGT on the motion captures.

### 8.4 Lift to 3D (`countonce/lift.py`)
1. Region: mask, else central 50% of the box, else a small disk around the point
2. Read `world_points` in the region; drop pixels below a confidence percentile, and drop pixels inside the frame's privacy mask (blurred faces/plates, §14.8)
3. `xyz` = per-axis median of remaining points
4. `size` ≈ 2D extent × median depth ÷ focal length
5. `lift_conf` from detector score, point confidence and depth spread
6. Fail (`xyz=None`) if too few confident pixels; such detections can still be matched in v2/v3 or are "rejected"

**Scene scale:** `scene_scale` = median camera-to-scene distance. All thresholds are relative to object `size` or `scene_scale`.

### 8.5 Associate v1: constrained 3D clustering (`countonce/merge/cluster.py`)
- Distance: Euclidean between `xyz`, normalized by the pair's mean `size`
- Merge if normalized distance < `k` (default 0.5; swept)
- **Cannot-link:** detections from the same frame never merge (unit-tested)
- Greedy agglomerative clustering with cluster-level cannot-link checks
- Output instances with members, view count, confidence

### 8.6 Associate v2: geometric cross-view matching (`countonce/merge/match.py`)
1. **Visibility:** project each instance into every frame; inside the image + **depth test** (projected depth ≈ frame depth there) means it **should be visible**
2. **Appearance:** DINOv2 features per detection crop; cosine similarity
3. **Assignment:** Hungarian matching between instance tracks and each frame's detections, cost = reprojection error (normalized by box size) + appearance distance; reject above threshold
4. **Hallucination filter:** instances seen in one view but expected in ≥2 others with no match get downweighted or dropped (configurable; ablated)
5. **Split check:** split clusters with low internal appearance similarity or an extent much larger than one object

### 8.7 Associate v3: learned matcher (Track C, Strong tier)
See §10. Same interface as v1/v2: `Associator.associate(lifted, recon, frames) -> list[Instance]`.

### 8.8 Confidence range (`countonce/confidence.py`)
**Core (sensitivity range):**
- Grid over `k` and detector thresholds (v1/v2); for v3, match-probability thresholds
- `count` = default setting; `count_low`/`count_high` = 10th/90th percentile
- Labeled honestly as a **sensitivity range**, not a statistical guarantee

**Strong (split conformal prediction, replaces the above as the default):**
- Hold out a **calibration set** (synthetic scenes + calibration-only Fuji-SfM trees + some own captures), never used for training or testing.
- Compute a nonconformity score per calibration scene, for example the normalized error `|true − predicted| / σ̂`, where `σ̂` is a per-scene uncertainty estimate (the sensitivity-range width or the variance of v3 match probabilities). This gives intervals that widen on hard scenes and narrow on easy ones.
- The interval for a new scene is `predicted ± q̂ · σ̂`, where `q̂` is the conformal quantile at the target level (e.g., 90%).
- **Guarantee and caveat:** coverage holds on average over scenes that are exchangeable with the calibration set (marginal, not per-scene). Report empirical coverage on the test split, broken down by scene tier and domain (synthetic vs. real). With only tens of real scenes, intervals will be wide; say so.
- **Minimum calibration size:** split conformal uses the ⌈(n+1)(1−α)⌉-th smallest calibration score. For that to exist, (n+1)(1−α) ≤ n, so a **90% interval needs n ≥ 9 calibration scenes** (95% needs n ≥ 19). A handful of Fuji-SfM trees is not enough on its own, and calibrating real scenes only on synthetic data breaks exchangeability.
- **Real-data calibration rule:** require **≥ 15 real calibration scenes** (own captures + calibration-only Fuji-SfM trees, held out by scene/tree) before reporting a split-conformal interval on real data. If fewer are available, use **Jackknife+ (or CV+ / cross-conformal)** on the small real set: refit whatever is tuned on real data (temperature scaling, the merge radius `k`, the σ̂ scale) leaving one scene out at a time, and build the interval from the leave-one-out residuals. Jackknife+'s worst-case guarantee is 1−2α (≥ 80% for a nominal 90% interval), and it is typically close to 1−α in practice. State which method produced each interval, and its `n`, in `result.json` and the README.
- Low-confidence flag when the interval is wide relative to the count (threshold tuned on calibration data).

### 8.9 Quality warnings (`countonce/warnings.py`)
Too few usable frames; low overlap; reflective/low-texture scene (low depth confidence); many failed lifts; wide confidence range. Each has a plain-English message and a fix ("Take 3–5 more photos, moving sideways a little between each").

### 8.10 Export (`countonce/export/`)
`result.json`, `overlays/` (new = green, duplicate = blue, rejected = gray, with IDs), `scene.ply` (downsampled), `markers.json` (viewer), `counts.csv`.

**Schema versioning (Strong):** `result.json` and `markers.json` carry `schema_version` (starting at `"1.0"`) and are defined by strict Pydantic models in `countonce/schemas/`. Readers support every older version (with upgrade functions), so changing the pipeline from v1 → v2 → v3 never breaks old shared links or historical jobs. Schema changes require a version bump and a test that loads a stored example of every older version.

### 8.11 Entry point and CLI
```python
def run(inputs: list[Path] | Path, prompt: str, config: Config) -> CountResult: ...
```
```
countonce run --images dir/ --prompt "bikes" [--detector countgd|gdino|countgdpp] [--backbone vggt|vggt_omega|vggt_t3|colmap] [--assoc v1|v2|v3] [--out results/]
countonce run --video clip.mp4 --prompt "apples" --fps 2
countonce bench --config configs/bench.yaml
countonce synth --config configs/synth.yaml        # Track C
countonce train --config configs/match.yaml        # Track C
countonce profile --config configs/profile.yaml    # Track F
countonce env
```

---

## 9. Track B: Benchmark and evaluation (Core → Strong)

### 9.1 Datasets
| Dataset | What it gives | Use | Tier |
|---|---|---|---|
| **Own captures** (10–20 scenes) | Real scenes with hand counts; identity labels on 3–5 scenes | Main real-world test | Core |
| **Fuji-SfM** | 582-image motion sequence of 11 Fuji apple trees with **1,455 apples and their ground-truth 3D positions**; 288 images with instance masks | Exact duplicate/miss measurement on a real orchard; also fine-tuning data for the learned matcher | Core |
| **RGBD-VideoCount** | 195 RGB-D clips, 6 categories, 2,032 annotated frames, 77,638 boxes, with association and de-dup annotations | Compare with DG-Net (which uses sensor depth) using **RGB only**; shelf/crowded scenes | Strong |
| **VideoCount** | 370 videos (TAO-Count, MOT20-Count, Science-Count) with unique-instance counts | Compare with CountVid on its own benchmark (subset that fits the setting) | Strong |
| **WGISD** | 300 images, 4,432 grape clusters, 5 varieties | Second crop; detector ablation | Moon |
| **MinneApple** | ~1,000 images, ~41,000 annotated apples; 13,000 counting patches | Detector fine-tuning ablation | Moon |
| **AppleGrowthVision** | 21,407 orchard images, multi-view; boxes on ~6.8%; manual counts on 10 trees | Extra robustness check | Moon |
| **FSC-147** | 6,135 images, 147 categories | Sanity-check detector wrappers only (single image) | Core |
| **OneScene-Bench** | Real multi-view scenes with unique-object counting questions, controlled view overlap | Compare CountOnce with the 12 VLMs it reports, on the same questions | Strong |

Claude Code must check availability, license and format for each in Phase 0 and record them in `docs/DATASETS.md`.

**Fuji-SfM ground-truth matching:** project each GT 3D apple into every image using the dataset's cameras (or align VGGT's reconstruction to the GT point cloud with Sim(3)), then assign each detection to the nearest GT apple within a threshold. That yields per-detection identity labels, so duplicate rate and miss rate can be measured exactly.

### 9.2 Own capture protocol
- **What is being counted (definition, fixed before capturing):** the ground truth is the number of target objects **visible in at least one submitted photo**, not every object physically present (fully hidden objects are out of scope, §3.3). Each `gt.json` records this definition, and the hand count is done from the photos, not from the physical scene.
- **Size:** **20–30 independently captured static scenes** (up from 10–20), including clutter, occlusion, near-identical objects and **at least 3 zero-target scenes**. Train/validation/test splits are **by physical scene** (never two captures of the same scene across splits).
- **Tiers:** easy (bike racks, chairs, shelf items, small car lot), medium (nursery plants, bins/cones), hard (fruit on trees or vines, dense similar objects)
- **Per scene:** a 30–60 s video along an arc, plus 20 stills along the same arc; note lighting and problems
- **Ground truth:** count by hand twice (two people if possible), and a third time if they disagree; save `gt.json`
- **Identity labels (3–5 scenes):** click each object in 5 photos with a small labeling tool (`tools/label_ids.py`, a simple local web page) to get cross-view identities

### 9.3 Baselines
1. **Sum** of per-image counts
2. **Max** single-image count
3. **Median** per image (sanity)
4. **COLMAP lifting** (same lift + association, COLMAP geometry)
5. **C-DOG** geometric association (if code runs), on the same detections + cameras
6. **CountVid** on video versions of scenes and on VideoCount
7. **DG-Net** on RGBD-VideoCount (it uses sensor depth; CountOnce uses RGB only, so state the difference clearly)
8. **Frontier VLMs (Strong tier):** send the same photos (≤ the provider's image limit) to 2–3 frontier multimodal models (for example Claude, Gemini and GPT) plus one open-weight model (for example Qwen-VL), with a fixed, documented prompt asking for the number of unique objects. Also try a "locate and label each object first" prompt, since published work shows explicit enumeration prompts improve VLM counting. Run on a subset (e.g., 30 scenes × 3 view counts), record model versions, dates, cost and refusals. **$0 budget:** use free API tiers where they exist (within their rate limits), run the open-weight model locally on the laptop, and skip any paid model rather than paying. Answers the interview question "why not just ask a VLM?" with numbers.
9. **Cycle-consistent synchronization baseline:** spectral permutation synchronization / CLEAR applied to v2's pairwise matches
10. **Human baseline (Strong tier):** 3–5 people count from the same photo sets (no 3D view). Record time taken and accuracy. This measures the real-world value: "CountOnce matches careful humans at X× less time" (or doesn't).

If a baseline can't run, document why in `docs/DEVIATIONS.md`.

### 9.4 Metrics
**Headline metrics are count metrics, not association F1.** F1 explains *why* a count is wrong; it is never reported as the main result, and 0.96 F1 is never described as "96% accurate".
- **Count MAE**, **signed count bias** (over- vs under-counting), **MAPE**, **within-10%** rate, and **false count on zero-target scenes**
- **Count error decomposition** per scene: missed objects + duplicate instances (one object counted twice) + false positives (not a target) − wrong merges (two objects counted once) = predicted − true. Reported as a stacked bar per method; it says whether the next fix belongs in detection, geometry or association
- **MAE**, **MAPE**, **within-10%** rate (by view count and tier)
- **Duplicate rate** and **miss rate** (scenes with identity labels; Fuji-SfM)
- **Association metrics** on labeled data: pairwise precision/recall/F1 of "same object" decisions; cluster purity
- **Coverage:** truth inside `[count_low, count_high]`
- **Runtime** per stage; **peak VRAM**

### 9.5 Sweeps
- Views: 3, 5, 10, 20, 40 (VRAM-capped), 3 random samples + 1 evenly spaced per setting
- Viewpoint spread: narrow arc vs. wide arc (where geometry should beat tracking)
- Ordered video vs. **shuffled unordered photos** (where tracking-based methods should struggle)
- Methods: baselines + CountOnce v1/v2/v3 × backbones
- Results to `results/bench.csv`, one row per run

### 9.6 Ablations
**Controlled failure-source experiments (Core, run on Fuji-SfM and own captures with identity labels):**
1. **Oracle detections + estimated geometry** (GT boxes projected into each photo, VGGT geometry): isolates reconstruction and association errors
2. **Real detections + oracle geometry** (dataset cameras / GT-aligned points): isolates detector errors
3. **Full real pipeline**
4. Every association method (v1, v2, sync, v3) on **identical upstream inputs**, each with settings **tuned on the validation split** (not defaults) before it's compared on the test split

**True order invariance:** permute the actual photo-to-frame assignment (reassign frame ids and reorder the camera arrays to match), not just the input list, and measure how much counts and partitions change across ≥10 permutations. If v2 depends on order, either add a global refinement step, use sync/v3 as the default for unordered photos, or narrow the "unordered photos" claim.

Merge radius `k`; cannot-link on/off; v1 vs v2 vs v3; v2 ± appearance; ± hallucination filter; box-center vs mask lifting; detector choice; backbone choice; synthetic-only vs synthetic + Fuji-SfM fine-tune for v3.

### 9.7 Figures
1. **Headline:** MAPE vs. number of views, one line per method
2. Ordered vs. unordered input, by method
3. Error by scene tier
4. Association F1, v1 vs v2 vs v3
5. Sim-to-real: v3 trained on synthetic only vs. + real fine-tune
6. Runtime vs. views (log scale); speedups from Track F
7. Ablation chart
8. Confidence range coverage
9. **Failure gallery:** merges, splits, hallucinations, each with overlay + cause
10. VLM vs. CountOnce error by view overlap (Strong)
11. Conformal coverage vs. target level, real vs. synthetic (Strong)

### 9.8 Statistical rigor (Core)
- **Frozen test split:** choose test scenes/trees once in Phase 2, record their hashes in `bench/splits.json`, and never tune on them. All tuning uses a separate validation split.
- **Uncertainty on every number:** report metrics with **95% bootstrap confidence intervals** (resampling scenes).
- **Paired comparisons:** when claiming method A beats B, use a paired bootstrap (or Wilcoxon signed-rank) over the same scenes and report the p-value or the CI of the difference.
- **Seeds:** learned components report mean ± std over ≥3 training seeds.
- **No cherry-picking:** the failure gallery is sampled systematically (e.g., the worst 10% by error), not hand-picked.

### 9.8b Zero-count and distractor robustness (Strong)
Real counters fail hardest when the target isn't there at all ("oranges" on an apple tree, "bikes" in an empty hallway).
- **Test suite:** 10–20 scenes (own captures + synthetic) where the true count is 0, or only look-alike distractors are present. Also wrong-prompt tests on existing scenes (ask for an object that isn't in them).
- **Metrics:** false positive rate (share of zero-count scenes with a predicted count > 0), mean false count, and whether the confidence interval includes 0.
- **Negative prompts:** with CountGD++, test whether negative prompts ("not leaves", "not green apples") suppress false positives without losing true objects.
- **Product behavior:** when the prediction is 0 or near 0 with high uncertainty, the UI says "No [objects] found" plus what it did detect instead, rather than showing a confident small number.

### 9.9 Error slicing (Strong)
- Log per-scene metadata: object count, density (objects per image area), mean occlusion (from identity labels or synthetic GT), camera baseline spread, number of views, mean depth confidence, luminance, blur score, detector score distribution.
- `bench/slices.py` computes error and association F1 per slice (e.g., count > 50 vs. < 20; depth confidence < 0.6) and flags slices that are significantly worse than average.
- Results feed the quality warnings (§8.9) so warnings are backed by data, not guesses.

### 9.10 Real-world value validation (Moon)
- With a farm or research partner, compare CountOnce counts with **harvest or hand-tallied yield** for a few trees or vines. This is the gold-standard check that the numbers matter.
- Report time saved vs. the partner's current method (from the Phase 0 interviews).

---

## 10. Track C: Learned matching (CountOnce-Match) on synthetic data (Strong tier)

### 10.1 Why
Hand-tuned rules (v1/v2) break on dense, similar-looking objects like fruit clusters. A learned model can combine 3D position, size, appearance and visibility far better than fixed thresholds. Training data with perfect identity labels can be **generated**, so data isn't a bottleneck.

### 10.2 Synthetic data generation (`synth/`)
Built on **Kubric** (PyBullet physics + Blender rendering, Docker), which outputs instance masks, depth and camera poses.

**Scene recipe (randomized per scene):**
- **Objects:** 5–150 instances drawn from asset sets: fruit-like shapes (spheres/ellipsoids with fruit textures), ShapeNet/Google Scanned Objects assets (check licenses), simple clutter. Include **many visually identical instances** (the hard case).
- **Arrangements:** scattered on a ground plane; piled with physics; attached to "canopy" structures (branches/leaf planes) for orchard-like occlusion; on shelves in rows.
- **Occluders:** leaf-like planes, branches, boxes.
- **Lighting:** random HDRI environment maps, sun direction, intensity.
- **Cameras:** 3–40 views along arcs, circles, random walks; vary distance, height, field of view; include large viewpoint changes and sparse sets.
- **Distractors:** objects similar to but not matching the target prompt (for negative examples).

**Outputs per scene:** RGB images, instance masks, depth, camera intrinsics/extrinsics, object IDs and 3D positions, a text prompt for the target category.

**Scale target:** start with 1,000 scenes (smoke test), then 10,000+ scenes. Render resolution matches the backbone input. Track storage and render time; record them in `docs/SYNTH.md`.

**Important:** run the **same Detect + Reconstruct + Lift pipeline** on synthetic images (not ground-truth geometry). The matcher then learns to handle real detector errors and real reconstruction noise. Ground truth is used only for labels.

**Domain randomization knobs** (logged per scene): texture variety, blur, noise, exposure, JPEG compression, partial detections (randomly drop), false positives (inject).

**Renderer fallback:** test Kubric GPU rendering on the RTX 5070 in the first session of Phase 3. Kubric pins Blender 2.93, which may not support the newest GPUs. If GPU rendering fails, either render on CPU (slower but fine for 1,000-scene smoke sets) or switch to **BlenderProc**, which offers RGB, depth, instance segmentation, camera poses and physics through a Python API. Keep the scene recipe renderer-agnostic (`synth/recipe.py` produces a scene description; `synth/render_kubric.py` and `synth/render_blenderproc.py` render it).

**Data-quality gates (`synth/validate.py`)**, run right after rendering and before any GPU model runs:
- Schema check with Pydantic/Pandera on every scene's metadata
- Quarantine scenes with NaN/inf depth, all-black or overexposed frames, zero visible target pixels, fewer than 2 views seeing any target, or degenerate camera rigs (all cameras nearly identical)
- Write a quarantine report with counts by reason; target < 2% quarantined

**Storage format:** see Track J (§16B.1). Raw renders go into sharded archives; the matcher trains on small precomputed per-detection features, not raw images.

### 10.3 Model architecture (`countonce/match/model.py`)
A multi-view attentional graph neural network, extending the pairwise ROM/SuperGlue idea to many views:
- **Node per detection.** Input features:
  - Lifted 3D position (normalized by `scene_scale`), size, lift confidence
  - 2D box geometry (normalized), detector score
  - Appearance embedding (DINOv2 crop feature, projected to model width)
  - Camera embedding (view index position encoding; ray direction to the detection)
  - Optional: VGGT token features pooled over the detection region (ablation)
- **Edges:** within-view (self) and cross-view (all detections in other views, or k-nearest in 3D for scale)
- **Layers:** L alternating self-attention (within view) and cross-attention (across views) blocks, with 3D relative-position bias
- **Output:** pairwise match scores between detections in different views; a **dustbin** per view for "no match" (occluded, out of view, false positive); **Sinkhorn** normalization for soft one-to-one assignment between view pairs
- **Instance formation:** threshold match probabilities → graph → connected components with cannot-link enforcement (split components containing two detections from the same frame using a min-cut on match scores)
- **Size:** small (e.g., 4–8 layers, width 128–256), so it trains on a laptop GPU

**Cycle consistency (Strong tier).** Pairwise Sinkhorn matches aren't guaranteed to agree across three or more views: A↔B and B↔C don't force A↔C. In dense scenes like orchards and shelves, that creates merge and split errors. Two fixes, both implemented and ablated:
1. **Training-time cycle-consistency loss:** for sampled view triplets (i, j, k), penalize the difference between the composed soft assignment P_ij · P_jk and P_ik (excluding dustbin mass).
2. **Inference-time permutation synchronization:** build the multi-view affinity matrix from all pairwise match probabilities and run a spectral synchronization step (Pachauri et al. 2013; normalized spectral sync), or the CLEAR algorithm, which also enforces distinctness (same-view detections never match). Then form instances from the synchronized matches.

Report association F1 and count error for: pairwise only, + cycle loss, + synchronization, + both.

### 10.4 Training (`countonce/match/train.py`)
- **Loss:** negative log-likelihood of the true partial assignment (including dustbin) per view pair, plus a count-consistency auxiliary loss (predicted instance count vs. true)
- **Sampling:** view pairs and view subsets per scene, varying the number of views
- **Optimizer:** AdamW, cosine schedule, mixed precision (bf16), gradient clipping
- **Validation:** held-out synthetic scenes + Fuji-SfM validation trees
- **Fine-tuning (ablation):** a few Fuji-SfM trees with identity labels from §9.1; strict tree-level train/val/test split so no tree is used for both
- **Checkpoints** saved by validation association F1

### 10.5 Evaluation
- Association F1, duplicate/miss rates and count MAPE on: synthetic test, Fuji-SfM test trees, own captures with identity labels
- Compare v1 vs v2 vs v3, and v3 synthetic-only vs + fine-tune
- **Sim-to-real gap** analysis: which domain randomization knobs mattered (ablate 2–3 of them)
- **Feature-group attribution (Strong):** retrain the matcher with input groups removed or isolated: 3D position only; appearance (DINOv2) only; 3D + camera rays; appearance + 3D; full features; full + VGGT tokens. Report association F1 for each with CIs. This shows exactly how much the 3D geometry adds over 2D appearance, which is the project's central claim and a key paper figure.
- **Probability calibration (Strong):** measure Expected Calibration Error (ECE) and draw reliability diagrams for match probabilities and per-instance confidence on the validation split. Fit temperature scaling (one parameter, on validation data only) so that a 0.85 match score really means about an 85% chance of being the same object. Report ECE before and after on the test split. Calibrated probabilities also make the conformal intervals (§8.8) and the "Explain this match" inspector more meaningful.

### 10.6 Experiment tracking
Log every run (config, commit, data version, metrics, curves) with a local tracker (Trackio, or MLflow/W&B if preferred). Store final metrics as CSV in `results/` so figures don't depend on the tracker.

---

## 11. Track D: Detector adaptation (Moon tier)

- **Question:** when does fine-tuning the open-vocabulary counter on one crop beat zero-shot?
- **Data:** MinneApple (apples), WGISD (grapes); hold out Fuji-SfM images as an unseen-orchard test
- **Method:** parameter-efficient fine-tuning of CountGD (LoRA on the text/vision fusion layers, or the head only), using the repo's own training code where possible
- **Report:** per-image detection quality and end-to-end deduplicated count error, zero-shot vs fine-tuned, on seen and unseen orchards
- **Later:** swap in CountGD++ (from the paper project) and compare, including negative prompts ("not leaves") and pseudo-exemplars

---

## 12. Track E: Reconstruction backbones (Moon tier)

- Implement `vggt_omega_recon.py` and `vggt_t3_recon.py` behind the same `Reconstructor` interface
- **VGGT-Ω:** reports ~30% of VGGT's memory and ~1.6× faster inference. Benchmark on the laptop: VRAM vs frames, runtime, and end-to-end count accuracy. Gated weights and a non-commercial license (§24).
- **VGG-T³:** linear scaling with the number of images (for long videos or large orchard rows). Use it if code and weights are available.
- Report the best backbone for each setting (few views vs many views)

---

## 13. Track F: Inference optimization (Strong tier, ML infra showcase)

**Goal:** measurably faster end-to-end jobs on the RTX 5070, with a written profiling report.

1. **Profile first:** per-stage time and peak VRAM (`torch.profiler`, Nsight Systems optional). Write `docs/PROFILE.md` with the baseline breakdown.
2. **Attention:** make sure models use PyTorch SDPA (FlashAttention / memory-efficient kernels) where possible.
3. **Precision:** bf16 everywhere it's safe; check accuracy impact on the benchmark.
4. **Compilation:** `torch.compile` on the detector and reconstructor (fixed input shapes where possible); try Torch-TensorRT (`torch.compile(backend="torch_tensorrt")` or AoT) for the detector if it compiles cleanly. Record what worked and what didn't.
5. **Batching:** batch per-image detection; overlap CPU preprocessing with GPU work (data-loader threads, pinned memory).
6. **Caching:** reuse features across confidence-range sweeps; skip recomputation on re-runs.
6b. **Content-addressable stage caching (Strong):** hash the uploaded image set (SHA-256 over sorted per-file hashes + preprocessing settings + backbone version) and store the `Recon` output in R2/disk with a TTL. When a user re-runs the same photos with a new prompt ("red apples" → "green apples"), skip reconstruction and only re-run detection + association. Cache detections the same way per prompt. Measure the repeat-job latency and GPU-time savings and report them (don't assume a number).
6c. **Memory hygiene + OOM fallback ladder (Strong):** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported (reduces fragmentation-related OOMs; not on WSL, where it broke the five-view run, see DEVIATIONS). Wrap each GPU stage in a degradation ladder: on CUDA OOM → free cached memory → retry with smaller chunks → retry at a lower resolution tier → fail with a clear `ERR_GPU_OOM` and a user tip. Log which rung succeeded so the memory calibration improves over time.
7. **Report:** a table of each optimization's speedup, VRAM change and accuracy change, plus end-to-end p50/p90 job time before and after.

**Moon-tier additions:**
8. **One custom Triton kernel.** Candidates: (a) fused pairwise 3D distance + size normalization + cannot-link masking + Sinkhorn iterations for the associator; (b) fused confidence-filtered median lifting over detection regions. Pick whichever profiling shows is a real bottleneck. Benchmark against eager PyTorch and `torch.compile` across input sizes, include an Nsight Systems trace, and report honestly if it doesn't beat `torch.compile`. The learning and the measurement are the point.
9. **Cold-start engineering for cloud workers:** model weights (reconstructor + detector + DINOv2) total several GB. Bake weights into the container image or mount a network volume; load with safetensors memory-mapping where supported; keep one warm worker during pilot hours. Measure cold-start time, warm p50/p95 latency and cost per 1,000 frames, and record them in `docs/SERVING.md`.
10. **Low-precision inference (Moon):** Blackwell GPUs (including GeForce RTX 50-series, `sm_120`) support FP8 and NVFP4 in hardware, but **software support on `sm_120` is still uneven** (several libraries restrict FP4 kernels to datacenter Blackwell or need patches). Plan: try INT8 / FP8 weight quantization of the detector and reconstructor with torchao or TensorRT, then NVFP4 if a working path exists for `sm_120`. Measure VRAM, max frames per pass, latency and the count-error change (MAPE delta with CI). Document exactly which library/kernel path served each result, and report failures honestly.
11. **Activation checkpointing (Moon, training only):** for training CountOnce-Match or fine-tuning the detector (Track D), use selective activation checkpointing to fit larger batches or more views in 8–12 GB. (It doesn't help inference, which stores no activations for backprop.)

---

## 14. Track G: Web product (Strong tier)

### 14.1 User flow
1. Sign in with Google
2. **New count:** drag in photos or a video, type what to count, optionally draw a box on one example object, submit
3. **Progress:** live stages (queued → frames → detecting → reconstructing → associating → done) streamed over Server-Sent Events, with elapsed time
4. **Results:** count card ("47 apples, range 44–51") + warnings; 3D viewer with markers (click → highlight in photos, and the reverse); photo strip with overlays; feedback ("looks right" / "off by about ___"); downloads; read-only share link
5. **History** of past counts

### 14.2 API (FastAPI)
| Method | Path | Purpose |
|---|---|---|
| POST | `/jobs` | Create job → id + signed upload URLs |
| POST | `/jobs/{id}/submit` | Queue the job |
| GET | `/jobs/{id}` | Status, stage, progress, timings |
| GET | `/jobs/{id}/events` | SSE stream of progress |
| GET | `/jobs/{id}/result` | Result + output URLs |
| POST | `/jobs/{id}/feedback` | User feedback |
| DELETE | `/jobs/{id}` | Delete job and files |
| GET | `/jobs` | History |
| GET | `/share/{token}` | Read-only shared result |
| GET | `/health` | Health + worker online status |

### 14.3 Database (Postgres)
Migrations with Alembic. Postgres holds live app data and the job queue only; analytics go to Parquet + DuckDB (§16B.1).

| Table | Key columns | Tier |
|---|---|---|
| `users` | id, email, name, created_at, `training_opt_in` (bool), deleted_at | Strong |
| `jobs` | id, user_id, prompt, negative_prompts, options_json, **status** (`queued` \| `running` \| `done` \| `failed` \| `dead_letter` \| `deleted`), stage, progress, **error_code**, error_message, attempts, worker_id, **idempotency_key** (unique per user), **input_hash**, **model_routing_id**, **site_id** (nullable), captured_at, created_at, started_at, finished_at | Strong |
| `job_files` | id, job_id, kind (`input` \| `output`), key, bytes, sha256, **exif_stripped** (bool), **privacy_masked** (bool) | Strong |
| `results` | job_id, **schema_version**, count, interval_low, interval_high, interval_level, **interval_method**, n_calibration, zero_count, result_json, warnings_json, model_versions_json, recon_cache_hit, oom_fallback_rung, peak_vram_mb | Strong |
| `feedback` | id, job_id, user_id, correct (bool), user_count, comment, created_at | Strong |
| `corrections` | id, job_id, user_id, action (`split` \| `merge` \| `reject`), detection_ids, instance_ids, created_at, exported_version | Moon (UI in Strong) |
| `share_tokens` | token, job_id, created_at, revoked_at | Strong |
| `workers` | id, name, kind (`laptop` \| `cloud`), last_heartbeat, gpu_name, vram_mb, cpu_name, cpu_count, version | Strong |
| `recon_cache` | input_hash, backbone_version, storage_key, created_at, expires_at, hits | Strong |
| `model_routing` | id, detector, backbone, associator, calibration, is_default, shadow_percent | Moon |
| `shadow_results` | job_id, model_routing_id, count, interval_low, interval_high, delta_vs_live, timings_json | Moon |
| `sites` | id, user_id, name, description, created_at (jobs link via `jobs.site_id`; each job is one visit) | Moon |
| `api_keys` | id, user_id, hashed_key, created_at, last_used_at, revoked_at | Moon |

### 14.4 Queue and workers
- Jobs table as the queue; claim with `SELECT … FOR UPDATE SKIP LOCKED`
- Worker loop: claim → download → `pipeline.run` → upload → write result → done; stage/progress updates; heartbeat every 15 s
- Missed heartbeats for 2 min → requeue (max 2 attempts, then failed with the error)
- **Typed error taxonomy + dead-letter queue (Strong):** every failure gets a code. **Retryable** infra errors (`ERR_WORKER_TIMEOUT`, `ERR_STORAGE_TRANSIENT`, `ERR_GPU_OOM` after the fallback ladder) are retried with backoff. **Permanent** user/media errors (`ERR_CORRUPT_MEDIA`, `ERR_UNSUPPORTED_CODEC`, `ERR_TOO_FEW_FRAMES`, `ERR_NO_OVERLAP`) fail fast, with no GPU retries, into a `dead_letter` status with an actionable user message ("Your video uses a codec we can't read. Export it as MP4/H.264"). The admin page lists dead-lettered jobs by code.
- **Laptop worker first**; "processing paused" banner when no worker is online
- **Cloud worker later:** the same Docker image on a pay-per-second GPU provider, started when the queue isn't empty and stopped when idle (scale to zero). Track cost per job.

### 14.5 Storage, security, privacy
- Cloudflare R2 (or S3), signed URLs only
- Limits: ≤60 images or one video ≤60 s / 200 MB; jpg/png/heic/mp4/mov
- Google sign-in; private by default; revocable share tokens; per-user rate limits
- Uploads deleted after 30 days (configurable); immediate delete available; never used for training without opt-in
- Short privacy note on the upload page

### 14.6 Observability
Structured logs; per-job stage timings, VRAM and errors in the DB; admin page with jobs/day, success rate, p50/p90 time, feedback accuracy, cost per job.

### 14.7 Deployment ($0 budget)
**Constraint: the project must run at $0.** Every service below has a free tier. Verify current limits before deploying, since free tiers change.

| Part | Free option | Limits to design around |
|---|---|---|
| Frontend | **Vercel Hobby** (or Cloudflare Pages) | Personal/non-commercial use; fine for a free pilot |
| API | **Render free web service** (no card) | Sleeps after 15 min without traffic and takes ~1 min to wake; filesystem is wiped on restart, so nothing is stored on the API server |
| Database | **Neon Free** (no card, no expiry) | 1 GB storage and 100 CU-hours per project per month; scales to zero when idle. **Don't use Render's free Postgres**: it expires after 30 days |
| File storage | **Cloudflare R2 free tier** (10 GB/month, free egress) | May need a card or PayPal on file to activate, even though usage under the free tier isn't charged. Without one: Neon Object Storage (free during beta) or Azure Blob Storage under Azure for Students |
| GPU worker | **The laptop** | Jobs only run while the laptop is on; the "processing paused" banner covers this |
| CI | **GitHub Actions** (free for public repos) + the self-hosted laptop runner | — |
| Auth | Google sign-in | Free |
| Big-name cloud (optional) | **Azure for Students:** $100 credit for 12 months, **no credit card**, renewable yearly while a student | Use it to host the API on Azure Container Apps and files on Azure Blob if "Azure" on the resume is wanted. When credits run out, the subscription is disabled (no surprise bill) |

**Designing around the free limits:**
- **API cold starts:** the frontend shows a "waking up the server" state if the API takes more than a few seconds to answer. The worker polls Neon directly (not through the API), so jobs still run while the API sleeps.
- **Storage cap:** the 30-day upload deletion (§14.5) plus a per-user storage limit keeps usage under 10 GB. Store downsampled point clouds and compressed overlays, not raw high-resolution renders.
- **Database cap:** big results (`result.json`, match evidence) live in object storage; Postgres stores only metadata and a pointer.
- **No paid cloud GPU:** the scale-to-zero cloud GPU worker (§14.4, §13 item 9) becomes optional, and is only done if free credits (Azure for Students, AWS Student Rewards credits) cover a short experiment.
- **Usage alerts:** check each provider's usage page weekly during the pilot; set spending limits to $0 wherever a provider allows it.

Config in repo, `.env.example` for every variable, **no secrets in git**. GitHub Actions deploys API and frontend on merge to main.

### 14.8 Product upgrades (Strong unless marked)
- **Resumable uploads:** S3-compatible multipart uploads to R2 (or the TUS protocol) so flaky phone connections on farms resume instead of restarting.
- **Idempotency keys** on `POST /jobs` and `POST /jobs/{id}/submit` (client-generated UUID header; the server stores it and returns the original response on retry), so retries never double-queue GPU jobs.
- **Client-side pre-flight check:** in the browser (Canvas, or WebAssembly if the plain-JS version is too slow), compute a blur score per photo and a frame-diversity score (small downsampled image differences) before upload. Show instant tips like "Photo 4 is blurry, retake it" or "These 3 photos are nearly identical."
- **"Explain this match" inspector:** click any two detections (in photos or as 3D markers) to see their 3D distance, reprojection error, appearance similarity and the matcher's match probability, plus why they were or weren't merged.
- **Correction tools:** on any marker, users can **Split** (two objects wrongly merged), **Merge** (a duplicate that was missed) or **Reject** (not the target object). Corrections are stored as structured training signal (Track L).
- **Automatic face and license-plate blurring** before storing any upload (privacy for photos taken in public places). Blurring changes pixels, which can produce bad depth there, so the blur step outputs a per-image **privacy mask**. During lifting (§8.4), pixels inside the mask are treated as zero-confidence: they're excluded from VGGT depth/point confidence and from each detection's median, and detections mostly inside a masked region are marked `rejected` with reason `privacy_mask`. The mask is also used to skip those regions when sampling points for the 3D viewer.
- **Media sanitization and EXIF privacy:** on ingest, apply EXIF orientation to the pixels, then **strip all EXIF/GPS metadata** before saving (farm and home photos contain exact GPS locations). Validate videos with `ffprobe` before decoding (codec allow-list, max resolution, max fps, max duration, max frame count) to block malformed files and decompression bombs; check image dimensions before full decode. Run ffmpeg with strict timeouts and memory limits.
- **1-click inspection report:** a PDF/HTML report with the final count, the interval, key annotated photos, a 3D snapshot, warnings, settings and a timestamp, ready to email to a farm manager or client.
- **Sites and visits (Moon):** users group jobs into a Site/Plot (e.g., "Row 4 apples") and see counts over time ("Week 1 vs. Week 3"), with a simple chart and per-visit reports.
- **Gaussian splat view toggle (Moon):** for selected scenes, an optional high-quality splat view (reusing the gsplat pipeline from `vggt-sparse-recon`) next to the fast point cloud.

---

## 15. Track H: Agent and developer integrations (Moon tier)

- **Python package:** `pip install countonce` exposing `countonce.run(...)`
- **REST API for developers:** API keys, the same job endpoints, rate limits, docs page (OpenAPI)
- **MCP server:** a `count_objects(images, prompt)` tool so LLM agents (Claude, etc.) can call CountOnce and get counts + a results link. Include an example where an agent answers "how many ripe apples are on these trees?" by calling the tool.
- **Natural-language prompts:** pass richer prompts through to the detector ("red apples, not leaves"); with CountGD++, map "not X" phrases to negative prompts.

---

## 16. Track I: Open source and releases (Moon tier)

- **GitHub:** clean repo, license, CONTRIBUTING, issue templates, CI badge
- **Hugging Face Space:** a free demo (photos + prompt → count + overlays), using ZeroGPU or the cheapest GPU tier that works; check current Space GPU options
- **Hugging Face model:** CountOnce-Match weights with a model card (training data, metrics, limitations, license)
- **Hugging Face dataset:** **CountOnce-Synth** (a subset of the synthetic scenes) and, if licenses allow, the own-capture benchmark with ground truth; dataset card included
- **Track outside usage:** downloads, stars, Space visits, issues from strangers
- **CountOnce-Bench (public benchmark):** release the frozen test split (where licenses allow), an evaluation script that scores any method's predictions, and a small leaderboard table in the README (CountOnce variants, baselines, VLMs). Other researchers submitting results is the strongest outside signal.
- **Model card + datasheet:** document intended use, training data, metrics with CIs, known failure slices, licenses and limitations (following the "model cards" and "datasheets for datasets" practices).

---

## 16B. Additional tracks (v4)

### 16B.1 Track J: Data engineering (Strong tier)
**Goal:** the synthetic data, real data, features and results are versioned, queryable and fast to stream.

- **Layout:** raw renders packed into **WebDataset `.tar` shards** (one scene per sample: images, depth, masks, cameras, metadata JSON), ~1 GB per shard, stored locally and mirrored to R2.
- **Feature store:** precomputed per-detection features (lifted xyz, size, box, score, DINOv2 embedding, view id, GT instance id) stored as **Parquet** (or **Lance** if random access/vector search is useful), one row per detection. This is what the matcher trains on. It's small, so training reads it quickly.
- **Catalog:** a **DuckDB** catalog over the Parquet metadata (scene id, recipe params, object count, occlusion, baseline spread, renderer, data version, quarantine status). Queries like "orchard-style scenes with > 80 instances and a wide baseline" pick training/eval slices in milliseconds.
- **Versioning and lineage:** **DVC** tracks dataset versions tied to git commits (LakeFS is the alternative if data moves fully to R2 and needs branch-style versioning of the bucket); every scene records the generator recipe version, random seed and asset hashes. Every model checkpoint records the dataset version it was trained on, so any result can be traced back to exactly how its data was made.
- **Data-quality gates:** §10.2 validation runs as a pipeline step; real-data loaders also validate (missing GT, mismatched image counts, corrupt files).
- **OLTP vs. OLAP separation (Moon):** Postgres runs the live job queue and app data only. A scheduled ETL job exports completed job telemetry, timings, feedback and corrections to **date-partitioned Parquet** in R2, queried with DuckDB for the admin analytics and drift monitoring. Heavy aggregation queries never touch the transactional queue.
- **Pipeline orchestration:** a simple, rerunnable DAG (`make`-style targets or a lightweight tool like Prefect) for render → validate → shard → run models → extract features → catalog.

### 16B.2 Track K: Reliability, observability and MLOps (Strong → Moon)
- **Golden benchmark regression gate (Strong):** a frozen 5-scene "golden set" run on a **self-hosted runner on the laptop GPU** (GitHub's hosted runners have no GPU), producing a table of MAPE, association F1, VRAM and latency vs. the last `main` result.
  - **Security (required for a public repo):** the GPU workflow triggers **only on `push` to `main` and on `workflow_dispatch`** (manual run by the owner). It **never** triggers on `pull_request` or `pull_request_target`, so code from forks can never run on the laptop. To check a PR's branch, the owner reviews the diff first, then starts `workflow_dispatch` on it by hand.
  - Runner hygiene: run it as a separate low-privilege user inside its own WSL2 distro, with no personal files mounted, no cloud credentials or repo secrets available to the job, restricted to this one repo, and with a job timeout.
  - A tiny CPU version (synthetic geometry only) runs on every PR in GitHub-hosted CI, where untrusted code is safe to run.
- **OpenTelemetry tracing (Moon):** instrument FastAPI → Postgres queue claim → worker stages (download, detect, reconstruct, associate, upload) → R2, exported to a free tier (Grafana Cloud/Tempo or Honeycomb). One trace shows a phone upload all the way to the GPU forward pass.
- **Metrics and alerts:** queue depth, job failure rate, p50/p95 per stage, GPU memory, cost per job; alerts on failure-rate spikes.
- **Drift monitoring (Strong):** compare incoming pilot jobs with the benchmark distribution on frame count, image resolution, mean depth confidence, lift-failure rate, detection density and prompt categories. Show the comparison on the admin page and flag large shifts (e.g., a population-stability index above a threshold), since a shift means benchmark numbers may not hold for those users.
- **Shadow deployments + model feature flags (Moon):** a `model_routing` table selects the detector, backbone and associator per job. A new version (e.g., CountOnce-Match v3, VGGT-Ω) can run in **shadow mode** on a percentage of live jobs: it runs in the background, its count is logged next to the live result, and users never see it. Promote it only after the shadow deltas and the golden benchmark look good.
- **Load test:** k6 or Locust against the API with a fake worker; record throughput and error rates.
- **Code quality:** `mypy` (or pyright) strict on core modules, `pre-commit` hooks (ruff, formatting, type checks), property-based tests with Hypothesis for associator invariants (permutation invariance, cannot-link, monotonic behavior as `k` grows).

### 16B.3 Track L: User-correction active-learning flywheel (Moon tier)
- **Capture:** Split / Merge / Reject actions (§14.8) become labeled pairs: Split → hard negative (these two detections are different objects); Merge → hard positive; Reject → false-positive detection.
- **Store:** a `corrections` table (job, detection ids, action, user, timestamp), exported to the feature store as a versioned dataset.
- **Selection:** prioritize scenes where the matcher was least confident (match probabilities near 0.5, wide conformal intervals) when asking pilot users to review.
- **Retrain:** a manual (or weekly) job fine-tunes CountOnce-Match on synthetic data + accumulated corrections, evaluates on the frozen test split and the golden set, and only promotes the new checkpoint if it doesn't regress (model registry with versioned checkpoints and a promotion log).
- **Report:** the before/after metrics of each promoted version, and how many corrections it took to move the numbers.
- **Consent:** only uses data from users who opted in to helping improve the model.

### 16B.4 Track M: Showcase and credibility (Strong → Moon)
- **2-minute demo video:** phone photos → upload → live progress → count, 3D view and "Explain this match". Recruiters watch videos more than they read READMEs.
- **Project page / blog post** with interactive figures (headline chart, failure gallery, the 3D viewer embedded) and a plain-English explanation.
- **Talk or poster** at a Cal Poly event (e.g., AI Day or a club meeting), and the professor's group.
- **Write-up quality bar:** every claim has a number with a CI; limitations section; reproducibility instructions; links to code, model, data and demo.

---

## 17. Product and frontend spec

**Goal:** CountOnce looks and works like a real product, not a research repo with a JSON endpoint. A stranger (a recruiter, a farm manager, a researcher) can open one link and, within 30 seconds and without signing in or waiting for a GPU, see what it does, try it on a real example, and understand why the result is trustworthy.

### 17.1 Status of the current mockups (`docs/mockups/`, reviewed 2026-11-25)
- **Good:** accessible structure (skip link, labels, legend that doesn't rely on color, focus states), the right sections on the results page, honest "MOCKUP" labelling, light/dark modes, phone layout.
- **Not good enough yet:** they're wireframes. There's no landing page, no way to try it without uploading, the 3D viewer is a placeholder box with four dots, the result page has no "why" (no match explanation, no method summary), and nothing shows the benchmark evidence. Visually it reads as a form, not a product.
- **Verdict:** keep them as the information-architecture reference. Don't approve them as the visual design. The approval gate (§17.11) applies to the new design below.

### 17.2 Product positioning (what the front page says)
- **One line:** "Count anything from a few photos, without counting it twice."
- **Three proof points** under it, each filled only with real numbers (`TODO: fill from results` until then): accuracy vs. adding up per-photo counts; works from 3–40 photos with no depth sensor; open benchmark with confidence intervals.
- **Primary action:** "Try a live example" (demo mode, no sign-in). **Secondary:** "Count your own photos" (sign-in).

### 17.3 Information architecture (all pages)
| Route | Page | Sign-in | Tier |
|---|---|---|---|
| `/` | Landing page: pitch, interactive hero (a real example result spinning in 3D), proof points, how it works in 4 steps, demo gallery, benchmark teaser, links | No | Strong |
| `/demo` | Demo gallery: 6–10 precomputed real scenes (Fuji-SfM trees where the license allows showing derived results, own captures: bike racks, chairs, shelves, an orchard row) | No | Strong |
| `/demo/[scene]` | Full results experience on a precomputed scene (same UI as a real job, labelled "Example") | No | Strong |
| `/new` | New count: capture guide, upload, prompt, optional example box, pre-flight checks | Yes | Strong |
| `/jobs/[id]` | Live progress, then the results experience | Yes (or share token) | Strong |
| `/history` | Past counts with thumbnails, counts, status, filters | Yes | Strong |
| `/s/[token]` | Read-only shared result, with a rich link preview (Open Graph image) | No | Strong |
| `/benchmark` | Benchmark explorer: interactive figures and tables from `results/`, method filters, CIs, failure gallery, leaderboard | No | Strong |
| `/how-it-works` | Interactive explainer: pipeline walk-through with a live example at each stage (detections → 3D lift → merging → count) | No | Strong |
| `/sites`, `/sites/[id]` | Sites and visits over time (Track G, Moon) | Yes | Moon |
| `/docs` | Developer docs: REST API, Python package, MCP tool, with copy-paste examples | No | Moon |
| `/status` | Worker online/offline, queue length, recent job times (public, read-only) | No | Strong |
| `/about` | Who built it, links to repo, paper, VGGT benchmark project, contact | No | Strong |
| `/privacy` | What's stored, for how long, blurring and EXIF stripping, opt-in for training | No | Strong |

### 17.4 Demo mode (the single most important frontend feature)
- **Why:** the GPU worker is a laptop that is often off, and the API on free hosting sleeps. A recruiter who clicks the link must still see a working product instantly.
- **How:** `scripts/build_demo.py` runs the real pipeline on the demo scenes and exports, per scene, `result.json`, downsampled `scene.ply` (converted to a compact binary point format for the browser), `markers.json`, overlay images, thumbnails and the match evidence. These are static files served from the frontend's CDN (or R2). The demo pages need **no backend at all**.
- **Honesty:** every demo page shows "Precomputed example. Results produced by CountOnce v[version] on [date]" and links to the exact commit and config. Demo outputs must come from real runs; never hand-edited.
- **Regenerate** whenever the pipeline version changes (a CI job checks that demo files' `schema_version` and model versions match the current release).

### 17.5 The results experience (`/jobs/[id]`, `/demo/[scene]`, `/s/[token]`)
Layout (desktop): count card across the top; left 60% the 3D viewer; right 40% a tabbed panel (Photos, Explain, Warnings, Details); photo strip along the bottom. On phones these stack, with the count card first.

**Count card:** the count in large type, the object name, the interval ("likely 44–51, 90% range, split conformal, n = 18 calibration scenes"), a confidence badge (High / Medium / Low with the reason), and buttons for Report, CSV, Share.

**3D viewer (react-three-fiber):**
- Point cloud rendered with level of detail (≤ 300k points on phones, ≤ 1M on desktop), colored from the photos
- One numbered marker per counted object (sphere sized by its estimated size), plus faint markers for rejected detections (toggle)
- **Camera frusta** for every photo; clicking one **flies the view to that camera** and fades the photo in over the 3D scene, so the user sees exactly what that photo saw (this is the "wow" moment)
- **Match lines:** selecting an object draws lines from its 3D marker to the camera of every photo where it was detected
- Two-way linking: hover a marker → its boxes highlight in every photo; hover a box → its marker pulses in 3D
- Controls: orbit, zoom, reset, "top view", screenshot (PNG), fullscreen; keyboard accessible (arrow keys to step through objects)
- Optional Gaussian splat view (Moon) via a toggle, for scenes that have one

**Photos tab:** every photo with boxes labelled NEW #n / DUPLICATE of #n / REJECTED (reason); filter by status; click a box to select its object everywhere.

**Explain tab ("Explain this match"):** pick any two detections (or select one object to list its members). Shows 3D distance in object sizes, reprojection error, appearance similarity, calibrated match probability, the decision and the rule or model that made it ("same photo, can't be the same object", "below the 0.5 match threshold", "merged by the learned matcher"). A short plain-English sentence on top, numbers below.

**Warnings tab:** each warning with what it means and how to fix it ("Photos 5 and 6 barely overlap: take one more photo between them").

**Details tab:** model versions, settings, timings per stage, cache hit, interval method, schema version, link to the raw `result.json`.

**Corrections (signed-in owner only):** Split / Merge / Reject on a selected object, with undo. The count updates instantly in the UI, and the correction is saved (Track L). Merging two detections from the same photo is blocked with an explanation.

**Feedback:** "Looks right" / "Off by about ___", one tap.

### 17.6 New count flow (`/new`)
1. **Capture guide** (dismissible, shown first time): a short animated diagram of walking an arc around the objects, plus three rules (overlap, steady, good light). Includes "Use my camera" on phones (`<input capture>`), so users can shoot directly from the page.
2. **Upload:** drag-and-drop or camera; thumbnails appear immediately; **resumable multipart upload** with per-file progress (§14.8).
3. **Pre-flight checks in the browser** (§14.8): blur score and near-duplicate detection per photo, shown as inline badges with "remove" and "retake" actions, before anything is uploaded.
4. **What to count:** a text field with suggestions ("apples", "grape clusters", "bikes"), an optional "not this" field (negative prompt), and an optional "draw a box around one example" step.
5. **Submit:** idempotent; goes straight to the progress page.

### 17.7 Progress page
Live stages over SSE (with polling fallback): queued → checking files → detecting → reconstructing in 3D → matching → done, each with elapsed time. Shows queue position, the "processing paused, your job will run when the GPU is back online" banner when no worker is online, and an email-me-when-done toggle (Moon). On completion it transitions straight into the results experience.

### 17.8 Benchmark explorer (`/benchmark`)
- Interactive versions of the headline figures (§9.7): count error vs. number of views per method with CI bands; ordered vs. shuffled; by scene type; association F1 for v1/v2/v3; VLM comparison
- Filters by dataset, method, view count; hover for exact values and CIs
- The failure gallery with each case's overlay, 3D view and one-line cause
- Leaderboard table (CountOnce-Bench, Track I) with a "submit your method" link
- Every number pulled from `results/*.csv` at build time; a footer shows the commit and date
- Empty state while results are missing: "Benchmark in progress", never placeholder numbers

### 17.9 Tech stack
| Concern | Choice | Why |
|---|---|---|
| Framework | **Next.js (App Router) + TypeScript (strict)** | Static demo pages + dynamic job pages in one app; first-class on Vercel |
| Styling | **Tailwind CSS** + **shadcn/ui** (Radix primitives) | Accessible components, consistent design tokens |
| 3D | **three.js via react-three-fiber + drei** | Declarative scene, picking, camera animation |
| Data fetching | **TanStack Query** | Caching, polling fallback, retries |
| API types | **OpenAPI → TypeScript** generated from the FastAPI schema (`openapi-typescript`), plus **Zod** validation of `result.json` by `schema_version` | Frontend and backend can't silently drift |
| Charts | Observable Plot or Recharts | Benchmark figures with CI bands |
| Uploads | S3-compatible multipart (Uppy or a small custom client) | Resumable uploads |
| Auth | Google sign-in (Auth.js) | Free; matches §14 |
| Testing | **Vitest** + React Testing Library (components), **Playwright** (end-to-end, including a demo-mode smoke test on every PR), **axe** accessibility checks in Playwright | |
| Quality gates | Lighthouse CI on `/`, `/demo/[scene]`, `/benchmark` | See budgets below |
| Hosting | Vercel (frontend) as its own project; demo assets on the CDN or R2 | $0 (§14.7) |
| Repo layout | `web/` folder in the same repo (monorepo), with its own CI job | One place for code, shared schemas |

### 17.10 Design system
- **Tokens:** one accent color, neutrals, status colors (new / duplicate / rejected) that also differ by outline style and label; 4 px spacing scale; radius and shadow scale; light and dark themes from the same tokens.
- **Type:** a clean sans-serif (e.g., Inter or Geist) with tabular numbers for counts; 16 px+ body; the count uses a large display size.
- **Motion:** short, purposeful transitions (camera fly-to, count card reveal); respect `prefers-reduced-motion`.
- **Components:** count card, confidence badge, warning card, photo tile with overlays, 3D viewer frame, stage list, evidence table, empty/loading/error states, toasts.
- **Accessibility target:** WCAG 2.2 AA; keyboard-only use of every feature, including selecting objects in 3D; screen-reader text for the count, interval and warnings.
- **Performance budgets:** landing page Largest Contentful Paint < 2.5 s on a mid-range phone; JS < 200 KB gzipped before the 3D viewer loads; the viewer and point cloud load lazily with a progress bar; Lighthouse ≥ 90 for performance, accessibility and best practices.

### 17.11 Design process and approval gate
1. **High-fidelity designs** of: landing, demo gallery, results (desktop + phone), new count, progress, benchmark, empty/error states. Use the existing mockups for structure only.
2. Shreev reviews (checklist in `docs/mockups/README.md`, updated for the new pages). **For the v6.0 finishing pass, review happens after the build (§0.1 rule 5)**; requested changes become follow-up tasks.
3. Build **demo mode first** (landing, demo gallery, results experience on precomputed scenes). That alone is a shippable product link.
4. Then the signed-in flow (new count, progress, history, sharing), wired to the API.
5. Then benchmark explorer, how-it-works, status, docs.

### 17.12 Definition of done for the frontend
- A stranger can open the link on a phone, tap "Try a live example", spin the 3D scene, tap an object and see why it was counted once, in under 30 seconds, with the laptop worker turned off
- Every number on every page traces to a real result file (no placeholders in production builds; a build check fails on `TODO: fill from results` strings outside clearly marked empty states)
- Playwright end-to-end tests pass for demo mode and the signed-in flow; axe reports no serious accessibility issues; Lighthouse budgets met
- A 2-minute demo video (Track M) recorded from the real product

---

## 18. Testing strategy

### Unit tests
- **Synthetic geometry test (most important):** random cameras + N random 3D points, projected with noise, random drops and false positives → associators must recover N within tolerance at several noise levels
- **Cannot-link:** same-frame detections never merge, even when coincident in 3D
- Lifting returns the right center on a known plane + box
- Confidence range widens with noise
- Metrics correct on toy inputs; association F1 correct on toy clusterings
- Sinkhorn rows/columns sum correctly with the dustbin
- Ingest blur filter

### Model tests (Track C)
- Overfit test: the matcher reaches ~100% train accuracy on 10 synthetic scenes
- Permutation test: shuffling view order doesn't change the output instances
- Data loader yields consistent shapes and labels

### Integration tests
- End-to-end CLI on a tiny bundled scene (CPU-capable where possible)
- Synthetic generation smoke test (render 2 tiny scenes in Docker) marked as slow

### Product tests
- API tests with a fake worker; worker test against local Postgres + storage emulator
- Frontend component tests; manual checklist for the 3D viewer
- Load test: 20 queued jobs, worker processes them in order, none lost after a simulated worker crash

### CI
GitHub Actions: lint + type check + unit + API tests + the CPU synthetic-geometry golden test on every push. GPU, Docker and slow tests are marked and run on the self-hosted laptop runner (the golden benchmark gate, §16B.2) or locally.

### Additional tests (v4)
- **Property-based tests** (Hypothesis) for associators: output is invariant to view order and detection order; cannot-link is never violated; counts don't increase as the merge radius grows (for v1)
- **Cycle-consistency test:** synchronized matches are transitively consistent on random noisy inputs
- **Conformal test:** on synthetic data with known noise, empirical coverage is close to the target level
- **Idempotency test:** submitting the same job twice with one idempotency key queues it once
- **Data-quality test:** each quarantine rule triggers on a deliberately broken scene
- **Schema compatibility test:** stored examples of every older `result.json` version still load and render
- **Media sanitization tests:** EXIF/GPS is gone after ingest; oversized, malformed and unsupported files are rejected with the right error code
- **Error taxonomy test:** permanent errors go straight to the dead-letter status without retries; retryable errors retry with backoff
- **OOM ladder test:** a simulated OOM at each stage falls through the ladder correctly
- **Cache test:** re-running the same photos with a new prompt skips reconstruction and gives identical geometry
- **Zero-count test:** an empty synthetic scene returns 0 with an interval that includes 0

---

## 19. Phases, tasks and acceptance criteria

### Phase 0: Validate, literature, model and data survey (this week) — Core
- [ ] Human: niche interviews and professor meeting (§20)
- [ ] Literature check: arXiv/Scholar/recent CVPR, ICCV, ECCV, AAAI and ACM MM for multi-view counting, video counting, 3D counting, open-vocabulary counting, fruit counting with SfM, feed-forward reconstruction for counting/detection. Read CountVid and DG-Net closely. Write `docs/RELATED_WORK.md`.
- [ ] `docs/MODEL_SURVEY.md`: CountGD, CountGD++, GroundingDINO, CountVid, DG-Net, C-DOG, VGGT, VGGT-Ω, VGG-T³ (code, weights, license, output format, VRAM)
- [ ] `docs/DATASETS.md`: every dataset in §9.1 (availability, license, format, size)
- **Done when:** gates G0 and G1 decided; niche, detector and datasets picked

### Phase 1: Prototype (~2 weeks, winter break) — Core
- [ ] Repo setup; env check; ported VGGT wrapper, memory guard, frame extraction
- [ ] Data model, detector wrapper, lifting, v1 associator, export, CLI
- [ ] Synthetic geometry unit test passing
- [ ] **Gate G2 (week 1):** one foliage scene + one easy scene; report in `docs/GATES.md`
- [ ] First 5 hand-counted scenes captured and run
- **Done when:** `countonce run` gives deduplicated counts + overlays + 3D markers on 5 real scenes; failures written down

### Phase 2: Core benchmark (~2–3 weeks) — Core
- [ ] Full own dataset (10–20 scenes) + identity labels on 3–5; labeling tool
- [ ] Fuji-SfM loader + GT identity matching
- [ ] v2 associator, confidence range, warnings
- [ ] Baselines 1–6 (C-DOG and CountVid attempted)
- [ ] Sweeps incl. ordered vs. unordered; ablations; figures; failure gallery
- [ ] Frozen test split (`bench/splits.json`); bootstrap CIs and paired tests on all comparisons (§9.8)
- [ ] README results section + EXPLAINER.md draft
- **Done when:** gate G3 decided; Core figures (with CIs) generated from `results/bench.csv`

### Phase 2.5: Strong-tier evaluation (~1–2 weeks, can overlap Phase 3)
- [ ] Frontier VLM baseline on a scene subset (§9.3 item 8); costs and model versions logged
- [ ] OneScene-Bench evaluation
- [ ] Synchronization baseline (§9.3 item 9)
- [ ] Error slicing (§9.9); warnings updated from the slice results
- [ ] Human baseline study (§9.3 item 10)
- [ ] Zero-count / distractor suite and false positive rate (§9.8b)
- **Done when:** VLM, slicing, zero-count and human-baseline figures exist with CIs

### Phase 3: Learned matcher (~3–4 weeks) — Strong
- [ ] Kubric Docker setup; **test GPU rendering on the RTX 5070 first** (fallback: CPU or BlenderProc); renderer-agnostic scene recipe; data-quality gates; 1,000-scene smoke set; `docs/SYNTH.md`
- [ ] Track J: WebDataset shards, Parquet feature store, DuckDB catalog, DVC versioning, rerunnable data DAG
- [ ] Run Detect + Reconstruct + Lift on synthetic scenes; build training examples
- [ ] CountOnce-Match model, Sinkhorn + dustbin, instance formation
- [ ] Overfit and permutation tests pass
- [ ] Scale to 10,000+ scenes; train; track experiments
- [ ] Cycle-consistency loss + inference-time synchronization; ablation table
- [ ] Evaluate on Fuji-SfM + own captures; sim-to-real analysis; fine-tune ablation (≥3 seeds)
- [ ] Split conformal intervals calibrated and coverage reported (§8.8)
- [ ] Feature-group attribution ablation; ECE + temperature scaling (§10.5)
- [ ] Golden benchmark regression gate on the self-hosted runner (§16B.2)
- [ ] **Gate G4:** v3 vs v2 on real data
- **Done when:** v3 results + figures in `results/`; G4 decided

### Phase 4: Inference optimization (~1–2 weeks, can overlap Phase 3) — Strong
- [ ] Baseline profile; SDPA, bf16, `torch.compile`, Torch-TensorRT attempt, batching, caching
- [ ] Content-addressable stage caching; `expandable_segments` + OOM fallback ladder (§13 items 6b–6c)
- [ ] `docs/PROFILE.md` with before/after table
- **Done when:** end-to-end speedup measured and accuracy impact reported

### Phase 5: Product + pilot (~3–4 weeks build + 2–4 weeks pilot) — Strong
- [ ] Mockups → approval
- [ ] API, DB, storage, queue, worker, auth, SSE progress
- [ ] Frontend per §17: demo mode first (landing, demo gallery, results experience on precomputed real scenes), then new count, progress, history, sharing, benchmark explorer, how-it-works, status
- [ ] Product upgrades (§14.8): resumable uploads, idempotency keys, browser pre-flight check, "Explain this match", correction tools, face/plate blurring, EXIF/GPS stripping + ffprobe validation, inspection report
- [ ] Typed error taxonomy + dead-letter status (§14.4); schema versioning (§8.10); drift monitoring (§16B.2)
- [ ] Load test; metrics and alerts (§16B.2)
- [ ] Demo video + project page (§16B.4)
- [ ] Admin metrics; deploy; phone end-to-end test
- [ ] Extra benchmarks: RGBD-VideoCount (vs DG-Net) and a VideoCount subset (vs CountVid)
- [ ] Pilot with 3–10 users; `docs/PILOT.md` with usage, repeat use, p50 time, feedback accuracy
- **Done when:** real users ran real counts; numbers recorded

### Phase 6: Moon tier (late Spring 2027 semester and summer)
- [ ] Backbones: VGGT-Ω, VGG-T³ (Track E)
- [ ] Detector fine-tuning + CountGD++ swap-in (Track D)
- [ ] Cloud GPU worker with scale-to-zero + cost per job; cold-start engineering and `docs/SERVING.md`
- [ ] Custom Triton kernel + Nsight trace (§13 item 8)
- [ ] OpenTelemetry tracing (§16B.2)
- [ ] User-correction flywheel with model registry and promotion log (§16B.3)
- [ ] Gaussian splat view toggle (§14.8)
- [ ] CountOnce-Bench public benchmark + leaderboard; model card + datasheet (§16)
- [ ] Harvest/yield validation with a partner (§9.10)
- [ ] Pose-graph chunk stitching (§8.3); dynamic-scene outlier filter + VGGT-Ω dynamic comparison (§8.3b)
- [ ] FP8/INT8/NVFP4 low-precision experiments; activation checkpointing for training (§13 items 10–11)
- [ ] OLTP → OLAP Parquet export (§16B.1); shadow deployments + feature flags (§16B.2)
- [ ] Sites and visits view (§14.8)
- [ ] pip package, developer API, MCP server (Track H)
- [ ] HF Space, HF model, HF dataset releases (Track I)
- [ ] Paper: target an agriculture-vision or 3D-vision workshop at CVPR/ICCV (check deadlines with the professor); arXiv preprint
- **Done when:** paper submitted and releases public

---

## 20. Human tasks (things only Shreev can do)

| Task | Blocks | Details |
|---|---|---|
| **Niche interviews** | Phase 0 | 3–5 ag (faculty, farm staff, researchers) + 2 backup niche. Questions below. Quotes in `docs/INTERVIEWS.md` |
| **Professor meeting** | Phase 0 | Bring the pitch, §2.3 table, §2.4 claim, benchmark plan. Ask about novelty vs CountVid/DG-Net, scope, venue, what result makes it publishable |
| **Captures + hand counts** | Phases 1–2 | Follow §9.2. Ask permission on farms/private property; avoid faces |
| **Identity labeling** | Phase 2 | Use the labeling tool on 3–5 scenes |
| **Gated model access** | Phase 6 | Request VGGT-Ω weights (and the commercial VGGT checkpoint only if ever charging money) |
| **Design approval** | Phase 5 | Approve mockups |
| **Pilot recruiting + check-ins** | Phase 5 | Go back to interviewees with real pain; weekly check-ins |
| **Gate decisions** | All | Decide at each gate |

**Interview questions** (how counting works now, not "would you use this?"):
1. How do you estimate yield or counts today? Who does it, and how many hours does it take?
2. What happens when the count is wrong? What decision depends on it?
3. Walk me through the last time you counted something like this, step by step.
4. Have you tried apps or drones? Where did they fall short?
5. Which crops or objects would be worth counting if it took five minutes?
6. Who else does this, here or elsewhere, and how do they do it differently?

---

## 21. Decision gates

| Gate | When | Pass condition | If it fails |
|---|---|---|---|
| **G0: Need** | End of Phase 0 | ≥3 of 5 interviewees describe real counting pain with a decision depending on it | Backup niche; if that fails too, keep it as a research project without a pilot |
| **G1: Novelty** | End of Phase 0 | Lit check + professor confirm the §2.4 claim isn't already done | Narrow further (for example, orchard-specific sparse-view counting, or the learned matcher alone) |
| **G2: Feasibility** | Phase 1, week 1 | VGGT geometry usable on one foliage + one easy scene | Make non-foliage the main niche; keep ag as a stretch; try VGGT-Ω early |
| **G3: Geometry helps** | End of Phase 2 | v1/v2 beat Sum and Max on most scenes | Publish the honest negative result with analysis; still a valid benchmark |
| **G4: Learning helps** | End of Phase 3 | v3 beats v2 on association F1 on Fuji-SfM and own captures | Analyze the sim-to-real gap; improve randomization or fine-tune; if it still fails, report it and keep v2 as default |

---

## 22. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| VGGT fails on foliage/repetitive texture | Medium–high | G2 in week 1; VGGT-Ω; backup niche |
| Noisy depth merges or splits objects | High | Size-relative radius, cannot-link, v2 matching, learned v3, confidence range |
| Novelty overlaps CountVid/DG-Net | Medium | Narrowed claim (§2.4); direct comparisons; professor check |
| Synthetic-to-real gap for the matcher | Medium–high | Run the real pipeline on synthetic images; domain randomization; Fuji-SfM fine-tune; honest analysis |
| Kubric setup pain (Docker, Blender 2.93) | Medium | Smoke test early; fallback to plain Blender scripts or simpler procedural scenes |
| Training compute on a laptop | Medium | Small model; precomputed features; short pay-per-second cloud runs only if needed |
| VRAM limits | Medium | Memory guard; sequential stages; chunking; VGGT-Ω/VGG-T³ |
| Baselines won't run | Medium | Document; use the others |
| Laptop worker offline during the pilot | High | Paused banner; requeue; cloud worker in Phase 6 |
| Licenses block commercial use | High | Free pilot only; check every license (§24) before charging |
| Kubric's Blender 2.93 can't GPU-render on the RTX 5070 | Medium | Test first; CPU render; BlenderProc fallback; renderer-agnostic recipe |
| VLM baseline costs or API limits | Low–medium | Subset of scenes; log cost; one open-weight model run locally |
| Conformal intervals too wide with few real scenes | Medium | Calibrate on synthetic + real; report coverage by domain; state the limitation |
| Self-hosted CI runner on the laptop is offline | Medium | Gate is advisory when the runner is offline; run manually before merging |
| Flywheel gets too few corrections | High | Ask pilot users to review low-confidence scenes; still report what changed |
| Scope creep / unfinished | **High** | Tiers, gates, phase criteria; Core first, always. **v4 added nothing to Core.** |

---

## 23. Documentation

### README.md (results first, no hype)
1. One-line summary + headline figure
2. Key findings: 3–5 bullets with real numbers, including at least one weakness
3. Demo link + screenshot
4. Results table (methods × views; MAPE, within-10%, association F1, runtime)
5. How it works (diagram)
6. Limitations
7. Reproduce: setup, data download scripts, one command per phase
8. Credits, citations (VGGT, VGGT-Ω, CountGD/CountGD++, Kubric, datasets) and licenses
9. Link to the VGGT benchmark repo

### EXPLAINER.md (A-to-Z study guide)
1. The problem: why multi-photo counting double-counts and single-photo counting misses
2. Cameras 101: intrinsics, extrinsics, projection, depth, point maps
3. VGGT: DINO patch tokens, camera tokens, frame-wise vs. global attention, heads, no bundle adjustment, memory vs. frames; what VGGT-Ω and VGG-T³ change
4. Open-vocabulary counting: CountGD/CountGD++, text vs. exemplar vs. negative prompts
5. Lifting: median, confidence filtering, relative scale
6. v1 clustering and the cannot-link rule
7. v2: reprojection, depth test, appearance features, Hungarian assignment, hallucination filter
8. v3: graph neural networks, self- and cross-attention, Sinkhorn, dustbin, training loss, instance formation
9. Synthetic data: Kubric, domain randomization, why run the real pipeline on synthetic images, sim-to-real gap
10. Evaluation: baselines, metrics, fair comparison, results, every failure case and its cause
11. Inference optimization: profiling, SDPA/FlashAttention, bf16, compile/TensorRT, what worked
12. System design: queue with SKIP LOCKED, heartbeats, retries, SSE, signed URLs, scale-to-zero workers
13. **Interview Q&A (30+)**, including:
    - Why lift to 3D instead of tracking through video? When does tracking win?
    - How is this different from CountVid and DG-Net?
    - Why can't same-photo detections merge?
    - How do you set thresholds when VGGT has no real units?
    - Why train on synthetic data? How did you check it transfers?
    - What does the dustbin do in Sinkhorn matching?
    - What's the biggest weakness of your benchmark?
    - Which optimization gave the biggest speedup, and why?
    - What happens if a worker dies mid-job?
    - How would you scale to 1,000 jobs a day?
14. Glossary

### Paper (Phase 6)
Structure: introduction, related work (§2.3), method (pipeline + CountOnce-Match), synthetic data, experiments (§9), ablations, limitations. Target a workshop first, with an arXiv preprint.

### Other docs
`docs/PLAN.md` (this file), `DEVIATIONS.md`, `GATES.md`, `RELATED_WORK.md`, `MODEL_SURVEY.md`, `DATASETS.md`, `SYNTH.md`, `PROFILE.md`, `INTERVIEWS.md`, `PILOT.md`

---

## 24. Licenses and data handling

| Item | License status (verify in Phase 0) | Implication |
|---|---|---|
| VGGT original checkpoint | Non-commercial | Fine for research and a free pilot |
| VGGT-1B-Commercial | Commercial use allowed (excl. military), by application | Required before charging money |
| VGGT-Ω | FAIR non-commercial research; gated | Research only |
| CountGD | MIT | OK |
| CountGD++, CountVid, DG-Net, C-DOG, GroundingDINO, DINOv2 | Check each | Record in MODEL_SURVEY |
| Fuji-SfM | Sources conflict (CC BY 4.0 vs. CC BY-NC-SA; authors say research/education only) | Treat as research-only |
| WGISD | CC BY-NC 4.0 | Non-commercial |
| MinneApple, AppleGrowthVision, RGBD-VideoCount, VideoCount, FSC-147 | Check each | Don't redistribute unless allowed |
| Kubric | Open source; check asset licenses (ShapeNet, Google Scanned Objects, HDRIs) | Only release synthetic data whose assets allow it |

- The repo ships **download scripts, not data**
- Own captures: permission on private property; avoid faces; raw captures stay out of git
- User uploads: private, deleted after 30 days, never used for training without opt-in
- **The product stays free/research until every model and dataset in the deployed path allows commercial use**

---

## 25. Timeline and weekly time budget

| Phase | Tier | Rough timing |
|---|---|---|
| 0. Validate + surveys | Core | Thanksgiving week |
| 1. Prototype | Core | Winter break, ~2 weeks |
| 2. Core benchmark | Core | Late winter break → start of the Spring 2027 semester, ~2–3 weeks |
| 3. Learned matcher | Strong | Spring 2027 semester (Jan–Feb), ~3–4 weeks |
| 4. Inference optimization | Strong | Overlaps Phase 3, ~1–2 weeks |
| 5. Product + pilot | Strong | Spring 2027 semester (Mar–May), ~5–8 weeks incl. pilot |
| 6. Moon tier + paper | Moon | Late Spring 2027 semester and summer 2027 |

**Weekly budget:**
| Activity | Hours/week |
|---|---|
| CountOnce | 6–8 (up to 15 over breaks) |
| CountGD++ learning | ~5 |
| DSA / Scaler | ≥4 |
| H4I, Hopscotch, research, classes | as needed |

If CountOnce runs over budget two weeks in a row, cut the current tier's optional items, not DSA. **Core is the only tier that must be finished; Strong and Moon are upgrades.**

**Realistic horizon with v4:** Core by the start of the Spring 2027 semester; Strong by the end of the Spring 2027 semester; Moon items over summer 2027. Your main recruiting push is fall 2027, so aim for **Core + most of Strong finished and written up by August 2027**, and treat Moon items as whatever is done by then.

---

## 26. Deliverables checklist and resume entries

**Core**
- [ ] `countonce` repo: tests, CI, reproducible setup, results-first README
- [ ] Benchmark on own captures + Fuji-SfM vs. baselines incl. CountVid; figures; failure gallery
- [ ] Frozen test split; bootstrap CIs and paired tests on every reported comparison
- [ ] EXPLAINER.md draft

**Strong**
- [ ] CountOnce-Match trained on synthetic data (Kubric or BlenderProc); cycle consistency + synchronization; sim-to-real results over ≥3 seeds
- [ ] Feature-group attribution ablation; calibration (ECE + temperature scaling)
- [ ] Split conformal intervals with ≥15 real calibration scenes (or Jackknife+ below that), coverage reported
- [ ] Frontier VLM baseline; OneScene-Bench, RGBD-VideoCount and VideoCount comparisons
- [ ] Zero-count / distractor false positive rate; error slicing; human baseline
- [ ] Data engineering: shards, Parquet feature store, DuckDB catalog, DVC, data-quality gates
- [ ] Profiling report with measured speedups; stage caching; OOM fallback ladder
- [ ] Golden benchmark gate on the secured self-hosted runner
- [ ] Deployed web app: resumable uploads, idempotency, pre-flight check, "Explain this match", correction tools, face/plate blurring with privacy masks, EXIF/GPS stripping, ffprobe validation, typed errors + dead-letter, versioned schemas, inspection report
- [ ] Drift monitoring; pilot numbers
- [ ] Demo video + project page

**Moon**
- [ ] VGGT-Ω / VGG-T³ results; dynamic-scene filter; pose-graph stitching
- [ ] Detector fine-tuning; CountGD++ swap-in; negative-prompt evaluation
- [ ] Custom Triton kernel; FP8/INT8/NVFP4 experiments; cold-start serving report
- [ ] OpenTelemetry tracing; shadow deployments; OLTP → OLAP export
- [ ] Correction flywheel with promotion log; sites/visits over time; Gaussian splat toggle
- [ ] pip package, developer API, MCP server
- [ ] HF Space, model and dataset releases; CountOnce-Bench + leaderboard; model card + datasheet
- [ ] Harvest/yield validation with a partner
- [ ] Workshop paper + arXiv preprint

**Resume / LinkedIn (real numbers only):**

> **CountOnce: Open-Vocabulary 3D Counting from a Few Photos**
> Combined an open-vocabulary counter with feed-forward 3D reconstruction (VGGT, CVPR 2025 Best Paper) to count objects across unordered photos without double counting.
> • Trained a multi-view graph matching model on [N]k synthetic Kubric scenes; raised association F1 from [A] to [B] on a real apple orchard (Fuji-SfM)
> • Cut count error from [X]% (summing per-photo counts) to [Y]% on [M] hand-counted scenes; [Z]× faster end-to-end after profiling and compiling the GPU pipeline
> • Deployed a web app (FastAPI, Postgres job queue, GPU workers, three.js 3D viewer) used by [U] pilot users; released the model and a synthetic dataset on Hugging Face
> Python · PyTorch · VGGT · Kubric · FastAPI · Postgres · three.js · [link]

---

## 27. Glossary

- **Open-vocabulary counter:** counts whatever object you describe in text
- **Feed-forward reconstruction:** 3D from images in one network pass, no optimization loop
- **Point map:** for every pixel, the 3D point it sees
- **Intrinsics / extrinsics:** a camera's lens properties / its position and rotation
- **Lifting:** turning a 2D detection into a 3D point using depth
- **Association:** deciding which detections in different photos are the same object
- **Cannot-link:** a rule that two items must never be grouped together
- **Hungarian algorithm:** the lowest-cost one-to-one matching
- **Sinkhorn:** turns a score matrix into a soft one-to-one assignment by alternately normalizing rows and columns
- **Dustbin:** an extra "no match" slot in matching, for objects that are hidden or out of view
- **Graph neural network (GNN):** a network that passes information between connected items (here, detections)
- **Domain randomization:** varying synthetic data widely so a model trained on it works on real data
- **Sim-to-real gap:** how much worse a model trained on synthetic data does on real data
- **SDPA / FlashAttention:** fast GPU implementations of attention
- **SKIP LOCKED:** a Postgres feature that lets several workers claim different jobs without clashing
- **Scale to zero:** cloud workers that shut down when there's no work, so you pay nothing when idle
- **MCP:** Model Context Protocol, a standard way for AI agents to call external tools
- **MAE / MAPE:** average absolute error / average percentage error
- **F1:** a score combining precision and recall
- **Cycle consistency:** if A matches B and B matches C, then A must match C
- **Permutation synchronization:** a method that turns noisy pairwise matches into one consistent multi-view matching
- **Conformal prediction:** a way to build intervals with a guaranteed average coverage, using a held-out calibration set
- **Bootstrap confidence interval:** an uncertainty range found by re-sampling the test scenes many times
- **Error slicing:** measuring error separately for subgroups of scenes (dense vs. sparse, etc.)
- **WebDataset / Parquet / DuckDB / DVC:** sharded training archives / columnar data files / an in-process analytics database / data version control
- **Idempotency key:** an ID that makes a repeated request have the same effect as one request
- **OpenTelemetry:** a standard for tracing a request across services
- **Triton (kernel):** a Python-like language for writing custom GPU kernels
- **Active learning:** choosing which examples to label next so the model improves fastest
- **ECE (Expected Calibration Error):** how far a model's confidence is from how often it's actually right
- **Temperature scaling:** a one-parameter fix that makes a model's probabilities better calibrated
- **Pose graph:** a graph of camera (or chunk) positions linked by measured relative transforms, optimized together to remove drift
- **Dead-letter queue:** where jobs that can't succeed go, instead of being retried forever
- **Shadow deployment:** running a new model on real traffic in the background without showing its results
- **OLTP / OLAP:** live transactional database work / analytics queries over history
- **FP8 / NVFP4:** 8-bit and 4-bit floating-point number formats for faster, smaller models
- **EXIF:** metadata stored inside photos, which can include GPS location

---

## 28. Brutal audit (2026-11-25) and recovery plan

> **Snapshot taken before the M1–M22 pass.** Statements in §28.1–28.3 describe the repo at that time. For the current state see `docs/STATUS.md` and §0.6.

An honest review of the repo as it stands after the overnight agent runs (114 of 115 tasks checked off).

### 28.1 What exists
- About **48,000 lines of Python** in `countonce/`, `bench/`, `synth/` and `scripts/`, plus about **15,000 lines of tests**, nearly all written by unattended agents over about two days
- Pipeline modules for every track: ingest, detection (GroundingDINO-tiny via Hugging Face works; **CountGD is still a stub**), VGGT reconstruction, lifting, v1/v2/sync/v3 association, sensitivity and conformal intervals, warnings, exports, reports
- Infrastructure far beyond Core: API (FastAPI, live on Vercel returning a JSON root), job store and worker, uploads, MCP server, model routing, shadow deployment, cold-start serving, sites, LLM router, golden GPU gate, DVC/DAG configs, paper draft (`paper/main.tex`), static mockups
- **Real runs so far:** VGGT-1B + GroundingDINO-tiny on the RTX 5070 for profiling; a 10-frame Fuji-SfM window (count 42, interval 14–120); the bundled 5-object tiny scene, where the real detector + VGGT path **counts 7, not 5**
- v3 matcher trained on **numpy geometry episodes** (not rendered images): association F1 0.964 vs. 0.741 for v2 on held-out synthetic episodes
- **Real-data benchmark rows: zero.** Every results table is `TODO: fill from results`. No interviews, no own captures, no pilot.

### 28.2 Scores (out of 10)
| Area | Score | Why |
|---|---|---|
| Engineering breadth / scaffolding | **8** | Almost every planned subsystem has a module, tests and docs |
| Honesty of claims | **9** | No fabricated numbers; limitations are stated plainly. Keep this. |
| Core result (does it count correctly?) | **1** | No real benchmark numbers. The two real runs show over-counting (7 vs. 5) and a very wide interval (14–120) |
| ML novelty demonstrated | **3** | v3 beats v2 only on synthetic geometry episodes, with no rendering, no real detector noise and no real data |
| Product / frontend | **1** | No frontend app. The live URL returns JSON. Mockups are wireframes |
| Code you can defend in an interview | **2–3** | 48k lines written by agents in two days. No one can explain all of it, and interviewers will pick any file |
| Scope discipline | **2** | The plan's own rule ("Core before Strong before Moon") was broken: MCP, shadow routing, cold-start serving, sites and a paper draft exist before the pipeline has a single real result |
| Public repo presentation | **4** | Agent instructions (`CLAUDE.md`, a 111 KB `TASKS_OVERNIGHT.md`) and personal Windows paths are public; no license; README is mostly TODO tables |
| **Overall, as a recruiter would see it today** | **3** | Impressive volume, no evidence it works, nothing to click |
| **Ceiling if the recovery plan is done** | **9** | The design is genuinely strong; it needs results, a product and ownership |

### 28.3 The core problem in one sentence
**A skyscraper of scaffolding on a foundation that hasn't been poured:** everything around the counter exists, but the counter itself hasn't been shown to count correctly on real data.

A 48k-line repo where nothing is proven is weaker than an 8k-line repo with one strong, real result and a working demo. Volume without evidence also invites the question "did you write this?", and the answer has to survive follow-up questions.

### 28.4 Recovery order (do these in order; don't start a step until the previous one is done)
**Step 1: Freeze features (now).** No new modules, tracks or infrastructure until Step 4 is done. The overnight orchestrator only gets bug-fix and evaluation tasks.

**Step 2: Make the counter correct on the easy case (1–3 days).**
- Debug the tiny scene (7 vs. 5) with the real detector + VGGT path: check per-frame detections (duplicate boxes on one object → per-frame NMS), lifting (background pixels in the box → mask or tighter region), and the merge radius. Add the tiny scene with the real path as a regression test once it's fixed.
- Decide the Core detector: either get real CountGD weights working, or **commit to GroundingDINO for Core** and document it. Don't leave a stub on the main path. **Make the working detector the config default** (today it defaults to the CountGD stub).
- **One reliable path:** one detector, one backbone (VGGT), one associator. Everything else is marked "planned" in `docs/STATUS.md`.

**Step 3: Real numbers on Fuji-SfM (1 week).** Finish task 115: download Fuji-SfM, run the Core sweep on the frozen test trees, and fill `docs/RESULTS_FUJI.md` and the README from `results/bench.csv`. Whatever the result is (good, bad or mixed), it's the first real evidence. If v1/v2 don't beat Sum/Max, that's gate G3: write it up honestly and fix it before anything else. Freeze all settings on the validation trees **before** touching the test trees; report failed and missing runs, not only successful ones. Then run the controlled failure-source experiments (§9.6) and the error decomposition (§9.4), and spend the next effort wherever the biggest error source is.

**Step 4: Own captures + interviews (1–2 weeks, overlapping Step 3).** 10 hand-counted scenes (at least 3 easy, including a bike rack), and the Phase 0 interviews. Run the Core benchmark on them.

**Step 5: Demo mode + real frontend (2–3 weeks).** Build `web/` per §17, demo mode first, using precomputed **real** results from Steps 3–4. This turns the repo into a product link.

**Step 6: Ownership pass (ongoing, ~2 hours a week).** Read and be able to explain, without notes: `lift.py`, `merge/cluster.py`, `merge/match.py`, `merge/sync.py`, `merge/v3.py`, `conformal.py`, `confidence.py`, `bench/metrics.py`, `bench/stats.py`, `pipeline.py`. That's the part interviewers will ask about. Rewrite anything you can't explain until you can.

**Step 7: Then the learned matcher on rendered data.** Kubric or BlenderProc renders → real detector + VGGT on them → retrain v3 → evaluate on Fuji-SfM and own captures. Only this result can support the "learned matcher transfers to real orchards" claim.

**Step 8: Then everything else** in tier order (Strong, then Moon).

### 28.5 Repo hygiene (one afternoon, before sharing the link anywhere)
- **License:** add one for the code (Apache-2.0 or MIT). State separately that model weights (VGGT non-commercial) and datasets keep their own licenses.
- **Agent files:** move `TASKS_OVERNIGHT.md`, the orchestrator scripts and the agent rules in `CLAUDE.md` out of the public repo (a private repo or a gitignored folder), and remove personal paths (e.g., `/mnt/c/Users/...`). Using AI agents is fine and worth discussing in interviews; a 111 KB task log in the repo root isn't the thing to lead with.
- **Quarantine unproven scaffolding:** move Moon-tier modules that aren't on the working path (shadow routing, cold-start serving, sites, LLM router, MCP server, paper draft) into `experimental/` or a branch, so a reviewer opening the repo sees a small, legible core first. Bring each back when its tier starts.
- **README:** results first. Until real results exist, replace the giant TODO tables with one short "Status: benchmark in progress" section and the tiny-scene demo. A wall of TODO cells reads as unfinished.
- **Commit history:** small, meaningful commits from here on. Make sure the public history isn't a single squashed commit for 48k lines.
- **CI:** make sure the badge is green on `main`; a red badge on the first screen hurts.
- **Live URL:** point the main link at the frontend once Step 5 exists; move the API to an `api.` address or a separate Vercel project.

### 28.6 What would make it top 1% (after recovery)
1. **One headline result with a CI:** "cut counting error from X% to Y% on N real orchard trees and M own scenes"
2. **A demo link that works instantly on a phone**, with the camera fly-to and "explain this match"
3. **The learned matcher beating geometric rules on real data** after training on rendered synthetic scenes
4. **A clear, honest comparison** with CountVid and frontier VLMs
5. **Real users** in a pilot, with numbers
6. **Being able to explain every line of the core** in an interview

Everything else in this plan is supporting detail for those six.

### 28.6b External review (ChatGPT, 2026-11-25): verified findings
Each item was checked against commit `63d8e9b`.

| Finding | Verified? | Action | Recovery step |
|---|---|---|---|
| Default detector is `countgd`, whose load and forward pass raise `NotImplementedError` (`countonce/config.py`, `detect/countgd.py`) | **Confirmed** | Make GroundingDINO (HF) the default; mark CountGD/CountGD++ "planned"; test the exact README quickstart from a clean environment | Step 2 |
| Association F1 isn't the product metric; v3's 0.964 F1 still leaves a mean count error of 19.1 | **Confirmed** (`docs/RESULTS_V3_SYNTH.md`) | Lead with count MAE, signed bias, within-10% and zero-scene false counts; add the error decomposition (§9.4) | Step 3 |
| Pairwise F1 ignores unlabelled detections, so injected false positives don't hurt it | **Confirmed** (`bench/metrics.py` docstring: "Unlabelled detections are ignored") | Count FPs explicitly in the decomposition; report FP rate next to F1 | Step 3 |
| The synthetic stand-in reconstruction uses NaN depth, so the occlusion/visibility check never fires | **Confirmed** (`bench/synth_assoc.py`) | Label the v3 synthetic result a component test; add a variant with finite noisy depth; real evidence comes from Step 7 | Step 7 |
| v1/v2 baselines run with default settings in the v3 comparison | **Confirmed** | Tune every method on the validation split before comparing (§9.6) | Step 3 |
| v2 processes frames in sorted frame-id order; the invariance test reorders input lists but keeps frame ids, so it doesn't test real photo-order invariance | **Confirmed** (`merge/match.py`, `tests/test_associators.py`) | Real permutation test (§9.6); fix or narrow the claim | Step 3 |
| CI installs a `db` extra that `pyproject.toml` doesn't define | **Confirmed** (`ci.yml` line 65) | Add the extra or remove it from CI; keep the badge green | Hygiene |
| README says real detector adapters haven't been run, but `docs/PROFILE.md` reports a GroundingDINO sanity run on real images | **Confirmed** | One authoritative status table (below) | Hygiene |
| Real detector + VGGT counts 7 on the 5-object tiny scene | **Confirmed** (already Step 2) | Correctness investigation before more performance work | Step 2 |
| A VGGT smoke check deduplicated five views down to one | Not found in the docs reviewed | Check during Step 2; if true, it means that check never tested multi-view merging | Step 2 |
| `docs/RELATED_WORK.md` says no paper was read in full, yet makes "does not do X" claims | **Confirmed** | Read the closest papers fully; state the hypothesis in §2.4 instead of a novelty claim | Step 3 |
| The README front page is placeholders and TODO tables | **Confirmed** (also §28.5) | Replace with a short honest demo (below) | Step 5 |

**Authoritative status table (`docs/STATUS.md`, replaces scattered status notes):** one row per component (each detector, backbone, associator, interval method, API piece) with four columns: implemented · tested on synthetic/fixtures · run with real models · benchmarked on held-out real data. The README links to it and never contradicts it. Small evaluation CSVs and artifact references (commit, config, data version) are published next to each report.

**README front page (until the full product exists):** input photos, predicted count vs. the hand count, the same object IDs drawn across views, one measured results table, one representative failure, and one command someone else can run. Planned experiments move to a "Roadmap" section further down.

### 28.7 Rules added for the agents (§4)
9. **No new modules outside the current recovery step (§28.4).** Bug fixes, evaluation and docs only.
10. **Every new feature needs a real-data check** before it's marked done; "tested on synthetic or tmp fixtures only" means the task stays open.
11. **No more scaffolding for a tier that hasn't started.**

---

## Appendix A: What changed in v4, v4.1 and v4.2, and why

An external review suggested additions. Each was checked; this is what was kept and corrected.

| Suggestion | Verdict | What the plan does |
|---|---|---|
| VGGT-Ω, VGG-T³, CountGD++ and CountVid have public repos | **Confirmed** (VGG-T³ code at nv-dvl/vgg-ttt and model on Hugging Face) | Model table updated |
| Cycle-consistent / synchronized multi-view matching | **Correct and important.** Pairwise matching isn't transitively consistent; permutation synchronization and CLEAR are established fixes | Added to Track C (Strong) + baseline |
| Frontier VLM baseline | **Correct and high value.** OneScene-Bench (NeurIPS 2026 workshop) already shows VLM unique-counting accuracy collapsing on views with partial overlap | VLM baseline + OneScene-Bench evaluation (Strong) |
| Split conformal intervals | **Correct, with a caveat:** coverage is average (marginal) and assumes calibration scenes resemble test scenes; with few real scenes intervals are wide | Strong-tier default; Core keeps a clearly labeled sensitivity range |
| Error slicing | **Correct** | §9.9 (Strong), feeding the warnings |
| WebDataset/Lance + DuckDB + versioning + quality gates | **Mostly correct, but the I/O concern was overstated:** the matcher trains on small per-detection features, not raw images. Sharding matters for raw renders and reproducibility | Track J (Strong) |
| Kubric's Blender 2.93 may fight the RTX 5070 | **Plausible** (Kubric does pin Blender 2.93); BlenderProc is a real alternative | Test first; fallbacks (§10.2) |
| Custom Triton kernel | **Good learning and ML-infra signal**, but may not beat `torch.compile` on small matrices | Moon; report honestly |
| Cold-start engineering | **Correct** | Moon (§13 item 9) |
| ML regression gate in CI | **Correct idea, but GitHub's hosted runners have no GPU** | Self-hosted laptop runner + CPU-only proxy (§16B.2) |
| OpenTelemetry tracing | Correct | Moon |
| Resumable uploads + idempotency keys | Correct | Strong |
| Browser pre-flight check | Correct | Strong |
| "Explain this match" + splat toggle | Correct | Explain = Strong; splat = Moon |
| Correction flywheel | Correct, but needs enough pilot users | Moon, with confidence-based selection |
| "Staff-level flagship" framing | **Overstated.** Strong evidence comes from finished, measured results, not from the number of technologies | Tiers kept; nothing added to Core |

### Second review (v4.1)
| Suggestion | Verdict | Where it went |
|---|---|---|
| Dynamic-scene outlier filter + VGGT-Ω dynamic eval | Correct (VGGT assumes static scenes; VGGT-Ω reports dynamic-scene support) | §8.3b, Moon |
| Sim(3) pose-graph stitching for long captures | Correct | §8.3, Moon |
| Zero-count / distractor false positive rate | Correct and important | §9.8b, Strong |
| ECE + temperature scaling | Correct | §10.5, Strong |
| FP8 / NVFP4 quantization | **Partly correct:** the hardware supports FP8 and FP4, but `sm_120` software support is still uneven, so it's an experiment that may need TensorRT or patches | §13 item 10, Moon, with that caveat |
| Activation checkpointing | **Only helps training**, not inference | §13 item 11, training only |
| `expandable_segments` + OOM ladder | Correct | §13 item 6c, Strong |
| Content-addressable caching for re-prompting | Correct; the "60–75% savings" figure isn't verified, so the plan measures it | §13 item 6b, Strong |
| Paired bootstrap + Wilcoxon | Already in v4 (Core, §9.8) | No change |
| Feature-group attribution ablation | Correct, and directly tests the central claim | §10.5, Strong |
| Drift monitoring | Correct | §16B.2, Strong |
| OLTP → OLAP Parquet export | Correct | §16B.1, Moon |
| Schema versioning | Correct | §8.10, Strong |
| EXIF/GPS stripping + ffprobe sanitization | Correct and important (privacy) | §14.8, Strong |
| Typed errors + dead-letter queue | Correct | §14.4, Strong |
| Shadow deployments + feature flags | Correct | §16B.2, Moon |
| PDF inspection report; sites/visits over time | Correct | §14.8; report Strong, sites Moon |
| Tier placement of cycle consistency and the CI gate (review said Moon) | Kept in Strong: both are cheap and protect the main results | No change |

### Cleanup pass (v4.2)
- §8.8: split conformal needs n ≥ 9 calibration scenes for a finite 90% quantile (n ≥ 19 for 95%); real-data intervals require ≥ 15 real calibration scenes, otherwise Jackknife+/CV+ (worst-case 1−2α guarantee)
- §16B.2: the GPU runner only triggers on `push` to `main` and `workflow_dispatch`, never on `pull_request`/`pull_request_target`; runner isolation rules added
- §5: WSL2 data on native ext4 (not `/mnt/c`, which goes through slow 9P); `.wslconfig` memory/CPU/swap tuning
- §14.8 + §8.4: privacy masks from face/plate blurring are excluded from depth/point confidence during lifting
- §7 and §14.3: data model and database schema synced with all v4.1 features
- Title, hardware table (Ryzen 9, WSL2), TOC, §13 numbering, §25 timeline (Spring 2027 semester) and §26 checklist updated

**Also added after a gap check (not in the review):** frozen test split and bootstrap CIs on every metric (Core); paired significance tests and multiple seeds; human baseline (time + accuracy); harvest/yield validation with a partner; CountOnce-Bench public benchmark + leaderboard; model card and datasheet; automatic face/plate blurring; property-based tests, type checking and load testing; a demo video and project page.
