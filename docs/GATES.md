# Gates and roadblocks

Log decision-gate failures (PLAN §21) and genuine roadblocks (missing credentials, gated weights,
broken downloads, two failed fix attempts) here: exact error, what was tried, what is needed.


## 2026-11-24: Human actions needed (logged by tech-lead audit, not overnight tasks)

- **Dev environment:** countonce has no venv. `ruff`/`pytest` currently run from the read-only
  `~/vggtProject/.venv`, and mypy has never been run. Needed: create `countonce/.venv` (Python 3.11) and
  `pip install -e .[dev,recon]`, then run `mypy` once.
- **Gates G0 / G1 (PLAN §21):** need the niche interviews and professor meeting (PLAN §20) before they
  can be decided. `docs/RELATED_WORK.md` has `TODO: verify` items that need a full-paper read.
- **Gate G2 (PLAN §21):** needs one foliage capture and one easy capture, then a VGGT run on the GPU.
- **Own captures + hand counts (PLAN §9.2):** 10–20 scenes, plus identity labels on 3–5 of them.
- **Fuji-SfM download and license (PLAN §9.1):** ~6.9 GB from Zenodo 3712808. The license is conflicting (see
  `docs/DATASETS.md`). Read `LICENSE.txt` and confirm the file layout so the loader's format TODOs can be closed.
- **Detector weights:** CountGD and GroundingDINO checkpoints need to be downloaded on a networked session
  before the real detector adapters can be checked end to end.

## Detector weights needed (2026-11-24, detector interface task)
- CountGD: clone niki-amini-naieni/CountGD, download its checkpoint (`checkpoint_fsc147_best.pth`) and the GroundingDINO Swin-B base weights it builds on; then implement `CountGDDetector._load`/`_infer` (currently stubs) and verify the output-format `TODO` in `countonce/detect/countgd.py`.
- GroundingDINO: `pip install groundingdino` (IDEA-Research), config `GroundingDINO_SwinT_OGC.py` and `groundingdino_swint_ogc.pth`; run `GroundingDinoDetector` once on a real frame to confirm `predict` score/box conventions.
