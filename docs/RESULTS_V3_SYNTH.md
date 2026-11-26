# v3 matcher: synthetic component test

**Component test, not an accuracy claim.** It tests only the association step, on generated
geometry. v1 and v2 ran with untuned default settings, and the stand-in reconstruction has NaN
depth, so the v2 visibility (occlusion) check never fires. The planned re-run with tuned baselines
and noisy depth (PLAN §0.2 M5) was **not completed**; the table below predates it.

First capped v3 run (take 2), PLAN §10.4. Synthetic data only; **no real-data v3 rows exist**
(Fuji-SfM v3 cells stay `TODO: fill from results`). Related: [[HANDOFF]], [[DECISIONS]].

- Data: `data/dag_v3_capped` (800 train / 100 val / 100 test episodes), dataset version
  `e9673a1bbeb2` (verified against the recorded hash before training).
- Config: `configs/match_v3_capped.yaml` (width 128, 4 layers, 4 heads, AdamW lr 3e-4, batch 8,
  count weight 0.1, cycle weight 0, seed 0, 45-minute cap, early-stop patience 8).
- Run: 37 epochs, stopped by the wall-clock cap at 46.2 min; best val F1
  0.9430 at epoch 35. Curves: `results/match_v3_capped_curves.csv` (local,
  gitignored) and `runs/match_v3_capped/log.jsonl`. Registered as `v001` in `models/registry.json`.
- Eval: `PYTHONPATH=. .venv-gpu/bin/python scripts/eval_v3_synth.py`, 100 test episodes,
  2,000-sample bootstrap over episodes, mean [95% CI]. Raw rows: `results/v3_capped_synth_eval.csv`.

| method | assoc F1 | duplicate rate | miss rate | count error (abs) |
|---|---|---|---|---|
| v1 | 0.468 [0.448, 0.489] | 0.643 [0.626, 0.659] | 0.000 [0.000, 0.000] | 79.5 [70.7, 89.4] |
| v2 | 0.741 [0.724, 0.756] | 0.454 [0.433, 0.476] | 0.000 [0.000, 0.000] | 47.6 [42.0, 53.9] |
| v3 | 0.964 [0.957, 0.970] | 0.037 [0.029, 0.047] | 0.004 [0.002, 0.006] | 19.1 [16.7, 21.7] |

Paired v3-minus-baseline differences (with CIs) are in the CSV.
