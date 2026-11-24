# CountOnce-Match model card (template)

Every number below is `TODO: fill from results`. Do not fill from memory or estimates.

## Model details
- Name / version: TODO: fill from results
- Architecture: multi-view attentional matcher (PLAN §10.3)
- Weights hash / release date: TODO: fill from results
- License: TODO: decide

## Intended use
Counting distinct physical objects across overlapping photos or video. Not for safety-critical counting.

## Training data
- Synthetic scenes (CountOnce-Synth): TODO: fill from results
- Real scenes and licenses: TODO: fill from results
- Consented user corrections: TODO: fill from results

## Evaluation (CountOnce-Bench, frozen test split)
| Metric | Value | 95% scene-bootstrap CI |
|---|---|---|
| Count MAE | TODO: fill from results | TODO: fill from results |
| MAPE | TODO: fill from results | TODO: fill from results |
| Association pairwise F1 | TODO: fill from results | TODO: fill from results |
| Interval coverage (nominal 95%) | TODO: fill from results | TODO: fill from results |
| Zero-count false-positive rate | TODO: fill from results | TODO: fill from results |
| Paired diff vs reference method (p) | TODO: fill from results | TODO: fill from results |

## Limitations and failure modes
TODO: fill from results (slices: reflective, repetitive texture, heavy occlusion, low overlap).

## Ethics, privacy and carbon
- Privacy masking and EXIF handling: TODO: describe
- Training compute / emissions: TODO: fill from results
