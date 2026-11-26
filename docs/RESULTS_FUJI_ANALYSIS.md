## Reading of the numbers (hand-written; appended by `bench.fuji_report`)

Protocol `fuji_core_v2`; thresholds frozen on the training-split trees before the test trees were
touched (`configs/fuji_core.yaml`: score 0.2, merge_k 0.3). 320 test rows (v1, v2 and baselines),
none failed, none dropped. 128 more rows (sync and v3, below) were added later, 448 in total. Two accidental duplicate rows (a second writer started by mistake) were removed from
`results/bench.csv`; their predictions were identical to the originals.

**Gate G3 ("v1/v2 beat Sum and Max on most scenes"): not met.** Per slice (tree x side, 16 per k),
the pipeline's absolute count error is lower than the baseline's in this many slices:

| k | v1 < Sum | v2 < Sum | v1 < Max | v2 < Max |
|---|---|---|---|---|
| 3 | 0/16 | 3/16 | 7/16 | 7/16 |
| 5 | 3/16 | 5/16 | 2/16 | 2/16 |
| 10 | 12/16 | 12/16 | 3/16 | 5/16 |
| 20 | 12/16 | 11/16 | 10/16 | 11/16 |

Only k=20 clears the bar against both baselines (v2 vs Max, paired MAE diff -51.9, CI [-81.7,
-20.3]). At k=3 and 5 all methods are within noise of each other, and at k=10 Max has the lowest
MAE (31.1 against v2's 40.1). v2 beats v1 on MAE only at k=10 (-12.5, CI [-20.5, -4.6]). Treat
this as the honest negative/mixed result of PLAN §21: the decision on how to proceed is the
owner's; see `docs/GATES.md`.

**Caveats.** The miss rate is about 0.95 at k=20 (undefined where fewer than all slices have
identity labels), so most visible GT apples are never detected at score 0.2; counts are
dominated by detection recall and by what the depth test calls visible. Assoc F1, duplicate rate
and miss rate stay `TODO` where some slices have no identity-labelled detections: those cells are
not filled rather than averaged over a subset. CIs resample the 8 trees, not the 16 slices.

### Failure sources (M6 harness on Fuji, k=10, 16 test windows)

`scripts/run_failure_sources_fuji.py`; associators tuned on the 6 training windows
(v1 k=0.5; v2 max_reproj_norm 0.5, depth_tol 0.5), identical upstream inputs per condition.
Fuji has no per-pixel depth or own camera poses, so **oracle geometry cannot be built**: only
"oracle detections + estimated geometry" and "full" were run, and the geometry share of the error
is not separated. Predictions here are not ROI-filtered, so MAE differs from the table above.

| condition | method | MAE | signed bias | missed | duplicates | false positives |
|---|---|---|---|---|---|---|
| full | v1 | 79.6 | +70.1 | 51.7 | 0.8 | 121.1 |
| full | v2 | 80.4 | +71.3 | 51.7 | 0.6 | 122.3 |
| oracle det + est. geometry | v1 | 24.1 | +24.1 | 10.2 | 34.4 | 0.0 |
| oracle det + est. geometry | v2 | 26.9 | +26.9 | 10.2 | 37.1 | 0.0 |

With a perfect detector the remaining error is mostly duplicates (about 34-37 per window): the
association/geometry stage over-splits apples. With the real detector the error is dominated by
false positives (about 121 per window, detections that match no visible GT apple within 24 px;
some are real apples outside the GT, so this is an upper bound) and by about 52 misses. The
biggest error source is therefore the detector, then duplicates from association under estimated
geometry. Oracle-detection boxes assume a 4 cm apple radius (not measured).

### Sync and v3 on the test trees (added later)

64 `countonce_sync` and 64 `countonce_v3` rows, one per v2 test window, none failed. Same
detector, backbone, prompt, score threshold, merge_k and ground truth as the v2 rows. Sync's
settings were frozen on the training-split trees (`configs/fuji_core_sync.yaml`); v3 is the
registered checkpoint `v001`, trained on synthetic data only, with no Fuji fine-tune.

- Paired MAE against v2 (table above): sync is within 1.6 apples at every k. v3 is lower by 1.7 at
  k=3, higher by 2.4 at k=5, by 25.5 [9.8, 40.2] at k=10 and by 47.9 [-1.5, 107.6] at k=20.
- v3's signed bias at k=10 and 20 (+61, +99) is close to Sum's (+65, +113): it merges few
  detections. At k=3, where the pipeline under-counts on average, merging less lowers MAE.
- Association F1 exists only for slices with identity-labelled detections, so it is not in the table.
  The subset comparison is in `docs/GATES.md` under G4.
