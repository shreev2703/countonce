#!/usr/bin/env bash
# Re-tune (score_threshold x merge_k) on the training trees under protocol fuji_core_v3, then run v2
# on the frozen test trees. Resumable (finished slices are skipped); writes runs/fuji_core/*.done.
# Run from the repo root: setsid nohup scripts/tune_fuji_v3.sh > runs/fuji_core/v3_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source .venv-gpu/bin/activate
export PYTHONPATH=.
W=runs/fuji_core
if [ ! -f $W/tune_v3.done ]; then
  for s in 0.2 0.3 0.4 0.5; do for k in 0.3 0.5 0.8; do
    python scripts/run_fuji_core.py bench --split val --ks 5 10 --assocs v1 v2 \
      --score-threshold $s --merge-k $k --config /nonexistent --budget-min 600 \
      --csv results/fuji_tune3_s${s}_k${k}.csv > $W/tune3_s${s}_k${k}.log 2>&1
  done; done
  touch $W/tune_v3.done
fi
# test run needs the refrozen configs/fuji_core.yaml (written by the interactive session)
while [ ! -f $W/freeze_v3.ok ]; do sleep 30; done
if [ ! -f $W/test_v3.done ]; then
  python scripts/run_fuji_core.py bench --split test --ks 3 5 10 20 --assocs v1 v2 \
    --budget-min 600 > $W/test_v3.log 2>&1 && touch $W/test_v3.done
fi
