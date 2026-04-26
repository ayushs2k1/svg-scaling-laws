#!/usr/bin/env bash
# usage: run_lr_sweep.sh <model> <sp|mup>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-tiny}
MODE=${2:-sp}

EXTRA=""
[[ "$MODE" == "mup" ]] && EXTRA="--mup"

python -m src.sweep \
    --config "configs/${MODEL}.yaml" \
    --data data/bin \
    --out-runs results/runs \
    --out "results/sweep_${MODE}.json" \
    --tag-prefix "${MODEL}_${MODE}" \
    --lrs 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 \
    $EXTRA
