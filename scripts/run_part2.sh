#!/usr/bin/env bash
# usage: run_part2.sh <best_lr_sp>   (use the lr selected from the SP sweep)
set -euo pipefail
cd "$(dirname "$0")/.."
LR=${1:?best LR required}

for SIZE in tiny small medium large xl; do
  TAG="sp_${SIZE}"
  echo "=== SP ${SIZE} (lr=${LR}) ==="
  python -m src.train \
      --config "configs/${SIZE}.yaml" --data data/bin \
      --out results/runs --tag "${TAG}" --lr "${LR}" --epochs 1
done

python -m src.scaling_fit \
    --runs results/runs --out results/scaling.json --plot results/scaling.png \
    --sp-pattern "sp_*"
