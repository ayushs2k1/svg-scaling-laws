#!/usr/bin/env bash
# usage: run_part3.sh <best_lr_mup>   (use the lr selected from the µP sweep on Tiny)
set -euo pipefail
cd "$(dirname "$0")/.."
LR=${1:?best µP LR required}

for SIZE in tiny small medium large xl; do
  TAG="mup_${SIZE}"
  echo "=== µP ${SIZE} (lr=${LR}) ==="
  python -m src.train \
      --config "configs/${SIZE}.yaml" --data data/bin \
      --out results/runs --tag "${TAG}" --lr "${LR}" --epochs 1 --mup --base-d-model 128
done

python -m src.scaling_fit \
    --runs results/runs --out results/scaling.json --plot results/scaling.png \
    --sp-pattern "sp_*" --mup-pattern "mup_*"
