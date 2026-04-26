#!/usr/bin/env bash
# usage: run_part4.sh <best_lr> <epochs>
# Trains the best (XL µP) model for more epochs, then generates and evaluates.
set -euo pipefail
cd "$(dirname "$0")/.."
LR=${1:?lr required}
EPOCHS=${2:-3}

python -m src.train \
    --config configs/xl.yaml --data data/bin \
    --out results/runs --tag "best" --lr "${LR}" --epochs "${EPOCHS}" --mup --base-d-model 128

python -m src.generate --ckpt results/runs/best/ckpt.pt --tok data/tok \
    --out results/samples --n-uncond 10 --temperatures 0.5 0.8 1.0 --top-p 0.95

python -m src.eval --ckpt results/runs/best/ckpt.pt --data data/bin \
    --samples results/samples --out results/eval.json
