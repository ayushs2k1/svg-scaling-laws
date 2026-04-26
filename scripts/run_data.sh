#!/usr/bin/env bash
# Build the dataset end to end. Run once.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.data.prepare \
    --out data/raw \
    --datasets starvector/svg-icons-simple starvector/svg-emoji-simple \
    --max-chars 8192 --min-chars 50

python -m src.tokenizer.train_bpe \
    --in data/raw --out data/tok --vocab-size 4096

python -m src.data.pack \
    --in data/raw --tok data/tok --out data/bin --seq-len 1024
