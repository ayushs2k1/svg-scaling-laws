# Scaling Laws for SVG Language Models

CS-GY 6923 optional project (Spring 2026). Trains decoder-only Transformers at multiple
scales on SVG code, fits power-law scaling curves, compares standard parameterization
(SP) vs. Maximal Update Parameterization (µP), and generates SVG samples.

## Layout

```
configs/         per-model architecture YAMLs (tiny/small/medium/large/xl)
src/
  data/prepare.py        download + clean + split SVG datasets
  tokenizer/train_bpe.py train a BPE tokenizer with HF tokenizers
  data/pack.py           tokenize + pack into binary shards
  model.py               nanoGPT-style decoder Transformer (SP and µP)
  train.py               single-run trainer (cosine + warmup, AdamW)
  sweep.py               LR sweep driver (Tiny model, SP and µP)
  scaling_fit.py         power-law fit + bootstrap CI + extrapolation
  generate.py            sampling (temperature / top-k / top-p / prefix)
  eval.py                perplexity + XML validity + render rate
scripts/         end-to-end shell pipelines
report/          LaTeX report template that pulls from results JSON
results/         (created at runtime) per-run JSON, plots, samples
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Data
python -m src.data.prepare --out data/raw --max-tokens 1024
python -m src.tokenizer.train_bpe --in data/raw --out data/tok --vocab-size 4096
python -m src.data.pack --in data/raw --tok data/tok --out data/bin --seq-len 1024

# 2. LR sweep on Tiny (SP)
bash scripts/run_lr_sweep.sh tiny sp

# 3. Train all sizes (SP) at best LR
bash scripts/run_part2.sh <best_lr_sp>

# 4. LR sweep on Tiny (µP), then transfer
bash scripts/run_lr_sweep.sh tiny mup
bash scripts/run_part3.sh <best_lr_mup>

# 5. Power-law fits + extrapolation
python -m src.scaling_fit --runs results/runs --out results/scaling.json

# 6. Best model: train longer + generate + eval
python -m src.train --config configs/xl.yaml --mup --lr <best_lr_mup> --epochs 3 --tag best
python -m src.generate --ckpt results/runs/best/ckpt.pt --n-uncond 20 --n-prefix 10
python -m src.eval --ckpt results/runs/best/ckpt.pt --samples results/samples

# 7. Build report
cd report && latexmk -pdf report.tex
```

## Compute notes

- Designed for a single A100/V100 (Colab Pro / NYU Greene). Largest model (XL ~88M)
  fits comfortably in 40GB at seq_len=1024, batch=64.
- Each scaling-plot run is exactly 1 epoch over the train shard (~100M tokens). On A100
  this is roughly: Tiny 5min, Small 10min, Medium 20min, Large 45min, XL 90min.
- Total compute budget for full project (Parts 2+3+4): ~10–15 A100-hours.

## What is from nanoGPT vs. ours

- `model.py`: structure is nanoGPT-style (causal self-attention block, MLP,
  pre-LayerNorm). µP adaptations (1/d attention, output-layer scaling, init scaling,
  per-param-group LR multipliers via `mup`) are ours.
- `train.py`: AdamW + cosine + warmup loop is conventional; throughput, memory,
  per-run JSON logging, sweep integration are ours.
- All preprocessing, tokenization, scaling-law fitting, evaluation, and generation
  scripts are ours.

## Honesty

All numbers in the report come from actual runs. `report/report.tex` reads JSON
artifacts under `results/`; if a run hasn't happened, the corresponding figure/table
shows `TODO` rather than a fabricated value.
