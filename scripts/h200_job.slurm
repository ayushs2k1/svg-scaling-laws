#!/bin/bash
# SLURM batch script — runs the full svg-scaling-laws pipeline on one H200 GPU.
# Usage (after setup_torch.sh):
#   sbatch scripts/h200_job.slurm
#
# Check your SLURM account name first:
#   my_slurm_accounts
# Then set SBATCH_ACCOUNT below or pass: sbatch --account=<account> scripts/h200_job.slurm

#SBATCH --job-name=svg-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/svg_%j.out
#SBATCH --error=logs/svg_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

NETID="${USER}"
SCRATCH="/scratch/${NETID}"
ENV_DIR="${SCRATCH}/conda_envs/svg-scaling"
REPO_DIR="${SCRATCH}/svg-scaling-laws"

cd "${REPO_DIR}"
mkdir -p logs results/runs

# ── Activate environment ──────────────────────────────────────────────────────
module purge
module load anaconda3/2024.02
source activate "${ENV_DIR}"
export PYTHONNOUSERSITE=True

echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time:  $(date)"
echo ""

# ── 1. Data preparation (skip if already done) ────────────────────────────────
if [ ! -f data/bin/meta.json ]; then
    echo "=== 1/7 Data: download + clean ==="
    python -m src.data.prepare \
        --out data/raw \
        --datasets starvector/svg-icons-simple starvector/svg-emoji-simple \
        --max-chars 8192 --min-chars 50

    echo "=== 1/7 Data: train BPE tokenizer ==="
    python -m src.tokenizer.train_bpe \
        --in data/raw --out data/tok --vocab-size 4096

    echo "=== 1/7 Data: pack into binary shards ==="
    python -m src.data.pack \
        --in data/raw --tok data/tok --out data/bin --seq-len 1024
else
    echo "[skip] data/bin already exists"
fi

# ── 2. SP LR sweep ────────────────────────────────────────────────────────────
echo "=== 2/7 LR sweep: Standard Parameterization ==="
python -m src.sweep \
    --config configs/tiny.yaml --data data/bin \
    --out-runs results/runs --out results/sweep_sp.json \
    --tag-prefix tiny_sp \
    --lrs 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2

BEST_LR_SP=$(python -c "import json; print(json.load(open('results/sweep_sp.json'))['best']['lr'])")
echo "Best SP LR: ${BEST_LR_SP}"

# ── 3. Train all sizes SP ─────────────────────────────────────────────────────
echo "=== 3/7 Train all SP sizes at lr=${BEST_LR_SP} ==="
for SIZE in tiny small medium large xl; do
    echo "  Training SP ${SIZE}..."
    python -m src.train \
        --config "configs/${SIZE}.yaml" --data data/bin \
        --out results/runs --tag "sp_${SIZE}" --lr "${BEST_LR_SP}" --epochs 1
done

# ── 4. µP LR sweep ────────────────────────────────────────────────────────────
echo "=== 4/7 LR sweep: µP ==="
python -m src.sweep --mup \
    --config configs/tiny.yaml --data data/bin \
    --out-runs results/runs --out results/sweep_mup.json \
    --tag-prefix tiny_mup \
    --lrs 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2

BEST_LR_MUP=$(python -c "import json; print(json.load(open('results/sweep_mup.json'))['best']['lr'])")
echo "Best µP LR: ${BEST_LR_MUP}"

# ── 5. Train all sizes µP ─────────────────────────────────────────────────────
echo "=== 5/7 Train all µP sizes at lr=${BEST_LR_MUP} ==="
for SIZE in tiny small medium large xl; do
    echo "  Training µP ${SIZE}..."
    python -m src.train \
        --config "configs/${SIZE}.yaml" --data data/bin \
        --out results/runs --tag "mup_${SIZE}" \
        --lr "${BEST_LR_MUP}" --epochs 1 --mup --base-d-model 128
done

# ── 6. Scaling-law fit ────────────────────────────────────────────────────────
echo "=== 6/7 Scaling-law fit ==="
python -m src.scaling_fit \
    --runs results/runs --out results/scaling.json --plot results/scaling.png

python -m src.plot_curves --runs results/runs --pattern "sp_*"  --out results/sp_training_curves.png
python -m src.plot_curves --runs results/runs --pattern "mup_*" --out results/mup_training_curves.png
python -m src.plot_curves --runs results/runs --pattern "*"     --out results/all_training_curves.png

# ── 7. Best model: train longer, generate, evaluate ──────────────────────────
echo "=== 7/7 Best model (XL µP, 3 epochs) ==="
python -m src.train \
    --config configs/xl.yaml --data data/bin \
    --out results/runs --tag best \
    --lr "${BEST_LR_MUP}" --epochs 3 --mup --base-d-model 128

python -m src.generate \
    --ckpt results/runs/best/ckpt.pt --tok data/tok \
    --out results/samples --n-uncond 10 \
    --temperatures 0.5 0.8 1.0 --top-p 0.95

python -m src.eval \
    --ckpt results/runs/best/ckpt.pt --data data/bin \
    --samples results/samples --out results/eval.json

python report/build_tables.py

echo ""
echo "=== Pipeline complete at $(date) ==="
echo "Results in: ${REPO_DIR}/results/"
