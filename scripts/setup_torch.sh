#!/usr/bin/env bash
# One-time environment setup on NYU Torch HPC.
# Run this from a login node AFTER cloning the repo to scratch:
#
#   cd /scratch/$USER
#   git clone https://github.com/ayushs2k1/svg-scaling-laws.git
#   cd svg-scaling-laws
#   bash scripts/setup_torch.sh
#
# After this completes, open OOD → Interactive Apps → Jupyter Notebook,
# select the "svg-scaling" kernel, and open notebooks/run.ipynb.
set -euo pipefail

NETID="${USER}"
SCRATCH="/scratch/${NETID}"
ENV_DIR="${SCRATCH}/conda_envs/svg-scaling"
REPO_DIR="${SCRATCH}/svg-scaling-laws"

echo "=== NYU Torch HPC setup for svg-scaling-laws ==="
echo "NetID:   ${NETID}"
echo "Scratch: ${SCRATCH}"
echo "Env:     ${ENV_DIR}"
echo ""

# ── Guard: must run on Torch, not locally ────────────────────────────────────
if [[ "$(uname)" == "Darwin" ]]; then
    echo "ERROR: This script must be run on the NYU Torch HPC cluster, not your Mac."
    echo ""
    echo "Steps:"
    echo "  1. ssh ${NETID}@login.torch.hpc.nyu.edu"
    echo "  2. cd /scratch/\$USER"
    echo "  3. git clone https://github.com/ayushs2k1/svg-scaling-laws.git"
    echo "  4. cd svg-scaling-laws"
    echo "  5. bash scripts/setup_torch.sh"
    exit 1
fi

if [[ ! -d "/scratch" ]]; then
    echo "ERROR: /scratch does not exist — this doesn't look like Torch HPC."
    echo "SSH in first: ssh ${NETID}@login.torch.hpc.nyu.edu"
    exit 1
fi

# ── 1. Load anaconda ──────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module purge
    module load anaconda3/2024.02
else
    echo "[warn] 'module' not found — using conda already on PATH"
fi

# Prevent base-env conda init from interfering with Singularity
conda deactivate 2>/dev/null || true

# ── 2. Write ~/.condarc so envs and pkg cache land on scratch (not $HOME) ─────
cat > "${HOME}/.condarc" <<EOF
envs_dirs:
  - ${SCRATCH}/conda_envs
pkgs_dirs:
  - ${SCRATCH}/conda_pkgs
always_copy: true
EOF

mkdir -p "${SCRATCH}/conda_pkgs"
ln -sfn "${SCRATCH}/conda_pkgs" "${HOME}/.conda/pkgs" 2>/dev/null || true

# ── 3. Create conda env ───────────────────────────────────────────────────────
if conda env list | grep -q "${ENV_DIR}"; then
    echo "[skip] Conda env already exists at ${ENV_DIR}"
else
    echo "[create] Creating conda env with Python 3.11..."
    conda create -p "${ENV_DIR}" python=3.11 -y
fi

source activate "${ENV_DIR}"

# Prevent ~/.local packages from shadowing conda packages
export PYTHONNOUSERSITE=True

# ── 4. Install cairosvg via conda-forge (bundles libcairo) ───────────────────
echo "[install] Installing cairosvg from conda-forge..."
conda install -p "${ENV_DIR}" -c conda-forge cairosvg -y

# ── 5. Install remaining Python deps via pip ──────────────────────────────────
echo "[install] Installing requirements.txt..."
pip install --no-user -r "${REPO_DIR}/requirements.txt"

# ── 6. Install ipykernel and register kernel for OOD Jupyter ─────────────────
echo "[install] Installing ipykernel..."
pip install --no-user ipykernel

KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/svg-scaling"
mkdir -p "${KERNEL_DIR}"

python -m ipykernel install --user --name svg-scaling --display-name "Python (svg-scaling)"

# ── 7. Create results/ dir structure so the notebook doesn't fail ─────────────
mkdir -p "${REPO_DIR}/results/runs"
mkdir -p "${REPO_DIR}/data/raw" "${REPO_DIR}/data/tok" "${REPO_DIR}/data/bin"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Go to https://ood.torch.hpc.nyu.edu"
echo "  2. Interactive Apps → Jupyter Notebook"
echo "  3. Set: Account = (your SLURM account from 'my_slurm_accounts')"
echo "           Partition = leave blank  |  GPU = h200  |  # GPUs = 1"
echo "           Time = 12:00:00  |  Memory = 64GB  |  CPUs = 8"
echo "  4. Launch → Connect to Jupyter"
echo "  5. Open notebooks/run.ipynb"
echo "  6. Kernel → Change Kernel → 'Python (svg-scaling)'"
