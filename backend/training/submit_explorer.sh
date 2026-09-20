#!/bin/bash
#SBATCH --job-name=stance_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=munusamy.s@northeastern.edu

# set -e turns a failed `conda activate` into an immediate, legible failure
# instead of silently falling through to the wrong Python (see below: this
# bit a real run on 2026-09-20 -- the script kept going after `conda activate`
# failed and trained against the cluster's shared, ABI-mismatched numpy).
# #SBATCH directives must stay contiguous above this line with no shell
# commands before them, or SLURM stops parsing them.
set -euo pipefail

# ── Environment setup ────────────────────────────────────────────────────────
# One-time, before the first submission: module load anaconda3/... (check
# available versions with `module avail python`) then
#   bash backend/training/setup_explorer_env.sh
# to create the claim-verify env this script activates below.

module load anaconda3/2024.06
# `module load` alone does not give a non-interactive shell conda's
# activate/deactivate shell function -- conda.sh must be sourced explicitly,
# or `conda activate` fails with "Run 'conda init' before 'conda activate'"
# and (without set -e) execution falls through to the base Python instead.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate claim-verify

# Verify GPU is visible. ${VAR:-unset} guards against `set -u` killing the
# job over an echo if the scheduler doesn't export this variable.
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# ── Paths ────────────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR

# Create output dirs
mkdir -p logs
mkdir -p checkpoints/stance_deberta

# Cache HuggingFace datasets/models to scratch (faster than home dir on HPC)
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface/transformers
mkdir -p $HF_HOME

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Starting training at $(date)"
python backend/training/train_stance.py
echo "Finished at $(date)"
