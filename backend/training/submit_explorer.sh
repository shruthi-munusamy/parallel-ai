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

# ── Environment setup ────────────────────────────────────────────────────────
# One-time, before the first submission: module load anaconda3/... (check
# available versions with `module avail python`) then
#   bash backend/training/setup_explorer_env.sh
# to create the claim-verify env this script activates below.

module load anaconda3/2024.06
conda activate claim-verify

# Verify GPU is visible
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
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
