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
# Check available modules with: module avail python
# Adjust these lines to match what Explorer has

module load anaconda3/2023.07  # or whatever version Explorer has
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
