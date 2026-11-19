#!/bin/bash
# setup_explorer_env.sh - One-time conda environment setup for the fine-tune job.
#
# Run on a compute node (the login node's memory cap kills conda's solver),
# inside tmux so a dropped SSH connection doesn't interrupt the install:
#   tmux new -s setup
#   srun --partition=short --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash
#   module load anaconda3/2024.06
#   bash backend/training/setup_explorer_env.sh
#
# Safe to re-run. submit_explorer.sh activates the env this creates.

set -euo pipefail

ENV_NAME="claim-verify"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Environment '$ENV_NAME' already exists, skipping creation."
else
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda run -n "$ENV_NAME" pip install -r requirements.txt

# PyPI's default torch build needs a newer CUDA driver than Explorer's (12.3)
# and silently falls back to CPU. The CUDA 12.1 build works with that driver.
conda run -n "$ENV_NAME" pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
# The reinstall above can pull a newer fsspec than datasets pins against.
conda run -n "$ENV_NAME" pip install "fsspec<=2026.6.0"

# deberta-v3's tokenizer needs both sentencepiece and protobuf.
conda run -n "$ENV_NAME" pip install datasets accelerate sentencepiece protobuf

echo "Environment '$ENV_NAME' ready. Verify on an actual GPU node (not this one) with:"
echo "  srun -p gpu-interactive --gres=gpu:1 -t 00:05:00 --pty bash"
echo "  module load anaconda3/2024.06"
echo "  conda run -n $ENV_NAME python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
