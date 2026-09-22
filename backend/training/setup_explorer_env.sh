#!/bin/bash
# setup_explorer_env.sh - One-time conda environment setup for the fine-tune job.
#
# Run this on a COMPUTE node, not the login node -- `conda create`'s dependency
# solver gets OOM-killed under the login node's per-process memory cap. Get an
# interactive allocation first (a plain CPU one is enough, no GPU needed just
# to build the environment), and run this inside tmux/screen so a dropped SSH
# connection doesn't kill a multi-minute pip install partway through:
#   tmux new -s setup
#   srun --partition=short --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash
#   module load anaconda3/2024.06
#   bash backend/training/setup_explorer_env.sh
#
# Safe to re-run if interrupted -- skips env creation if claim-verify already
# exists, and pip install is idempotent.
#
# submit_explorer.sh assumes this environment (name: claim-verify) already
# exists; it only activates it, it does not create it.

set -euo pipefail

ENV_NAME="claim-verify"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Environment '$ENV_NAME' already exists, skipping creation."
else
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda run -n "$ENV_NAME" pip install -r requirements.txt

# requirements.txt's plain `torch>=2.2` pulls whatever PyPI's default build is
# (confirmed on 2026-09-22: torch 2.14.0+cu130, needing a CUDA 13 driver).
# Explorer's GPU nodes only have driver-level CUDA 12.3, so that build silently
# leaves torch.cuda.is_available() == False -- training then "succeeds" on CPU
# and blows through the walltime limit. Force a CUDA-12.1 build instead, which
# is backward compatible with a 12.3 driver. Confirmed working via:
#   srun -p gpu-interactive --gres=gpu:1 -t 00:05:00 --pty bash
#   conda run -n claim-verify python -c "import torch; print(torch.cuda.is_available())"
conda run -n "$ENV_NAME" pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
# The reinstall above can pull a newer fsspec than datasets pins against.
conda run -n "$ENV_NAME" pip install "fsspec<=2026.6.0"

# protobuf is needed alongside sentencepiece for deberta-v3's tokenizer on
# this transformers/tokenizers version combo (confirmed missing on Explorer's
# anaconda3/2024.06 -- "SentencePieceExtractor requires the protobuf library").
conda run -n "$ENV_NAME" pip install datasets accelerate sentencepiece protobuf

echo "Environment '$ENV_NAME' ready. Verify on an actual GPU node (not this one) with:"
echo "  srun -p gpu-interactive --gres=gpu:1 -t 00:05:00 --pty bash"
echo "  module load anaconda3/2024.06"
echo "  conda run -n $ENV_NAME python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
