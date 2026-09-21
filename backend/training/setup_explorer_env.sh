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
# protobuf is needed alongside sentencepiece for deberta-v3's tokenizer on
# this transformers/tokenizers version combo (confirmed missing on Explorer's
# anaconda3/2024.06 -- "SentencePieceExtractor requires the protobuf library").
conda run -n "$ENV_NAME" pip install datasets accelerate sentencepiece protobuf

echo "Environment '$ENV_NAME' ready. Verify GPU + CUDA torch build with:"
echo "  conda run -n $ENV_NAME python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
