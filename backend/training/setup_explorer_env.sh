#!/bin/bash
# setup_explorer_env.sh - One-time conda environment setup for the fine-tune job.
#
# Run this once on a login node before the first `sbatch submit_explorer.sh`:
#   module load anaconda3/2023.07   # match whatever `module avail python` shows
#   bash backend/training/setup_explorer_env.sh
#
# submit_explorer.sh assumes this environment (name: claim-verify) already
# exists; it only activates it, it does not create it.

set -euo pipefail

ENV_NAME="claim-verify"

conda create -y -n "$ENV_NAME" python=3.11
conda run -n "$ENV_NAME" pip install -r requirements.txt
conda run -n "$ENV_NAME" pip install datasets accelerate sentencepiece

echo "Environment '$ENV_NAME' ready. Verify GPU + CUDA torch build with:"
echo "  conda run -n $ENV_NAME python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
