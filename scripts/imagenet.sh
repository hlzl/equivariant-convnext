#!/usr/bin/env bash
set -euo pipefail

model=${1:?Usage: bash scripts/imagenet.sh convnext\|convnext_c4 [Hydra overrides]}
shift
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
rate=0.001
[[ "$model" != convnext_c4 ]] || rate=0.003
uv run --frozen torchrun --standalone \
  --nproc_per_node="${GPUS:-${SLURM_GPUS_ON_NODE:-1}}" \
  src/train.py "model=$model" data=imagenet1k train=imagenet \
  "train.lr=$rate" "$@"
