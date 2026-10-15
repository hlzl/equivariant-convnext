#!/usr/bin/env bash
# DATASETS and SEEDS can select any part of the nine-dataset panel.
set -euo pipefail

mode=${1:?Usage: bash scripts/train.sh scratch\|finetune MODEL [Hydra overrides]}
model=${2:?Select a model config}
shift 2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

for seed in ${SEEDS:-0 1 2}; do
  for dataset in ${DATASETS:-cifar100 pcam minc2500 caltech256 food101 officehome fgvc_aircraft sun397 oxford_pets}; do
    crop=()
    if [[ "$mode" == finetune && "$dataset" != cifar100 && "$dataset" != pcam ]]; then
      crop=('data.crop_scale=[0.5,1.0]')
    fi
    uv run --frozen src/train.py "train=$mode" "model=$model" \
      "data=$dataset" "seed=$seed" "${crop[@]}" "$@"
  done
done
