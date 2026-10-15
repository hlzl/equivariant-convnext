#!/usr/bin/env bash
# Submit from the repository root: sbatch [resources] scripts/slurm.sh scripts/train.sh ...
#SBATCH --job-name=equivariant
#SBATCH --partition=gpu-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.log

set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec bash "$@"
