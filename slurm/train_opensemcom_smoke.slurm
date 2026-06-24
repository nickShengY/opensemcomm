#!/usr/bin/env bash
#SBATCH --job-name=osc-train-smoke
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MANIFEST="$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv"
CONFIG="$OPENSEMCOM_PROJECT_ROOT/configs/experiments/closed-id.yaml"

python -m opensemcom.cli.validate_manifest "$MANIFEST"
python -m opensemcom.cli.run_config \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --samples 16 \
  --calibration-samples 16 \
  --users 2 \
  --run-name "smoke_train_${SLURM_JOB_ID}"
