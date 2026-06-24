#!/usr/bin/env bash
#SBATCH --job-name=osc-eval
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%A_%a.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%A_%a.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

REGIMES=(closed-id channel-open source-open class-open task-open resource-open full-open)
REGIME="${REGIMES[$SLURM_ARRAY_TASK_ID]}"
CONFIG="$OPENSEMCOM_PROJECT_ROOT/configs/experiments/${REGIME}.yaml"
MANIFEST="$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv"

python -m opensemcom.cli.validate_manifest "$MANIFEST"
python -m opensemcom.cli.run_config \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --run-name "${REGIME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --traces
