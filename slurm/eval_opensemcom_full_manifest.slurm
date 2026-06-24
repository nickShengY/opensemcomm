#!/usr/bin/env bash
#SBATCH --job-name=osc-full-manifest
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MANIFEST="$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv"
SAMPLES="${OPENSEMCOM_FULL_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_FULL_CALIBRATION_SAMPLES:-640}"
USERS="${OPENSEMCOM_FULL_USERS:-4}"
SEED="${OPENSEMCOM_FULL_SEED:-0}"

python -m opensemcom.cli.validate_manifest "$MANIFEST"

for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
  CONFIG="$OPENSEMCOM_PROJECT_ROOT/configs/experiments/${REGIME}.yaml"
  python -m opensemcom.cli.run_config \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
    --samples "$SAMPLES" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --users "$USERS" \
    --seed "$SEED" \
    --run-name "${REGIME}_full_${SLURM_JOB_ID}" \
    --traces
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --pattern "*_full_${SLURM_JOB_ID}/metrics.json" \
  --output-csv "$OPENSEMCOM_PROJECT_ROOT/runs/full_manifest_results_${SLURM_JOB_ID}.csv" \
  --output-json "$OPENSEMCOM_PROJECT_ROOT/runs/full_manifest_results_${SLURM_JOB_ID}.json"
