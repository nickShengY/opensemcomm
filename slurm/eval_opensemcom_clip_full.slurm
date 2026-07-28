#!/usr/bin/env bash
#SBATCH --job-name=osc-clip-full
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/lustre10/scratch/nickyun/scratch/ahmed/opensemcomm/logs/%x-%j.out
#SBATCH --error=/lustre10/scratch/nickyun/scratch/ahmed/opensemcomm/logs/%x-%j.err

set -euo pipefail

export OPENSEMCOM_PROJECT_ROOT="/lustre10/scratch/nickyun/scratch/ahmed/opensemcomm"
export OPENSEMCOM_SCRATCH_ROOT="$OPENSEMCOM_PROJECT_ROOT"
source "$OPENSEMCOM_PROJECT_ROOT/scripts/env_scratch.sh"

MANIFEST="$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv"
SAMPLES="${OPENSEMCOM_FULL_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_FULL_CALIBRATION_SAMPLES:-640}"
USERS="${OPENSEMCOM_FULL_USERS:-4}"
SEED="${OPENSEMCOM_FULL_SEED:-0}"

python -m opensemcom.cli.validate_manifest "$MANIFEST"

for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
  CONFIG="$OPENSEMCOM_PROJECT_ROOT/configs/experiments_features/${REGIME}.yaml"
  python -m opensemcom.cli.run_config \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
    --samples "$SAMPLES" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --users "$USERS" \
    --seed "$SEED" \
    --run-name "${REGIME}_clip_full_${SLURM_JOB_ID}" \
    --traces
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --pattern "*_clip_full_${SLURM_JOB_ID}/metrics.json" \
  --output-csv "$OPENSEMCOM_PROJECT_ROOT/runs/clip_full_results_${SLURM_JOB_ID}.csv" \
  --output-json "$OPENSEMCOM_PROJECT_ROOT/runs/clip_full_results_${SLURM_JOB_ID}.json"
