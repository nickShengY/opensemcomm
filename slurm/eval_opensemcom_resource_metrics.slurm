#!/usr/bin/env bash
#SBATCH --job-name=osc-resource-metrics
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MANIFEST="${OPENSEMCOM_RESOURCE_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv}"
SAMPLES="${OPENSEMCOM_RESOURCE_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_RESOURCE_CALIBRATION_SAMPLES:-640}"
USERS="${OPENSEMCOM_RESOURCE_USERS:-4}"
SEED="${OPENSEMCOM_RESOURCE_SEED:-0}"
RUNS_DIR="${OPENSEMCOM_RESOURCE_RUNS_DIR:-$OPENSEMCOM_PROJECT_ROOT/runs}"
RUN_TAG="resource_metrics_${SLURM_JOB_ID}"

python -m opensemcom.cli.validate_manifest "$MANIFEST"

for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
  CONFIG="$OPENSEMCOM_PROJECT_ROOT/configs/experiments/${REGIME}.yaml"
  python -m opensemcom.cli.run_config \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$RUNS_DIR" \
    --samples "$SAMPLES" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --users "$USERS" \
    --seed "$SEED" \
    --run-name "${REGIME}_${RUN_TAG}"
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$RUNS_DIR" \
  --pattern "*_${RUN_TAG}/metrics.json" \
  --output-csv "$RUNS_DIR/${RUN_TAG}.csv" \
  --output-json "$RUNS_DIR/${RUN_TAG}.json"
