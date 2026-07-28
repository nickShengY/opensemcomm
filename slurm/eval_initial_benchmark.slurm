#!/usr/bin/env bash
#SBATCH --job-name=osc-initial-bench
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(pwd)"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env_scratch.sh"

MANIFEST="$PROJECT_ROOT/manifests/opensemcom_real.csv"
RUN_TAG="initial_${SLURM_JOB_ID}"

python -m opensemcom.cli.validate_manifest "$MANIFEST" --allow-non-scratch

for REGIME in closed-id task-open full-open; do
  CONFIG="$PROJECT_ROOT/configs/experiments/${REGIME}.yaml"
  python -m opensemcom.cli.run_config \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$PROJECT_ROOT/runs" \
    --samples 512 \
    --calibration-samples 256 \
    --users 4 \
    --run-name "${REGIME}_${RUN_TAG}" \
    --traces
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$PROJECT_ROOT/runs" \
  --pattern "*_${RUN_TAG}/metrics.json" \
  --output-csv "$PROJECT_ROOT/runs/${RUN_TAG}.csv" \
  --output-json "$PROJECT_ROOT/runs/${RUN_TAG}.json"
