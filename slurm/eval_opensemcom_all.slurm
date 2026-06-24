#!/usr/bin/env bash
#SBATCH --job-name=osc-eval-all
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MANIFEST="$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv"
python -m opensemcom.cli.validate_manifest "$MANIFEST"

for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
  CONFIG="$OPENSEMCOM_PROJECT_ROOT/configs/experiments/${REGIME}.yaml"
  python -m opensemcom.cli.run_config \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
    --run-name "${REGIME}_${SLURM_JOB_ID}" \
    --traces
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --output-csv "$OPENSEMCOM_PROJECT_ROOT/runs/aggregate_results.csv" \
  --output-json "$OPENSEMCOM_PROJECT_ROOT/runs/aggregate_results.json"
