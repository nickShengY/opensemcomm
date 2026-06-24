#!/usr/bin/env bash
#SBATCH --job-name=osc-aggregate
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --output-csv "$OPENSEMCOM_PROJECT_ROOT/runs/aggregate_results.csv" \
  --output-json "$OPENSEMCOM_PROJECT_ROOT/runs/aggregate_results.json"
