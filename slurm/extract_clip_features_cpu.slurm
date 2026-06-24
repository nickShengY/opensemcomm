#!/usr/bin/env bash
#SBATCH --job-name=osc-clip-feats-cpu
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MODEL_ID="${OPENSEMCOM_FEATURE_MODEL_ID:-openai/clip-vit-base-patch32}"
BATCH_SIZE="${OPENSEMCOM_FEATURE_BATCH_SIZE:-32}"

python -u -m opensemcom.cli.extract_features \
  --manifest "$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv" \
  --output-manifest "$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_clip.csv" \
  --feature-root "$OPENSEMCOM_PROJECT_ROOT/data/features" \
  --model-id "$MODEL_ID" \
  --batch-size "$BATCH_SIZE" \
  --device cpu

python -m opensemcom.cli.validate_manifest "$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_clip.csv"
