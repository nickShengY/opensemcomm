#!/usr/bin/env bash
#SBATCH --mem=16G
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=osc-clip-feats
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

export OPENSEMCOM_PROJECT_ROOT="/home/nickyun/links/scratch/scratch/ahmed/opensemcomm"
export OPENSEMCOM_SCRATCH_ROOT="$OPENSEMCOM_PROJECT_ROOT"
source "$OPENSEMCOM_PROJECT_ROOT/scripts/env_scratch.sh"

MODEL_ID="${OPENSEMCOM_FEATURE_MODEL_ID:-openai/clip-vit-base-patch32}"
BATCH_SIZE="${OPENSEMCOM_FEATURE_BATCH_SIZE:-64}"
DEVICE="${OPENSEMCOM_FEATURE_DEVICE:-auto}"

python -u -m opensemcom.cli.extract_features \
  --manifest "$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv" \
  --output-manifest "$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_clip.csv" \
  --feature-root "$OPENSEMCOM_PROJECT_ROOT/data/features" \
  --model-id "$MODEL_ID" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE"

python -m opensemcom.cli.validate_manifest "$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_clip.csv"
