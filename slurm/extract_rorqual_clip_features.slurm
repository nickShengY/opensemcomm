#!/usr/bin/env bash
#SBATCH --job-name=osc-rorqual-clip-feats
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/nickyun/links/scratch/scratch/ahmed/opensemcomm/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/scratch/ahmed/opensemcomm/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}" && pwd)"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env_scratch.sh"

MODEL_ID="${OPENSEMCOM_FEATURE_MODEL_ID:-openai/clip-vit-base-patch32}"
BATCH_SIZE="${OPENSEMCOM_FEATURE_BATCH_SIZE:-16}"
DEVICE="${OPENSEMCOM_FEATURE_DEVICE:-auto}"
MANIFEST="$PROJECT_ROOT/manifests/opensemcom_rorqual_real.csv"
OUTPUT_MANIFEST="$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv"

python -u -m opensemcom.cli.extract_features \
  --manifest "$MANIFEST" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --feature-root "$PROJECT_ROOT/data/features" \
  --model-id "$MODEL_ID" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE"

python -m opensemcom.cli.validate_manifest "$OUTPUT_MANIFEST"