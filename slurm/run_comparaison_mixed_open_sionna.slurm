#!/usr/bin/env bash
#SBATCH --job-name=osc-comparison-mixed
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the OpenSemCom repository root}"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
source "$PROJECT_ROOT/scripts/env_scratch.sh"

# This is an image-classification benchmark with text declared as task-open.
# DINO is intentionally excluded because its existing feature manifest has no text rows.
RAW_MANIFEST="${OPENSEMCOM_MIXED_RAW_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_real.csv}"
OPENSEMCOM_MANIFEST="${OPENSEMCOM_MIXED_OPENSEMCOM_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv}"
: "${OPENSEMCOM_MIXED_SIGLIP_MANIFEST:?Set the aligned mixed SigLIP2 feature manifest before submitting.}"
: "${OPENSEMCOM_MIXED_OPENCLIP_MANIFEST:?Set the aligned mixed OpenCLIP feature manifest before submitting.}"

CONFIG_ROOT="${OPENSEMCOM_MIXED_CONFIG_ROOT:-$PROJECT_ROOT/configs/experiments_features_sionna_progressive}"
OUTPUT_ROOT="${OPENSEMCOM_MIXED_OUTPUT_ROOT:-$PROJECT_ROOT/runs/comparaison_mixed_open_${SLURM_JOB_ID}}"
SEED="${OPENSEMCOM_MIXED_SEED:-0}"
PAYLOAD_BLOCKS="${OPENSEMCOM_MIXED_PAYLOAD_BLOCKS:-1}"
REGIMES=(closed-id channel-open source-open class-open task-open resource-open full-open)

mkdir -p "$OUTPUT_ROOT"
git rev-parse HEAD

for REGIME in "${REGIMES[@]}"; do
  case "$REGIME" in
    closed-id) CONFIG="$CONFIG_ROOT/closed-id.yaml" ;;
    channel-open) CONFIG="$CONFIG_ROOT/channel-open.yaml" ;;
    source-open) CONFIG="$CONFIG_ROOT/source-open.yaml" ;;
    class-open) CONFIG="$CONFIG_ROOT/class-open.yaml" ;;
    task-open) CONFIG="$CONFIG_ROOT/task-open.yaml" ;;
    resource-open) CONFIG="$CONFIG_ROOT/resource-open.yaml" ;;
    full-open) CONFIG="$CONFIG_ROOT/full-open.yaml" ;;
  esac

  python -m opensemcom.cli.run_comparaison \
    --config "$CONFIG" \
    --regime "$REGIME" \
    --raw-manifest "$RAW_MANIFEST" \
    --opensemcom-manifest "$OPENSEMCOM_MANIFEST" \
    --siglip-manifest "$OPENSEMCOM_MIXED_SIGLIP_MANIFEST" \
    --openclip-manifest "$OPENSEMCOM_MIXED_OPENCLIP_MANIFEST" \
    --methods opensemcom,siglip,openclip \
    --output-dir "$OUTPUT_ROOT/$REGIME" \
    --seed "$SEED" \
    --payload-blocks "$PAYLOAD_BLOCKS" \
    --write-traces
done
