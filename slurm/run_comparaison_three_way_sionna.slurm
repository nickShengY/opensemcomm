#!/usr/bin/env bash
#SBATCH --job-name=osc-comparaison-3way
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

RAW_MANIFEST="${OPENSEMCOM_COMPARAISON_RAW_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_real.csv}"
OPENSEMCOM_MANIFEST="${OPENSEMCOM_COMPARAISON_OPENSEMCOM_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv}"
: "${OPENSEMCOM_COMPARAISON_SIGLIP_MANIFEST:?Set the aligned SigLIP feature manifest before submitting.}"
: "${OPENSEMCOM_COMPARAISON_OPENCLIP_MANIFEST:?Set the aligned OpenCLIP feature manifest before submitting.}"

CONFIG_ROOT="${OPENSEMCOM_COMPARAISON_CONFIG_ROOT:-$PROJECT_ROOT/configs/experiments_features_sionna_progressive}"
OUTPUT_ROOT="${OPENSEMCOM_COMPARAISON_OUTPUT_ROOT:-$PROJECT_ROOT/runs/comparaison_three_way_${SLURM_JOB_ID}}"
SEED="${OPENSEMCOM_COMPARAISON_SEED:-0}"
PAYLOAD_BLOCKS="${OPENSEMCOM_COMPARAISON_PAYLOAD_BLOCKS:-1}"
METHODS="opensemcom,siglip,openclip"
REGIMES=(closed-id channel-open source-open class-open task-open resource-open full-open)

mkdir -p "$OUTPUT_ROOT"
git rev-parse HEAD

for REGIME in "${REGIMES[@]}"; do
  CONFIG="$CONFIG_ROOT/${REGIME}.yaml"
  python -m opensemcom.cli.run_comparaison \
    --config "$CONFIG" \
    --regime "$REGIME" \
    --raw-manifest "$RAW_MANIFEST" \
    --opensemcom-manifest "$OPENSEMCOM_MANIFEST" \
    --siglip-manifest "$OPENSEMCOM_COMPARAISON_SIGLIP_MANIFEST" \
    --openclip-manifest "$OPENSEMCOM_COMPARAISON_OPENCLIP_MANIFEST" \
    --output-dir "$OUTPUT_ROOT/$REGIME" \
    --methods "$METHODS" \
    --seed "$SEED" \
    --payload-blocks "$PAYLOAD_BLOCKS" \
    --write-traces
done
