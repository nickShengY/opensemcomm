#!/usr/bin/env bash
# Exploratory threshold-policy diagnostic only. This does not replace the
# fixed six-regime benchmark configuration or select a final published policy.
#SBATCH --job-name=osc-threshold-diagnostic
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=01:00:00
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
# Print each stage's actual calibrated thresholds. A mixed-open cap may make
# a changed YAML quantile ineffective, so this is essential evidence.
export OPENSEMCOM_CALIBRATION_DEBUG="${OPENSEMCOM_CALIBRATION_DEBUG:-1}"
source "$PROJECT_ROOT/scripts/env_scratch.sh"

RAW_MANIFEST="${OPENSEMCOM_MIXED_RAW_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_real.csv}"
OPENSEMCOM_MANIFEST="${OPENSEMCOM_MIXED_OPENSEMCOM_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv}"
: "${OPENSEMCOM_MIXED_SIGLIP_MANIFEST:?Set the aligned mixed SigLIP feature manifest before submitting.}"
: "${OPENSEMCOM_MIXED_OPENCLIP_MANIFEST:?Set the aligned mixed OpenCLIP feature manifest before submitting.}"

CONFIG="$PROJECT_ROOT/configs/experiments_features_sionna_progressive_threshold_diagnostic/closed-id.yaml"
OUTPUT_ROOT="${OPENSEMCOM_THRESHOLD_DIAGNOSTIC_OUTPUT_ROOT:-$PROJECT_ROOT/runs/threshold_diagnostic_q50_q60_${SLURM_JOB_ID}}"
SEED="${OPENSEMCOM_MIXED_SEED:-0}"
PAYLOAD_BLOCKS="${OPENSEMCOM_MIXED_PAYLOAD_BLOCKS:-1}"

mkdir -p "$OUTPUT_ROOT"
git rev-parse HEAD

python -m opensemcom.cli.run_comparaison \
  --config "$CONFIG" \
  --regime closed-id \
  --raw-manifest "$RAW_MANIFEST" \
  --opensemcom-manifest "$OPENSEMCOM_MANIFEST" \
  --siglip-manifest "$OPENSEMCOM_MIXED_SIGLIP_MANIFEST" \
  --openclip-manifest "$OPENSEMCOM_MIXED_OPENCLIP_MANIFEST" \
  --methods opensemcom,siglip,openclip \
  --output-dir "$OUTPUT_ROOT" \
  --seed "$SEED" \
  --payload-blocks "$PAYLOAD_BLOCKS" \
  --expected-calibration-known 384 \
  --expected-calibration-open 256 \
  --write-traces
