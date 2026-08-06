#!/usr/bin/env bash
# Mixed calibration: known CIFAR images plus labelled task-open text.
# Resource-open is excluded until it applies a verified perturbation.
#SBATCH --job-name=osc-comparison-imagebind
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

RAW_MANIFEST="${OPENSEMCOM_MIXED_RAW_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_real.csv}"
OPENSEMCOM_MANIFEST="${OPENSEMCOM_MIXED_OPENSEMCOM_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv}"
SIGLIP_MANIFEST="${OPENSEMCOM_MIXED_SIGLIP_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_siglip.csv}"
OPENCLIP_MANIFEST="${OPENSEMCOM_MIXED_OPENCLIP_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_openclip.csv}"
IMAGEBIND_MANIFEST="${OPENSEMCOM_MIXED_IMAGEBIND_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_imagebind.csv}"
CONFIG_ROOT="${OPENSEMCOM_MIXED_CONFIG_ROOT:-$PROJECT_ROOT/configs/experiments_features_sionna_progressive}"
OUTPUT_ROOT="${OPENSEMCOM_MIXED_OUTPUT_ROOT:-$PROJECT_ROOT/runs/comparaison_mixed_open_imagebind_${SLURM_JOB_ID}}"
SEED="${OPENSEMCOM_MIXED_SEED:-0}"
PAYLOAD_BLOCKS="${OPENSEMCOM_MIXED_PAYLOAD_BLOCKS:-1}"
EXPECTED_CALIBRATION_KNOWN="${OPENSEMCOM_MIXED_EXPECTED_CALIBRATION_KNOWN:-384}"
EXPECTED_CALIBRATION_OPEN="${OPENSEMCOM_MIXED_EXPECTED_CALIBRATION_OPEN:-256}"
REGIME_LIST="${OPENSEMCOM_MIXED_REGIMES:-closed-id channel-open source-open class-open task-open full-open}"
read -r -a REGIMES <<< "$REGIME_LIST"

for manifest in "$RAW_MANIFEST" "$OPENSEMCOM_MANIFEST" "$SIGLIP_MANIFEST" "$OPENCLIP_MANIFEST" "$IMAGEBIND_MANIFEST"; do
  test -f "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }
done
mkdir -p "$OUTPUT_ROOT"
git rev-parse HEAD

for REGIME in "${REGIMES[@]}"; do
  case "$REGIME" in
    closed-id|channel-open|source-open|class-open|task-open|full-open) ;;
    *) echo "Unsupported comparison regime: $REGIME" >&2; exit 2 ;;
  esac
  python -m opensemcom.cli.run_comparaison \
    --config "$CONFIG_ROOT/$REGIME.yaml" \
    --regime "$REGIME" \
    --raw-manifest "$RAW_MANIFEST" \
    --opensemcom-manifest "$OPENSEMCOM_MANIFEST" \
    --siglip-manifest "$SIGLIP_MANIFEST" \
    --openclip-manifest "$OPENCLIP_MANIFEST" \
    --imagebind-manifest "$IMAGEBIND_MANIFEST" \
    --methods opensemcom,siglip,openclip,imagebind \
    --output-dir "$OUTPUT_ROOT/$REGIME" \
    --seed "$SEED" \
    --payload-blocks "$PAYLOAD_BLOCKS" \
    --expected-calibration-known "$EXPECTED_CALIBRATION_KNOWN" \
    --expected-calibration-open "$EXPECTED_CALIBRATION_OPEN" \
    --write-traces
done