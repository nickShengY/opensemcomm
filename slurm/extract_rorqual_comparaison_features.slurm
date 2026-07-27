#!/usr/bin/env bash
#SBATCH --job-name=osc-rorqual-baseline-feats
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the OpenSemCom repository root}"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
source "$PROJECT_ROOT/scripts/env_scratch.sh"

if [[ -n "${OPENSEMCOM_COMPARAISON_TRANSFORMERS_CACHE:-}" ]]; then
  export TRANSFORMERS_CACHE="$OPENSEMCOM_COMPARAISON_TRANSFORMERS_CACHE"
fi

RAW_MANIFEST="${OPENSEMCOM_COMPARAISON_RAW_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_real.csv}"
FEATURE_ROOT="${OPENSEMCOM_COMPARAISON_FEATURE_ROOT:-$PROJECT_ROOT/data/features}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    METHOD="dino"
    MODEL_ID="facebook/dinov3-vitb16-pretrain-lvd1689m"
    FEATURE_SLUG="rorqual_dino_vitb16"
    BATCH_SIZE="${OPENSEMCOM_COMPARAISON_DINO_BATCH_SIZE:-128}"
    ;;
  1)
    METHOD="siglip"
    MODEL_ID="google/siglip-base-patch16-224"
    FEATURE_SLUG="rorqual_siglip_base"
    BATCH_SIZE="${OPENSEMCOM_COMPARAISON_SIGLIP_BATCH_SIZE:-128}"
    ;;
  2)
    METHOD="openclip"
    MODEL_ID="apple/DFN5B-CLIP-ViT-H-14-378"
    FEATURE_SLUG="rorqual_openclip_dfn5b"
    BATCH_SIZE="${OPENSEMCOM_COMPARAISON_OPENCLIP_BATCH_SIZE:-16}"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_MANIFEST="$PROJECT_ROOT/manifests/opensemcom_rorqual_${METHOD}.csv"
METADATA_PATH="$PROJECT_ROOT/manifests/opensemcom_rorqual_${METHOD}.comparison_extraction.json"

test -f "$RAW_MANIFEST"
echo "method=$METHOD model_id=$MODEL_ID raw_manifest=$RAW_MANIFEST"
sha256sum "$RAW_MANIFEST"

python -u -m opensemcom.cli.extract_features \
  --manifest "$RAW_MANIFEST" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --feature-root "$FEATURE_ROOT" \
  --model-id "$MODEL_ID" \
  --feature-slug "$FEATURE_SLUG" \
  --batch-size "$BATCH_SIZE" \
  --device cuda

python -m opensemcom.cli.validate_manifest "$OUTPUT_MANIFEST"

RAW_MANIFEST="$RAW_MANIFEST" OUTPUT_MANIFEST="$OUTPUT_MANIFEST" METADATA_PATH="$METADATA_PATH" \
METHOD="$METHOD" MODEL_ID="$MODEL_ID" FEATURE_SLUG="$FEATURE_SLUG" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

raw_manifest = Path(os.environ["RAW_MANIFEST"]).resolve()
metadata = {
    "raw_manifest": str(raw_manifest),
    "raw_manifest_sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
    "output_manifest": str(Path(os.environ["OUTPUT_MANIFEST"]).resolve()),
    "method": os.environ["METHOD"],
    "model_id": os.environ["MODEL_ID"],
    "feature_slug": os.environ["FEATURE_SLUG"],
}
Path(os.environ["METADATA_PATH"]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
