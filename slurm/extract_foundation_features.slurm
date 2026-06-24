#!/usr/bin/env bash
#SBATCH --job-name=osc-found-feats
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MODEL_ID="${OPENSEMCOM_FEATURE_MODEL_ID:?Set OPENSEMCOM_FEATURE_MODEL_ID}"
FEATURE_SLUG="${OPENSEMCOM_FEATURE_SLUG:-}"
BATCH_SIZE="${OPENSEMCOM_FEATURE_BATCH_SIZE:-128}"
DEVICE="${OPENSEMCOM_FEATURE_DEVICE:-auto}"
SLUG="$(python - <<'PY'
import os
print((os.environ.get("OPENSEMCOM_FEATURE_SLUG") or os.environ["OPENSEMCOM_FEATURE_MODEL_ID"]).replace("/", "__").replace(":", "_"))
PY
)"
OUTPUT_MANIFEST="${OPENSEMCOM_FEATURE_OUTPUT_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_${SLUG}.csv}"

EXTRA_ARGS=()
if [[ -n "$FEATURE_SLUG" ]]; then
  EXTRA_ARGS+=(--feature-slug "$FEATURE_SLUG")
fi

python -u -m opensemcom.cli.extract_features \
  --manifest "$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --feature-root "$OPENSEMCOM_PROJECT_ROOT/data/features" \
  --model-id "$MODEL_ID" \
  "${EXTRA_ARGS[@]}" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE"

python -m opensemcom.cli.validate_manifest "$OUTPUT_MANIFEST"
