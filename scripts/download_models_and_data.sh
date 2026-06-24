#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_scratch.sh"

MANIFEST="${OPENSEMCOM_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real.csv}"

echo "Project root: $OPENSEMCOM_PROJECT_ROOT"
echo "Scratch root: $OPENSEMCOM_SCRATCH_ROOT"
echo "Manifest: $MANIFEST"

python -m opensemcom.cli.build_manifest \
  --output "$MANIFEST" \
  --project-root "$OPENSEMCOM_PROJECT_ROOT" \
  --max-per-source "${OPENSEMCOM_MAX_PER_SOURCE:-512}" \
  --max-calibration-per-class "${OPENSEMCOM_MAX_CALIBRATION_PER_CLASS:-64}" \
  --max-eval-per-class "${OPENSEMCOM_MAX_EVAL_PER_CLASS:-64}"

python -m opensemcom.cli.validate_manifest "$MANIFEST"

if [[ "${OPENSEMCOM_PREFETCH_MODELS:-0}" == "1" ]]; then
  python - <<'PY'
import os
from pathlib import Path

models = [
    "openai/clip-vit-base-patch32",
    "facebook/dinov2-small",
    "google/siglip-base-patch16-224",
]

try:
    from transformers import AutoModel, AutoProcessor
except Exception as exc:
    raise SystemExit(f"transformers is required for model prefetch: {exc}")

cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
for name in models:
    AutoModel.from_pretrained(name, cache_dir=cache_dir)
    try:
        AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
    except Exception:
        pass
    print(f"prefetched {name} into {cache_dir}")
PY
else
  echo "Skipping optional foundation-model prefetch; set OPENSEMCOM_PREFETCH_MODELS=1 to enable."
fi
