#!/usr/bin/env bash
#SBATCH --job-name=osc-rorqual-model-prefetch
#SBATCH --time=04:00:00
#SBATCH --array=0-1
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

# This job intentionally permits network access. The extractor itself remains
# offline-only so benchmark runs cannot silently change a model revision.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    METHOD="dino"
    MODEL_ID="facebook/dinov3-vitb16-pretrain-lvd1689m"
    REVISION="${OPENSEMCOM_COMPARAISON_DINO_REVISION:-5931719e67bbdb9737e363e781fb0c67687896bc}"
    ;;
  1)
    METHOD="openclip"
    MODEL_ID="apple/DFN5B-CLIP-ViT-H-14-378"
    REVISION="${OPENSEMCOM_COMPARAISON_OPENCLIP_REVISION:-}"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

METADATA_PATH="$TRANSFORMERS_CACHE/opensemcom_rorqual_${METHOD}_model.json"
echo "method=$METHOD model_id=$MODEL_ID revision=${REVISION:-default} cache=$TRANSFORMERS_CACHE"

METHOD="$METHOD" MODEL_ID="$MODEL_ID" REVISION="$REVISION" METADATA_PATH="$METADATA_PATH" python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

model_id = os.environ["MODEL_ID"]
revision = os.environ["REVISION"] or None
snapshot = Path(
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        local_files_only=False,
    )
).resolve()

# Confirm that both the model configuration and preprocessing files can be
# opened from the downloaded, immutable snapshot.
AutoConfig.from_pretrained(snapshot, local_files_only=True)
try:
    AutoProcessor.from_pretrained(snapshot, local_files_only=True)
except Exception:
    AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)

metadata = {
    "method": os.environ["METHOD"],
    "model_id": model_id,
    "requested_revision": revision,
    "resolved_snapshot": str(snapshot),
    "resolved_revision": snapshot.name,
    "transformers_cache": os.environ["TRANSFORMERS_CACHE"],
}
metadata_path = Path(os.environ["METADATA_PATH"])
metadata_path.parent.mkdir(parents=True, exist_ok=True)
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
