#!/usr/bin/env bash
#SBATCH --job-name=osc-rorqual-imagebind-feats
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
FEATURE_ROOT="${OPENSEMCOM_COMPARAISON_FEATURE_ROOT:-$PROJECT_ROOT/data/features}"
OUTPUT_MANIFEST="${OPENSEMCOM_IMAGEBIND_MANIFEST:-$PROJECT_ROOT/manifests/opensemcom_rorqual_imagebind.csv}"
IMAGEBIND_CACHE="${OPENSEMCOM_IMAGEBIND_CACHE:-$PROJECT_ROOT/cache/imagebind}"
BATCH_SIZE="${OPENSEMCOM_IMAGEBIND_BATCH_SIZE:-16}"
IMAGEBIND_REVISION="${OPENSEMCOM_IMAGEBIND_REVISION:-53680b02d7e37b19b124fa37bae4b6c98c38f5be}"
python -c "import imagebind" || { echo 'ImageBind is not installed. Run scripts/install_imagebind.sh first.' >&2; exit 2; }
test -f "$RAW_MANIFEST"
git rev-parse HEAD
python -u -m opensemcom.cli.extract_features \
  --manifest "$RAW_MANIFEST" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --feature-root "$FEATURE_ROOT" \
  --model-id facebookresearch/imagebind-huge \
  --feature-slug rorqual_imagebind_huge \
  --extractor imagebind \
  --imagebind-cache-dir "$IMAGEBIND_CACHE" \
  --imagebind-source-revision "$IMAGEBIND_REVISION" \
  --batch-size "$BATCH_SIZE" \
  --device cuda
python -m opensemcom.cli.validate_manifest "$OUTPUT_MANIFEST"