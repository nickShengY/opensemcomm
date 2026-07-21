#!/usr/bin/env bash
#SBATCH --job-name=osc-rorqual-clip-full-sionna
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env_scratch.sh"

MANIFEST="$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv"
SAMPLES="${OPENSEMCOM_FULL_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_FULL_CALIBRATION_SAMPLES:-640}"
USERS="${OPENSEMCOM_FULL_USERS:-4}"
SEED="${OPENSEMCOM_FULL_SEED:-0}"

python - <<'PY'
import sionna
import torch
print({"sionna": sionna.__version__, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
PY

python -m opensemcom.cli.validate_manifest "$MANIFEST"

for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
  CONFIG="$PROJECT_ROOT/configs/experiments_features_sionna/${REGIME}.yaml"
  python -m opensemcom.cli.run_config \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$PROJECT_ROOT/runs" \
    --samples "$SAMPLES" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --users "$USERS" \
    --seed "$SEED" \
    --run-name "${REGIME}_rorqual_clip_full_sionna_${SLURM_JOB_ID}" \
    --traces
 done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$PROJECT_ROOT/runs" \
  --pattern "*_rorqual_clip_full_sionna_${SLURM_JOB_ID}/metrics.json" \
  --output-csv "$PROJECT_ROOT/runs/rorqual_clip_full_sionna_results_${SLURM_JOB_ID}.csv" \
  --output-json "$PROJECT_ROOT/runs/rorqual_clip_full_sionna_results_${SLURM_JOB_ID}.json"
