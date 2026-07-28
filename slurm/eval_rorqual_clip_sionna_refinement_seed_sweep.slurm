#!/usr/bin/env bash
#SBATCH --job-name=osc-sionna-refinement-sweep
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5%2
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env_scratch.sh"

SEEDS=(1 2 3)
VARIANTS=(progressive no_refinement)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / ${#VARIANTS[@]}))
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID % ${#VARIANTS[@]}))
SEED="${SEEDS[$SEED_INDEX]}"
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

MANIFEST="$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv"
if [[ "$VARIANT" == "progressive" ]]; then
  CONFIG_ROOT="$PROJECT_ROOT/configs/experiments_features_sionna_progressive"
else
  CONFIG_ROOT="$PROJECT_ROOT/configs/experiments_features_sionna_progressive_no_refinement"
fi
SAMPLES="${OPENSEMCOM_REPLICATION_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_REPLICATION_CALIBRATION_SAMPLES:-640}"
USERS="${OPENSEMCOM_REPLICATION_USERS:-4}"

python - <<'PY'
import sionna
import torch
print({"sionna": sionna.__version__, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
PY

python -m opensemcom.cli.validate_manifest "$MANIFEST"

for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
  CONFIG="$CONFIG_ROOT/${REGIME}.yaml"
  python -m opensemcom.cli.run_config \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$PROJECT_ROOT/runs" \
    --samples "$SAMPLES" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --users "$USERS" \
    --seed "$SEED" \
    --run-name "${REGIME}_rorqual_clip_sionna_${VARIANT}_seed${SEED}_${JOB_TAG}" \
    --traces
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$PROJECT_ROOT/runs" \
  --pattern "*_rorqual_clip_sionna_${VARIANT}_seed${SEED}_${JOB_TAG}/metrics.json" \
  --output-csv "$PROJECT_ROOT/runs/rorqual_clip_sionna_${VARIANT}_seed${SEED}_results_${JOB_TAG}.csv" \
  --output-json "$PROJECT_ROOT/runs/rorqual_clip_sionna_${VARIANT}_seed${SEED}_results_${JOB_TAG}.json"
