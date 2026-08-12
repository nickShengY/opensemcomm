#!/usr/bin/env bash
#SBATCH --job-name=osc-refinement-score-diagnostic
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Post-hoc score diagnostics only: all configs retain their existing calibration
# and controller settings.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env_scratch.sh"

MANIFEST="$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv"
CONFIG_ROOT="$PROJECT_ROOT/configs/experiments_features_sionna_progressive"
RUN_ROOT="$PROJECT_ROOT/runs/refinement_score_diagnostic_${SLURM_JOB_ID}"
SAMPLES="${OPENSEMCOM_DIAGNOSTIC_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_DIAGNOSTIC_CALIBRATION_SAMPLES:-640}"
USERS="${OPENSEMCOM_DIAGNOSTIC_USERS:-4}"
SEED="${OPENSEMCOM_DIAGNOSTIC_SEED:-0}"
REGIMES=(closed-id channel-open source-open class-open task-open resource-open full-open)

python -m opensemcom.cli.validate_manifest "$MANIFEST"

for REGIME in "${REGIMES[@]}"; do
  python -m opensemcom.cli.run_config \
    --config "$CONFIG_ROOT/${REGIME}.yaml" \
    --manifest "$MANIFEST" \
    --output-dir "$RUN_ROOT" \
    --samples "$SAMPLES" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --users "$USERS" \
    --seed "$SEED" \
    --run-name "$REGIME" \
    --traces
done

python -m opensemcom.cli.analyze_refinement_transitions \
  --traces "$RUN_ROOT"/*/traces.json \
  --output-dir "$RUN_ROOT/refinement_score_analysis"
