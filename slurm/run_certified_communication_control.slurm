#!/usr/bin/env bash
#SBATCH --job-name=osc-cert-control
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

DINO_MANIFEST="${OPENSEMCOM_DINO_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_dinov3_mixed_open_calibration.csv}"
SIGLIP_MANIFEST="${OPENSEMCOM_SIGLIP_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_siglip2_base.csv}"
OPENCLIP_MANIFEST="${OPENSEMCOM_OPENCLIP_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_openclip_dfn5b.csv}"
OUTPUT_PREFIX="${OPENSEMCOM_CERTIFIED_OUTPUT_PREFIX:-$OPENSEMCOM_PROJECT_ROOT/runs/comm_control_certified_${SLURM_JOB_ID}}"
CHECKPOINT_DIR="${OPENSEMCOM_CERTIFIED_CHECKPOINT_DIR:-$OPENSEMCOM_PROJECT_ROOT/models/checkpoints/communication_control_certified_${SLURM_JOB_ID}}"

for manifest in "$DINO_MANIFEST" "$SIGLIP_MANIFEST" "$OPENCLIP_MANIFEST"; do
  python -m opensemcom.cli.validate_manifest "$manifest"
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m opensemcom.cli.communication_control_suite \
  --feature-manifest "dino=$DINO_MANIFEST" \
  --feature-manifest "siglip2=$SIGLIP_MANIFEST" \
  --feature-manifest "openclip=$OPENCLIP_MANIFEST" \
  --output-prefix "$OUTPUT_PREFIX" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --seeds "${OPENSEMCOM_CERTIFIED_SEEDS:-0,1,2,3,4}" \
  --targets "${OPENSEMCOM_CERTIFIED_TARGETS:-0.05}" \
  --resource-budgets "${OPENSEMCOM_CERTIFIED_BUDGETS:-0.80}" \
  --eval-size "${OPENSEMCOM_CERTIFIED_EVAL_SIZE:-1024}" \
  --train-known-per-class "${OPENSEMCOM_CERTIFIED_TRAIN_KNOWN_PER_CLASS:-64}" \
  --train-open "${OPENSEMCOM_CERTIFIED_TRAIN_OPEN:-512}" \
  --cal-known-per-class "${OPENSEMCOM_CERTIFIED_CAL_KNOWN_PER_CLASS:-32}" \
  --cal-open "${OPENSEMCOM_CERTIFIED_CAL_OPEN:-2000}" \
  --certificate-fraction "${OPENSEMCOM_CERTIFIED_CERTIFICATE_FRACTION:-0.70}" \
  --certification-alpha "${OPENSEMCOM_CERTIFIED_ALPHA:-0.05}" \
  --certificate-family-size "${OPENSEMCOM_CERTIFIED_FAMILY_SIZE:-1}" \
  --primary-method "${OPENSEMCOM_CERTIFIED_PRIMARY_METHOD:-opensemcom_progressive}" \
  --selection-safety-factor "${OPENSEMCOM_CERTIFIED_SELECTION_SAFETY_FACTOR:-0.50}" \
  --full-open-severity "${OPENSEMCOM_CERTIFIED_SEVERITIES:-mild:0.25,medium:0.50,hard:0.75,extreme:0.91}"

export OUTPUT_PREFIX
python - <<'PY'
import csv
import os
from pathlib import Path

prefix = Path(os.environ["OUTPUT_PREFIX"])
summary_path = prefix.with_name(prefix.name + "_summary.csv")
policy_path = prefix.with_name(prefix.name + "_policies.csv")

with summary_path.open("r", encoding="utf-8", newline="") as handle:
    summary = list(csv.DictReader(handle))
with policy_path.open("r", encoding="utf-8", newline="") as handle:
    policies = list(csv.DictReader(handle))

if not summary or not policies:
    raise SystemExit("Certified suite produced empty summary or policy output.")

primary_rows = [
    row
    for row in summary
    if str(row.get("is_primary_policy", "")).lower() == "true"
]
if not primary_rows:
    raise SystemExit("Certified suite did not record the predeclared primary policy.")

violations = []
for row in summary:
    accepted = int(float(row.get("accepted", "0") or 0))
    valid = str(row.get("certificate_valid", "")).lower() == "true"
    upper = float(row.get("certificate_upper_bound", "1") or 1)
    target = float(row["target_openout"])
    if accepted > 0 and (not valid or upper > target):
        violations.append(
            (
                row.get("task"),
                row.get("method"),
                row.get("seed"),
                target,
                upper,
                accepted,
            )
        )

if violations:
    raise SystemExit(f"Accepted decisions without a valid target certificate: {violations[:10]}")

valid_rows = sum(
    str(row.get("certificate_valid", "")).lower() == "true"
    for row in summary
)
print(
    {
        "summary_rows": len(summary),
        "policy_rows": len(policies),
        "primary_rows": len(primary_rows),
        "certificate_valid_rows": valid_rows,
        "summary": str(summary_path),
    }
)
PY
