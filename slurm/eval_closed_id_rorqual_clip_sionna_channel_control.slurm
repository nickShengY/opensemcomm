#!/usr/bin/env bash
#SBATCH --job-name=osc-sionna-channel-control
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CONFIG_STEM="${1:?Usage: sbatch $0 <closed-id-awgn-18db|closed-id-awgn-17db|closed-id-awgn-12db|closed-id-rayleigh-12db|closed-id-rician-12db>}"
export CONFIG_STEM
PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}" && pwd)"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_CALIBRATION_DEBUG=1
source "$PROJECT_ROOT/scripts/env_scratch.sh"

MANIFEST="$PROJECT_ROOT/manifests/opensemcom_rorqual_clip.csv"
CONFIG="$PROJECT_ROOT/configs/experiments_features_sionna/${CONFIG_STEM}.yaml"
SAMPLES="${OPENSEMCOM_DEBUG_SAMPLES:-384}"
CALIBRATION_SAMPLES="${OPENSEMCOM_DEBUG_CALIBRATION_SAMPLES:-256}"
USERS="${OPENSEMCOM_DEBUG_USERS:-4}"
SEED="${OPENSEMCOM_DEBUG_SEED:-0}"
RUN_NAME="${CONFIG_STEM}_${SLURM_JOB_ID}"

case "$CONFIG_STEM" in
  closed-id-awgn-18db|closed-id-awgn-17db|closed-id-awgn-12db|closed-id-rayleigh-12db|closed-id-rician-12db) ;;
  *) echo "Unsupported channel-control config: $CONFIG_STEM" >&2; exit 2 ;;
esac

python - <<'PY'
import sionna
import torch
print({"sionna": sionna.__version__, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
PY

python -m opensemcom.cli.validate_manifest "$MANIFEST"
python -m opensemcom.cli.run_config \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output-dir "$PROJECT_ROOT/runs" \
  --samples "$SAMPLES" \
  --calibration-samples "$CALIBRATION_SAMPLES" \
  --users "$USERS" \
  --seed "$SEED" \
  --run-name "$RUN_NAME" \
  --traces

python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

run_name = f"{os.environ['CONFIG_STEM']}_{os.environ['SLURM_JOB_ID']}"
path = Path(os.environ["OPENSEMCOM_PROJECT_ROOT"]) / "runs" / run_name / "traces.json"
traces = json.loads(path.read_text(encoding="utf-8"))
scores = np.array([trace["risk_score"] for trace in traces], dtype=float)
confidences = np.array([trace["confidence"] for trace in traces], dtype=float)
quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
phy_keys = (
    "phy_payload_bit_error_rate",
    "phy_ldpc_block_error_rate",
    "phy_payload_mse",
    "phy_quantization_mse",
)
print(
    "EVAL_DEBUG",
    {
        "n": len(scores),
        "risk_q": np.quantile(scores, quantiles).tolist(),
        "confidence_q": np.quantile(confidences, quantiles).tolist(),
        "phy_q": {
            key: np.quantile([trace["features"].get(key, 0.0) for trace in traces], quantiles).tolist()
            for key in phy_keys
        },
        "prediction_set_sizes": dict(sorted(Counter(trace["prediction_set_size"] for trace in traces).items())),
        "decisions": dict(sorted(Counter(trace["decision"] for trace in traces).items())),
    },
    flush=True,
)
PY
