#!/usr/bin/env bash
#SBATCH --job-name=osc-found-full
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MANIFEST="${OPENSEMCOM_FEATURE_MANIFEST:?Set OPENSEMCOM_FEATURE_MANIFEST}"
RUN_TAG="${OPENSEMCOM_RUN_TAG:-foundation}"
SAMPLES="${OPENSEMCOM_FULL_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_FULL_CALIBRATION_SAMPLES:-640}"
USERS="${OPENSEMCOM_FULL_USERS:-4}"
SEEDS="${OPENSEMCOM_FULL_SEEDS:-0 1 2}"
CONFIG_ROOT="$OPENSEMCOM_PROJECT_ROOT/tmp/configs_${RUN_TAG}_${SLURM_JOB_ID}"

python -m opensemcom.cli.validate_manifest "$MANIFEST"
export MANIFEST

FEATURE_DIM="$(python - <<'PY'
import csv
import os
from pathlib import Path
import numpy as np

manifest = Path(os.environ["MANIFEST"]).expanduser().resolve()
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        path = Path(row["source_path"]).expanduser().resolve()
        print(np.load(path).reshape(-1).shape[0])
        break
PY
)"
mkdir -p "$CONFIG_ROOT"
export FEATURE_DIM MANIFEST CONFIG_ROOT

python - <<'PY'
import os
from pathlib import Path
import yaml

manifest = str(Path(os.environ["MANIFEST"]).expanduser().resolve())
config_root = Path(os.environ["CONFIG_ROOT"])
feature_dim = int(os.environ["FEATURE_DIM"])
regimes = ("closed-id", "channel-open", "source-open", "class-open", "task-open", "resource-open", "full-open")
base_model = {
    "input_dim": feature_dim,
    "latent_dim": feature_dim,
    "projection": "identity",
    "classifier": "logistic",
    "num_known_classes": 6,
    "num_unknown_classes": 2,
    "tasks": ["classification", "detection", "segmentation", "driving-detection", "beam-prediction", "text-classification"],
    "train_tasks": ["classification"],
    "train_domains": ["cifar10"],
}
detector = {
    "prediction": 0.15,
    "prototype": 0.1,
    "energy": 0.1,
    "mahalanobis": 0.15,
    "reconstruction": 0.0,
    "channel": 0.1,
    "task": 0.1,
    "domain": 0.25,
    "adaptation": 0.05,
}
for regime in regimes:
    payload = {
        "name": f"{regime}_{os.environ.get('OPENSEMCOM_RUN_TAG', 'foundation')}",
        "regime": regime,
        "dataset_manifest": manifest,
        "output_dir": str(Path(os.environ["OPENSEMCOM_PROJECT_ROOT"]) / "runs"),
        "samples": int(os.environ.get("OPENSEMCOM_FULL_SAMPLES", "20000")),
        "calibration_samples": int(os.environ.get("OPENSEMCOM_FULL_CALIBRATION_SAMPLES", "640")),
        "users": int(os.environ.get("OPENSEMCOM_FULL_USERS", "4")),
        "seed": 0,
        "write_traces": regime == "full-open",
        "model": base_model,
        "channel": {"kind": "awgn", "snr_db": 12.0 if regime == "resource-open" else 18.0},
        "detector_weights": detector,
    }
    (config_root / f"{regime}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

for SEED in $SEEDS; do
  for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
    python -m opensemcom.cli.run_config \
      --config "$CONFIG_ROOT/${REGIME}.yaml" \
      --manifest "$MANIFEST" \
      --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
      --samples "$SAMPLES" \
      --calibration-samples "$CALIBRATION_SAMPLES" \
      --users "$USERS" \
      --seed "$SEED" \
      --run-name "${REGIME}_${RUN_TAG}_seed${SEED}_${SLURM_JOB_ID}" \
      --traces
  done
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --pattern "*_${RUN_TAG}_seed*_${SLURM_JOB_ID}/metrics.json" \
  --output-csv "$OPENSEMCOM_PROJECT_ROOT/runs/${RUN_TAG}_full_results_${SLURM_JOB_ID}.csv" \
  --output-json "$OPENSEMCOM_PROJECT_ROOT/runs/${RUN_TAG}_full_results_${SLURM_JOB_ID}.json"
