#!/usr/bin/env bash
#SBATCH --job-name=osc-dino-unk-full
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

BASE_MANIFEST="${OPENSEMCOM_FEATURE_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_facebook__dinov3-vitb16-pretrain-lvd1689m.csv}"
MIXED_MANIFEST="${OPENSEMCOM_MIXED_MANIFEST:-$OPENSEMCOM_PROJECT_ROOT/manifests/opensemcom_real_dinov3_mixed_open_calibration.csv}"
RUN_TAG="${OPENSEMCOM_RUN_TAG:-dinov3_unknown_fullopen}"
SAMPLES="${OPENSEMCOM_FULL_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_FULL_CALIBRATION_SAMPLES:-1280}"
OPEN_CALIBRATION_ROWS="${OPENSEMCOM_OPEN_CALIBRATION_ROWS:-1024}"
USERS="${OPENSEMCOM_FULL_USERS:-4}"
SEEDS="${OPENSEMCOM_FULL_SEEDS:-0 1 2}"
CONFIG_ROOT="$OPENSEMCOM_PROJECT_ROOT/tmp/configs_${RUN_TAG}_${SLURM_JOB_ID}"

if [[ ! -s "$MIXED_MANIFEST" || "${OPENSEMCOM_REBUILD_MIXED_MANIFEST:-0}" == "1" ]]; then
  python -m opensemcom.cli.make_mixed_calibration_manifest \
    --input "$BASE_MANIFEST" \
    --output "$MIXED_MANIFEST" \
    --open-rows "$OPEN_CALIBRATION_ROWS" \
    --seed 17 \
    --regime full-open \
    --train-task classification \
    --train-domain cifar10
fi

python -m opensemcom.cli.validate_manifest "$MIXED_MANIFEST"
export MANIFEST="$MIXED_MANIFEST"

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
export FEATURE_DIM CONFIG_ROOT

python - <<'PY'
import os
from pathlib import Path
import yaml

manifest = str(Path(os.environ["MANIFEST"]).expanduser().resolve())
config_root = Path(os.environ["CONFIG_ROOT"])
feature_dim = int(os.environ["FEATURE_DIM"])
payload = {
    "name": f"full-open_{os.environ.get('OPENSEMCOM_RUN_TAG', 'dinov3_unknown_fullopen')}",
    "regime": "full-open",
    "dataset_manifest": manifest,
    "output_dir": str(Path(os.environ["OPENSEMCOM_PROJECT_ROOT"]) / "runs"),
    "samples": int(os.environ.get("OPENSEMCOM_FULL_SAMPLES", "20000")),
    "calibration_samples": int(os.environ.get("OPENSEMCOM_FULL_CALIBRATION_SAMPLES", "1280")),
    "users": int(os.environ.get("OPENSEMCOM_FULL_USERS", "4")),
    "seed": 0,
    "write_traces": True,
    "model": {
        "input_dim": feature_dim,
        "latent_dim": feature_dim,
        "projection": "identity",
        "classifier": "logistic",
        "num_known_classes": 6,
        "num_unknown_classes": 2,
        "tasks": ["classification", "detection", "segmentation", "driving-detection", "beam-prediction", "text-classification"],
        "train_tasks": ["classification"],
        "train_domains": ["cifar10"],
    },
    "channel": {"kind": "awgn", "snr_db": 18.0},
    "calibration": {
        "delta": 0.1,
        "accept_quantile": 0.45,
        "refine_quantile": 0.65,
        "mixed_open": True,
        "open_split": "open-calibration",
        "open_fraction": 0.70,
        "target_open_outage": 0.05,
    },
    "detector_weights": {
        "prediction": 0.10,
        "prototype": 0.05,
        "energy": 0.05,
        "unknown": 0.30,
        "openmax": 0.15,
        "vim": 0.10,
        "mahalanobis": 0.10,
        "reconstruction": 0.0,
        "channel": 0.05,
        "task": 0.05,
        "domain": 0.10,
        "adaptation": 0.0,
    },
}
(config_root / "full-open.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

for SEED in $SEEDS; do
  python -m opensemcom.cli.run_config \
    --config "$CONFIG_ROOT/full-open.yaml" \
    --manifest "$MIXED_MANIFEST" \
    --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
    --samples "$SAMPLES" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --users "$USERS" \
    --seed "$SEED" \
    --run-name "full-open_${RUN_TAG}_seed${SEED}_${SLURM_JOB_ID}" \
    --traces
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --pattern "full-open_${RUN_TAG}_seed*_${SLURM_JOB_ID}/metrics.json" \
  --output-csv "$OPENSEMCOM_PROJECT_ROOT/runs/${RUN_TAG}_fullopen_results_${SLURM_JOB_ID}.csv" \
  --output-json "$OPENSEMCOM_PROJECT_ROOT/runs/${RUN_TAG}_fullopen_results_${SLURM_JOB_ID}.json"

python -m opensemcom.cli.analyze_traces \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --pattern "full-open_${RUN_TAG}_seed*_${SLURM_JOB_ID}/traces.json" \
  --output-prefix "$OPENSEMCOM_PROJECT_ROOT/runs/analysis_${RUN_TAG}_${SLURM_JOB_ID}"
