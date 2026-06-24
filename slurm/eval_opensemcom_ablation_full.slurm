#!/usr/bin/env bash
#SBATCH --job-name=osc-ablate-full
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.out
#SBATCH --error=/home/nickyun/links/scratch/new_study/opensemcom/logs/%x-%j.err

set -euo pipefail

source /home/nickyun/links/scratch/new_study/opensemcom/scripts/env_scratch.sh

MANIFEST="${OPENSEMCOM_FEATURE_MANIFEST:?Set OPENSEMCOM_FEATURE_MANIFEST}"
RUN_TAG="${OPENSEMCOM_RUN_TAG:-dinov3_ablation}"
SAMPLES="${OPENSEMCOM_FULL_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${OPENSEMCOM_FULL_CALIBRATION_SAMPLES:-384}"
USERS="${OPENSEMCOM_FULL_USERS:-4}"
SEEDS="${OPENSEMCOM_FULL_SEEDS:-0}"
VARIANTS="${OPENSEMCOM_ABLATION_VARIANTS:-full no_detector no_conformal no_harq no_adaptation fixed_scheduler}"
CONFIG_ROOT="$OPENSEMCOM_PROJECT_ROOT/tmp/configs_${RUN_TAG}_${SLURM_JOB_ID}"
export RUN_TAG VARIANTS

python -m opensemcom.cli.validate_manifest "$MANIFEST"
export MANIFEST CONFIG_ROOT

FEATURE_DIM="$(python - <<'PY'
import csv
import os
from pathlib import Path
import numpy as np
manifest = Path(os.environ["MANIFEST"]).expanduser().resolve()
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        print(np.load(Path(row["source_path"]).expanduser().resolve()).reshape(-1).shape[0])
        break
PY
)"
mkdir -p "$CONFIG_ROOT"
export FEATURE_DIM

python - <<'PY'
import os
from pathlib import Path
import yaml

manifest = str(Path(os.environ["MANIFEST"]).expanduser().resolve())
config_root = Path(os.environ["CONFIG_ROOT"])
feature_dim = int(os.environ["FEATURE_DIM"])
regimes = ("closed-id", "channel-open", "source-open", "class-open", "task-open", "resource-open", "full-open")
variants = os.environ["VARIANTS"].split()

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
variant_flags = {
    "full": {},
    "no_detector": {"use_detector": False},
    "no_conformal": {"use_conformal": False},
    "no_harq": {"use_harq": False},
    "no_adaptation": {"use_adaptation": False},
    "fixed_scheduler": {"use_scheduler": False},
}
for variant in variants:
    if variant not in variant_flags:
        raise ValueError(f"Unknown ablation variant: {variant}")
    for regime in regimes:
        payload = {
            "name": f"{regime}_{variant}_{os.environ.get('RUN_TAG', 'ablation')}",
            "regime": regime,
            "dataset_manifest": manifest,
            "output_dir": str(Path(os.environ["OPENSEMCOM_PROJECT_ROOT"]) / "runs"),
            "samples": int(os.environ.get("OPENSEMCOM_FULL_SAMPLES", "20000")),
            "calibration_samples": int(os.environ.get("OPENSEMCOM_FULL_CALIBRATION_SAMPLES", "384")),
            "users": int(os.environ.get("OPENSEMCOM_FULL_USERS", "4")),
            "seed": 0,
            "write_traces": regime == "full-open",
            "model": base_model,
            "channel": {"kind": "awgn", "snr_db": 12.0 if regime == "resource-open" else 18.0},
            "detector_weights": detector,
            "ablation": {"use_detector": True, "use_conformal": True, "use_harq": True, "use_adaptation": True, "use_scheduler": True, **variant_flags[variant]},
        }
        path = config_root / variant / f"{regime}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

for VARIANT in $VARIANTS; do
  for SEED in $SEEDS; do
    for REGIME in closed-id channel-open source-open class-open task-open resource-open full-open; do
      python -m opensemcom.cli.run_config \
        --config "$CONFIG_ROOT/${VARIANT}/${REGIME}.yaml" \
        --manifest "$MANIFEST" \
        --output-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
        --samples "$SAMPLES" \
        --calibration-samples "$CALIBRATION_SAMPLES" \
        --users "$USERS" \
        --seed "$SEED" \
        --run-name "${REGIME}_${RUN_TAG}_${VARIANT}_seed${SEED}_${SLURM_JOB_ID}" \
        --traces
    done
  done
done

python -m opensemcom.cli.aggregate_results \
  --runs-dir "$OPENSEMCOM_PROJECT_ROOT/runs" \
  --pattern "*_${RUN_TAG}_*_seed*_${SLURM_JOB_ID}/metrics.json" \
  --output-csv "$OPENSEMCOM_PROJECT_ROOT/runs/${RUN_TAG}_ablation_results_${SLURM_JOB_ID}.csv" \
  --output-json "$OPENSEMCOM_PROJECT_ROOT/runs/${RUN_TAG}_ablation_results_${SLURM_JOB_ID}.json"
