#!/usr/bin/env bash
#SBATCH --job-name=osc-comparaison-smoke
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/lustre10/scratch/nickyun/scratch/ahmed/opensemcomm/logs/%x-%j.out
#SBATCH --error=/lustre10/scratch/nickyun/scratch/ahmed/opensemcomm/logs/%x-%j.err
#SBATCH --mail-user=amaghri@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

PROJECT_ROOT="${OPENSEMCOM_PROJECT_ROOT:-/lustre10/scratch/nickyun/scratch/ahmed/opensemcomm}"
export OPENSEMCOM_PROJECT_ROOT="$PROJECT_ROOT"
export OPENSEMCOM_SCRATCH_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env_scratch.sh"

python - <<'PY'
import numpy as np
import torch

from opensemcom.channels import build_channel
from opensemcom.config import ChannelConfig
from opensemcom.types import ChannelBackend, ChannelKind

if not torch.cuda.is_available():
    raise RuntimeError("This smoke job requires a Slurm GPU allocation, but CUDA is unavailable.")

channel = build_channel(
    ChannelConfig(
        backend=ChannelBackend.SIONNA,
        kind=ChannelKind.AWGN,
        snr_db=30.0,
        sionna_device="cuda",
        sionna_seed=31,
    ),
    np.random.default_rng(31),
)
observation = channel.transmit(np.linspace(-0.8, 0.8, 32, dtype=np.float64))
assert observation.received.shape == (32,)
print({"cuda": torch.cuda.get_device_name(0), "phy": observation.state}, flush=True)
PY

python -m pytest -q tests/test_comparaison.py
