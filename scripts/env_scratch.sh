#!/usr/bin/env bash
set -euo pipefail

export OPENSEMCOM_PROJECT_ROOT="${OPENSEMCOM_PROJECT_ROOT:-/home/nickyun/links/scratch/new_study/opensemcom}"
export OPENSEMCOM_SCRATCH_ROOT="${OPENSEMCOM_SCRATCH_ROOT:-$OPENSEMCOM_PROJECT_ROOT}"

mkdir -p \
  "$OPENSEMCOM_SCRATCH_ROOT"/{data,models,manifests,runs,logs,cache,tmp} \
  "$OPENSEMCOM_SCRATCH_ROOT"/cache/{hf,datasets,transformers,torch,xdg,wandb}

export HF_HOME="$OPENSEMCOM_SCRATCH_ROOT/cache/hf"
export HF_DATASETS_CACHE="$OPENSEMCOM_SCRATCH_ROOT/cache/datasets"
export TRANSFORMERS_CACHE="$OPENSEMCOM_SCRATCH_ROOT/cache/transformers"
export TORCH_HOME="$OPENSEMCOM_SCRATCH_ROOT/cache/torch"
export XDG_CACHE_HOME="$OPENSEMCOM_SCRATCH_ROOT/cache/xdg"
export WANDB_DIR="$OPENSEMCOM_SCRATCH_ROOT/cache/wandb"
export TMPDIR="$OPENSEMCOM_SCRATCH_ROOT/tmp"
export PYTHONPATH="$OPENSEMCOM_PROJECT_ROOT/src:${PYTHONPATH:-}"

if [[ -d "$OPENSEMCOM_PROJECT_ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$OPENSEMCOM_PROJECT_ROOT/.venv/bin/activate"
fi
