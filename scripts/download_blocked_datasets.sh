#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_scratch.sh"

DATA_ROOT="$OPENSEMCOM_PROJECT_ROOT/data"
mkdir -p "$DATA_ROOT"/{nuscenes,nuimages,deepsense6g,bdd100k,bdd100k_hf,ag_news,cifar10,cifar100}

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    echo "exists $output"
    return 0
  fi
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$output.part" "$url"
  mv "$output.part" "$output"
}

extract_tgz_once() {
  local archive="$1"
  local dest="$2"
  local marker="$3"
  mkdir -p "$dest"
  if [[ -e "$marker" ]]; then
    echo "extracted $archive"
    return 0
  fi
  tar -xzf "$archive" -C "$dest"
}

extract_zip_once() {
  local archive="$1"
  local dest="$2"
  local marker="$3"
  mkdir -p "$dest"
  if [[ -e "$marker" ]]; then
    echo "extracted $archive"
    return 0
  fi
  unzip -q "$archive" -d "$dest"
}

download_file "https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz" "$DATA_ROOT/nuimages/nuimages-v1.0-mini.tgz"
extract_tgz_once "$DATA_ROOT/nuimages/nuimages-v1.0-mini.tgz" "$DATA_ROOT/nuimages" "$DATA_ROOT/nuimages/v1.0-mini"

download_file "https://www.nuscenes.org/data/v1.0-mini.tgz" "$DATA_ROOT/nuscenes/v1.0-mini.tgz"
extract_tgz_once "$DATA_ROOT/nuscenes/v1.0-mini.tgz" "$DATA_ROOT/nuscenes" "$DATA_ROOT/nuscenes/v1.0-mini"

download_file "https://www.dropbox.com/s/j1ignuyie2som3b/Scenario1.zip?dl=1" "$DATA_ROOT/deepsense6g/Scenario1.zip"
extract_zip_once "$DATA_ROOT/deepsense6g/Scenario1.zip" "$DATA_ROOT/deepsense6g" "$DATA_ROOT/deepsense6g/Scenario1"

OPENSEMCOM_DATA_ROOT="$DATA_ROOT" "$SCRIPT_DIR/download_cifar_archives.sh"
OPENSEMCOM_BDD_ROOT="$DATA_ROOT/bdd100k" "$SCRIPT_DIR/download_bdd100k_archives.sh"
python -m opensemcom.cli.download_datasets --data-root "$DATA_ROOT" --skip-bdd-parquet --skip-bdd-images

OPENSEMCOM_MAX_PER_SOURCE="${OPENSEMCOM_MAX_PER_SOURCE:-512}" \
OPENSEMCOM_MAX_CALIBRATION_PER_CLASS="${OPENSEMCOM_MAX_CALIBRATION_PER_CLASS:-64}" \
OPENSEMCOM_MAX_EVAL_PER_CLASS="${OPENSEMCOM_MAX_EVAL_PER_CLASS:-64}" \
  "$SCRIPT_DIR/download_models_and_data.sh"
