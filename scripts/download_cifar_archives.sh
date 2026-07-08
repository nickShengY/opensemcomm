#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_scratch.sh"

DATA_ROOT="${OPENSEMCOM_DATA_ROOT:-$OPENSEMCOM_SCRATCH_ROOT/data}"
CIFAR10_ROOT="${OPENSEMCOM_CIFAR10_ROOT:-$DATA_ROOT/cifar10}"
CIFAR100_ROOT="${OPENSEMCOM_CIFAR100_ROOT:-$DATA_ROOT/cifar100}"

mkdir -p "$CIFAR10_ROOT" "$CIFAR100_ROOT"

download_and_extract() {
  local url="$1"
  local archive="$2"
  local dest="$3"
  local marker="$4"

  mkdir -p "$dest"
  if [[ -e "$marker" ]]; then
    echo "staged $(basename "$marker")"
    return 0
  fi
  if [[ ! -s "$archive" ]]; then
    curl -L --fail --retry 5 --retry-delay 10 -C - -o "$archive.part" "$url"
    mv "$archive.part" "$archive"
  else
    echo "exists $archive"
  fi
  tar -xzf "$archive" -C "$dest"
}

download_and_extract \
  "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" \
  "$CIFAR10_ROOT/cifar-10-python.tar.gz" \
  "$CIFAR10_ROOT" \
  "$CIFAR10_ROOT/cifar-10-batches-py"

download_and_extract \
  "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" \
  "$CIFAR100_ROOT/cifar-100-python.tar.gz" \
  "$CIFAR100_ROOT" \
  "$CIFAR100_ROOT/cifar-100-python"

ls -ld "$CIFAR10_ROOT/cifar-10-batches-py" "$CIFAR100_ROOT/cifar-100-python"
