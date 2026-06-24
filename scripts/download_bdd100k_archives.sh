#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_scratch.sh"

BDD_ROOT="${OPENSEMCOM_BDD_ROOT:-$OPENSEMCOM_SCRATCH_ROOT/data/bdd100k}"
JOBS="${OPENSEMCOM_BDD_DOWNLOAD_JOBS:-8}"
CHUNK_BYTES="${OPENSEMCOM_BDD_CHUNK_BYTES:-67108864}"
TEST_ZIP="${OPENSEMCOM_BDD_TEST_ZIP:-1}"

mkdir -p "$BDD_ROOT"

download_archive() {
  local name="$1"
  local url="$2"
  local out="$BDD_ROOT/$name"
  local parts_dir="$BDD_ROOT/.parts/$name"

  echo "==> $name"
  if [[ -s "$out" ]] && unzip -tq "$out" >/dev/null 2>&1; then
    echo "already valid: $out"
    return 0
  fi

  local size
  size="$(curl -fsSI "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:" {print $2}' | tail -n 1)"
  if [[ -z "$size" || ! "$size" =~ ^[0-9]+$ ]]; then
    echo "could not determine content length for $url" >&2
    return 1
  fi
  echo "remote bytes: $size"

  mkdir -p "$parts_dir"
  local plan="$parts_dir/plan.tsv"
  awk -v size="$size" -v chunk="$CHUNK_BYTES" 'BEGIN {
    idx = 0
    for (start = 0; start < size; start += chunk) {
      end = start + chunk - 1
      if (end >= size) end = size - 1
      printf "%05d\t%d\t%d\t%d\n", idx, start, end, end - start + 1
      idx++
    }
  }' > "$plan"

  export BDD_DOWNLOAD_URL="$url"
  export BDD_PARTS_DIR="$parts_dir"
  xargs -P "$JOBS" -n 4 bash -c '
    set -euo pipefail
    idx="$1"
    start="$2"
    end="$3"
    expected="$4"
    part="$BDD_PARTS_DIR/part-$idx"
    tmp="$part.tmp"
    if [[ -s "$part" ]] && [[ "$(stat -c %s "$part")" == "$expected" ]]; then
      exit 0
    fi
    rm -f "$tmp"
    curl -fL --retry 8 --retry-delay 10 --connect-timeout 30 \
      --range "$start-$end" -o "$tmp" "$BDD_DOWNLOAD_URL"
    actual="$(stat -c %s "$tmp")"
    if [[ "$actual" != "$expected" ]]; then
      echo "range $start-$end expected $expected bytes, got $actual" >&2
      exit 1
    fi
    mv "$tmp" "$part"
  ' _ < "$plan"

  local tmp_out="$out.assembled"
  rm -f "$tmp_out"
  while IFS=$'\t' read -r idx _start _end _expected; do
    cat "$parts_dir/part-$idx" >> "$tmp_out"
  done < "$plan"

  local actual_size
  actual_size="$(stat -c %s "$tmp_out")"
  if [[ "$actual_size" != "$size" ]]; then
    echo "assembled $name expected $size bytes, got $actual_size" >&2
    return 1
  fi
  mv "$tmp_out" "$out"

  if [[ "$TEST_ZIP" == "1" ]]; then
    unzip -tq "$out" >/dev/null
  fi
  echo "downloaded: $out"
}

download_archive \
  "bdd100k_labels.zip" \
  "http://128.32.162.150/bdd100k/bdd100k_labels.zip"

download_archive \
  "bdd100k_images_100k.zip" \
  "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"

ls -lh "$BDD_ROOT"/*.zip
