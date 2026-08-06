#!/usr/bin/env bash
# Install Meta's ImageBind package into the active virtual environment.
set -euo pipefail
IMAGEBIND_REPOSITORY="${OPENSEMCOM_IMAGEBIND_REPOSITORY:-https://github.com/facebookresearch/ImageBind.git}"
IMAGEBIND_REVISION="${OPENSEMCOM_IMAGEBIND_REVISION:-53680b02d7e37b19b124fa37bae4b6c98c38f5be}"
INSTALL_ROOT="${OPENSEMCOM_IMAGEBIND_SOURCE_DIR:-${TMPDIR:-/tmp}/opensemcom-imagebind}"
if [[ ! -d "$INSTALL_ROOT/.git" ]]; then
  git clone "$IMAGEBIND_REPOSITORY" "$INSTALL_ROOT"
fi
git -C "$INSTALL_ROOT" fetch --tags origin
git -C "$INSTALL_ROOT" checkout --detach "$IMAGEBIND_REVISION"
python -m pip install "$INSTALL_ROOT"
python -c 'import imagebind; print("ImageBind import OK")'