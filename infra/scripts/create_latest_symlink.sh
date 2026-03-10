#!/usr/bin/env bash
# Creates/updates a 'latest' symlink in the model registry pointing to the
# most recent version directory.  Run this after each `make train`.
#
# Usage:  bash infra/scripts/create_latest_symlink.sh [registry_dir] [model_name]
#
# The API container mounts artifacts/ and reads MODEL_DIR=/app/artifacts/models/lgbm_surrogate/latest
# If you use latest.txt instead (default registry behaviour), set:
#   MODEL_DIR=""   (unset) and REGISTRY_DIR + MODEL_NAME in .env

set -euo pipefail

REGISTRY_DIR="${1:-artifacts/models}"
MODEL_NAME="${2:-lgbm_surrogate}"
MODEL_ROOT="${REGISTRY_DIR}/${MODEL_NAME}"

LATEST_TXT="${MODEL_ROOT}/latest.txt"
if [[ ! -f "$LATEST_TXT" ]]; then
    echo "ERROR: ${LATEST_TXT} not found. Run 'make train' first." >&2
    exit 1
fi

VERSION=$(cat "$LATEST_TXT" | tr -d '[:space:]')
TARGET="${MODEL_ROOT}/${VERSION}"
LINK="${MODEL_ROOT}/latest"

if [[ ! -d "$TARGET" ]]; then
    echo "ERROR: version directory ${TARGET} not found." >&2
    exit 1
fi

# Remove old symlink/dir if present
if [[ -L "$LINK" ]]; then rm "$LINK"; fi

ln -s "$(realpath "$TARGET")" "$LINK"
echo "Created symlink: ${LINK} -> ${TARGET}"
