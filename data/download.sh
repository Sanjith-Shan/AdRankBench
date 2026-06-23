#!/usr/bin/env bash
# Thin wrapper that forwards to the real downloader in scripts/download_data.sh.
#
# The real logic lives in scripts/download_data.sh. This wrapper exists so that a
# user who looks inside the data directory can still kick off the download from
# here. Run from anywhere.
#   bash data/download.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec bash "${REPO_ROOT}/scripts/download_data.sh"
