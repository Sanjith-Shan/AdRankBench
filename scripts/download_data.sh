#!/usr/bin/env bash
# Download a small Criteo style sample for the AdRankBench benchmark.
#
# This script tries to fetch a public sample of the Criteo display advertising
# data into data/criteo.csv. The download is best effort. If every mirror fails
# the script prints a clear note and exits with success so the benchmark can fall
# back to synthetic data on its own. The benchmark never requires this file.
#
# Run from the repository root.
#   bash scripts/download_data.sh

# We intentionally do not use a global set -e here. A failed curl must not abort
# the whole script. We handle each failure explicitly and always exit 0.
set -u

# Resolve the repository root from this script location so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
OUT_FILE="${DATA_DIR}/criteo.csv"

mkdir -p "${DATA_DIR}"

if [ -f "${OUT_FILE}" ]; then
  echo "data file already present at ${OUT_FILE}. Nothing to do."
  exit 0
fi

# Candidate public mirrors for a small Criteo sample. These may rotate over time.
# The benchmark does not depend on any of them succeeding.
MIRRORS=(
  "https://raw.githubusercontent.com/rixwew/pytorch-fm/master/examples/sample.txt"
  "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/sample.csv"
)

download_one() {
  url="$1"
  echo "trying ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --max-time 60 -o "${OUT_FILE}" "${url}" && return 0
  elif command -v wget >/dev/null 2>&1; then
    wget -q --timeout=60 -O "${OUT_FILE}" "${url}" && return 0
  else
    echo "neither curl nor wget is available."
    return 1
  fi
  return 1
}

for url in "${MIRRORS[@]}"; do
  if download_one "${url}"; then
    # Guard against an empty or tiny file that is not real data.
    if [ -s "${OUT_FILE}" ]; then
      echo "downloaded sample to ${OUT_FILE}."
      exit 0
    fi
    rm -f "${OUT_FILE}"
  fi
done

echo ""
echo "could not download a Criteo sample from any mirror."
echo "this is fine. the benchmark will fall back to synthetic data automatically."
echo "run the benchmark with the synthetic flag to be explicit."
echo "  python scripts/run_benchmark.py --synthetic --sample-size 100000"
exit 0
