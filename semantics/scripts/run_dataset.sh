#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_config.json}"
DATASET_KEY="${DATASET_KEY:-euroc_v101}"

exec python3 "${SCRIPT_DIR}/offline_auto_degrade.py" \
  --config "${CONFIG_PATH}" \
  --dataset-key "${DATASET_KEY}" \
  "$@"
