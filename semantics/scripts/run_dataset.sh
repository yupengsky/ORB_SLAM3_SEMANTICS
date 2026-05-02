#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_CONFIG="${ROOT_DIR}/local/dataset_config.json"
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  CONFIG_PATH="${DATASET_CONFIG}"
elif [[ -f "${LOCAL_CONFIG}" ]]; then
  CONFIG_PATH="${LOCAL_CONFIG}"
else
  CONFIG_PATH="${SCRIPT_DIR}/dataset_config.json"
fi
DATASET_KEY="${DATASET_KEY:-euroc_v101}"

exec python3 "${SCRIPT_DIR}/offline_auto_degrade.py" \
  --config "${CONFIG_PATH}" \
  --dataset-key "${DATASET_KEY}" \
  "$@"
