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

exec python3 "${SCRIPT_DIR}/offline_test_suite.py" \
  --config "${CONFIG_PATH}" \
  "$@"
