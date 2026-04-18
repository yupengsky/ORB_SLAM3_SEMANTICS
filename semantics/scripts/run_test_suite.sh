#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_config.json}"

exec python3 "${SCRIPT_DIR}/offline_test_suite.py" \
  --config "${CONFIG_PATH}" \
  "$@"
