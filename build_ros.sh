#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS="${BUILD_JOBS:-4}"

echo "Building ROS nodes"

cd "${ROOT_DIR}/Examples/ROS/ORB_SLAM3"
mkdir -p build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"
