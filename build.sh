#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS="${BUILD_JOBS:-4}"

echo "Configuring and building Thirdparty/DBoW2 ..."

cd "${ROOT_DIR}/Thirdparty/DBoW2"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"

cd "${ROOT_DIR}/Thirdparty/g2o"

echo "Configuring and building Thirdparty/g2o ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"

cd "${ROOT_DIR}/Thirdparty/Sophus"

echo "Configuring and building Thirdparty/Sophus ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"

cd "${ROOT_DIR}"

echo "Uncompress vocabulary ..."

cd Vocabulary
if [[ ! -f ORBvoc.txt ]]; then
  tar -xf ORBvoc.txt.tar.gz
fi
cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"
