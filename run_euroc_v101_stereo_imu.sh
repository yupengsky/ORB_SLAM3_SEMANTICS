#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_PATH="${DATASET_PATH:-/media/yupeng/新加卷/Work/Git/ORB_SLAM_Datasets/vicon_room1/V1_01_easy}"
PANGOLIN_PREFIX="${PANGOLIN_PREFIX:-/media/yupeng/新加卷/Work/Git/pangolin_install}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/euroc_v101_stereo_imu}"
RUN_NAME="${RUN_NAME:-V101_stereo_imu}"

EXE="${REPO_ROOT}/Examples/Stereo-Inertial/stereo_inertial_euroc"
VOCAB="${REPO_ROOT}/Vocabulary/ORBvoc.txt"
SETTINGS="${REPO_ROOT}/Examples/Stereo-Inertial/EuRoC.yaml"
TIMESTAMPS="${REPO_ROOT}/Examples/Stereo-Inertial/EuRoC_TimeStamps/V101.txt"

for required_path in \
  "${EXE}" \
  "${VOCAB}" \
  "${SETTINGS}" \
  "${TIMESTAMPS}" \
  "${PANGOLIN_PREFIX}/lib/libpangolin.so" \
  "${DATASET_PATH}/mav0/cam0/data" \
  "${DATASET_PATH}/mav0/cam1/data" \
  "${DATASET_PATH}/mav0/imu0/data.csv"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${RESULT_DIR}"

export LD_LIBRARY_PATH="${REPO_ROOT}/lib:${REPO_ROOT}/Thirdparty/DBoW2/lib:${REPO_ROOT}/Thirdparty/g2o/lib:${PANGOLIN_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${RESULT_DIR}"
"${EXE}" "${VOCAB}" "${SETTINGS}" "${DATASET_PATH}" "${TIMESTAMPS}" "${RUN_NAME}"

echo "Trajectory: ${RESULT_DIR}/f_${RUN_NAME}.txt"
echo "Keyframes:  ${RESULT_DIR}/kf_${RUN_NAME}.txt"
