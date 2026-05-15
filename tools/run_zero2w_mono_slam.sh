#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-/media/yupeng/新加卷/0-3-Projects/Touvigation/自采数据集/vio_zero2w_20260514_122124}"
KALIBR_DIR="${KALIBR_DIR:-/media/yupeng/新加卷/0-3-Projects/Touvigation/自采数据集/vio_zero2w_20260514_122124_kalibr_result}"
PACKAGE_DIR="${PACKAGE_DIR:-${ROOT_DIR}/local/zero2w_mono_3m_5m}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/local/slam_pointcloud_mono_3m_5m}"
START_S="${START_S:-180}"
DURATION_S="${DURATION_S:-120}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"

POINT_CLOUD="${OUTPUT_DIR}/slam_pointcloud_timeline.ply"
WORK_DIR="${OUTPUT_DIR}/.work"

mkdir -p "${OUTPUT_DIR}"
rm -f "${POINT_CLOUD}"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

python3 "${ROOT_DIR}/tools/package_zero2w_mono.py" \
  --source-dir "${DATASET_DIR}" \
  --kalibr-dir "${KALIBR_DIR}" \
  --package-out "${PACKAGE_DIR}" \
  --start "${START_S}" \
  --duration "${DURATION_S}" \
  --frame-stride "${FRAME_STRIDE}"

export LD_LIBRARY_PATH="${ROOT_DIR}/lib:${ROOT_DIR}/Thirdparty/DBoW2/lib:${ROOT_DIR}/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"
export ORB_SLAM3_POINTCLOUD_TIMELINE_PATH="${POINT_CLOUD}"
export ORB_SLAM3_POINTCLOUD_ONLY=1

(
  cd "${WORK_DIR}"
  "${ROOT_DIR}/Examples/Monocular/mono_euroc" \
    "${ROOT_DIR}/Vocabulary/ORBvoc.txt" \
    "${PACKAGE_DIR}/ORB_SLAM3_mono.yaml" \
    "${PACKAGE_DIR}" \
    "${PACKAGE_DIR}/times.txt" \
    zero2w_mono_3m_5m
)

if [[ ! -s "${POINT_CLOUD}" ]]; then
  echo "SLAM did not produce ${POINT_CLOUD}" >&2
  exit 1
fi

rm -rf "${WORK_DIR}"
find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 ! -name "$(basename "${POINT_CLOUD}")" -exec rm -rf {} +
echo "${POINT_CLOUD}"
