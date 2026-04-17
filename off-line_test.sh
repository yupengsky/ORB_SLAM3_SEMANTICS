#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_PATH="${DATASET_PATH:-/media/yupeng/新加卷/Work/Git/ORB_SLAM_Datasets/vicon_room1/V1_01_easy}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/off-line_test}"
EXPORT_DIR="${EXPORT_DIR:-${RESULT_DIR}/slam_export}"
SLAM_WORK_DIR="${RESULT_DIR}/slam_run"
YOLO_MODEL="${YOLO_MODEL:-${ROOT_DIR}/semantics/checkpoints/yolo26l-seg.pt}"
SEMANTICS_PYTHON="${SEMANTICS_PYTHON:-/home/yupeng/miniconda3/envs/semantics/bin/python}"
RUN_NAME="${RUN_NAME:-V101_stereo_imu_semantic}"
RUN_SLAM="${RUN_SLAM:-1}"
RUN_YOLO="${RUN_YOLO:-1}"
OFFLINE_MAX_KEYFRAMES="${OFFLINE_MAX_KEYFRAMES:-0}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_CONF="${YOLO_CONF:-0.25}"
YOLO_DEVICE="${YOLO_DEVICE:-}"

SLAM_EXE="${ROOT_DIR}/Examples/Stereo-Inertial/stereo_inertial_euroc"
VOCABULARY="${ROOT_DIR}/Vocabulary/ORBvoc.txt"
SETTINGS="${ROOT_DIR}/Examples/Stereo-Inertial/EuRoC.yaml"
TIMESTAMPS="${ROOT_DIR}/Examples/Stereo-Inertial/EuRoC_TimeStamps/V101.txt"
SEMANTIC_SCRIPT="${ROOT_DIR}/semantics/offline_semantic_mapper.py"
SEMANTIC_JSON="${RESULT_DIR}/semantic_map.json"
SEMANTIC_PLY="${RESULT_DIR}/semantic_map.ply"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -x "${SLAM_EXE}" ]]; then
  echo "SLAM executable not found: ${SLAM_EXE}" >&2
  echo "Build first, for example: CMAKE_PREFIX_PATH=/media/yupeng/新加卷/Work/Git/pangolin_install cmake --build build -j\$(nproc)" >&2
  exit 1
fi

if [[ ! -f "${YOLO_MODEL}" ]]; then
  echo "YOLO model not found: ${YOLO_MODEL}" >&2
  exit 1
fi

if [[ ! -x "${SEMANTICS_PYTHON}" ]]; then
  echo "Semantics python not found: ${SEMANTICS_PYTHON}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${EXPORT_DIR}" "${SLAM_WORK_DIR}"

export LD_LIBRARY_PATH="${ROOT_DIR}/lib:${ROOT_DIR}/Thirdparty/DBoW2/lib:${ROOT_DIR}/Thirdparty/g2o/lib:/media/yupeng/新加卷/Work/Git/pangolin_install/lib:${LD_LIBRARY_PATH:-}"

if [[ "${RUN_SLAM}" == "1" ]]; then
  echo "[off-line_test] running ORB-SLAM3 stereo+IMU..."
  export ORB_SLAM3_SEMANTIC_EXPORT_DIR="${EXPORT_DIR}"
  (
    cd "${SLAM_WORK_DIR}"
    "${SLAM_EXE}" "${VOCABULARY}" "${SETTINGS}" "${DATASET_PATH}" "${TIMESTAMPS}" "${RUN_NAME}"
  )
fi

if [[ "${RUN_YOLO}" == "1" ]]; then
  echo "[off-line_test] running offline YOLO-seg semantic voting..."
  PY_ARGS=(
    "${SEMANTIC_SCRIPT}"
    --export-dir "${EXPORT_DIR}"
    --model "${YOLO_MODEL}"
    --settings "${SETTINGS}"
    --output "${SEMANTIC_JSON}"
    --semantic-ply "${SEMANTIC_PLY}"
    --imgsz "${YOLO_IMGSZ}"
    --conf "${YOLO_CONF}"
    --max-keyframes "${OFFLINE_MAX_KEYFRAMES}"
    --require-single-map
  )

  if [[ -n "${YOLO_DEVICE}" ]]; then
    PY_ARGS+=(--device "${YOLO_DEVICE}")
  fi

  "${SEMANTICS_PYTHON}" "${PY_ARGS[@]}"
fi

echo "[off-line_test] done"
echo "SLAM export: ${EXPORT_DIR}"
echo "Semantic JSON: ${SEMANTIC_JSON}"
echo "Semantic PLY: ${SEMANTIC_PLY}"
