#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-/media/yupeng/新加卷/Work/Git/ORB_SLAM_Datasets/vicon_room1/V1_01_easy}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/euroc_v101_video}"
FPS="${FPS:-20}"
MODE="${MODE:-stereo}"

CAM0_DIR="${DATASET_PATH}/mav0/cam0/data"
CAM1_DIR="${DATASET_PATH}/mav0/cam1/data"

mkdir -p "${OUTPUT_DIR}"

if [[ "${MODE}" == "mono" ]]; then
  OUTPUT="${OUTPUT_DIR}/V1_01_easy_cam0_${FPS}fps.mp4"
  ffmpeg -y \
    -framerate "${FPS}" -pattern_type glob -i "${CAM0_DIR}/*.png" \
    -vf "format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 18 \
    "${OUTPUT}"
elif [[ "${MODE}" == "stereo" ]]; then
  OUTPUT="${OUTPUT_DIR}/V1_01_easy_stereo_${FPS}fps.mp4"
  ffmpeg -y \
    -framerate "${FPS}" -pattern_type glob -i "${CAM0_DIR}/*.png" \
    -framerate "${FPS}" -pattern_type glob -i "${CAM1_DIR}/*.png" \
    -filter_complex "[0:v][1:v]hstack=inputs=2,format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 18 \
    "${OUTPUT}"
else
  echo "Unknown MODE=${MODE}; use MODE=stereo or MODE=mono" >&2
  exit 1
fi

echo "${OUTPUT}"
