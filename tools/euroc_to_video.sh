#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DATASET_ROOT="${EXTERNAL_DATASET_ROOT:-/media/yupeng/新加卷/Work/Git/ORB_SLAM_Datasets}"

DATASET_PATH="${DATASET_PATH:-/media/yupeng/新加卷/Work/Git/ORB_SLAM_Datasets/vicon_room1/V1_01_easy}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXTERNAL_DATASET_ROOT}/ORB_SLAM3_SEMANTICS_outputs/videos}"
FPS="${FPS:-20}"
MODE="${MODE:-stereo}"
SEQUENCE_NAME="${SEQUENCE_NAME:-$(basename "${DATASET_PATH}")}"

CAM0_DIR="${DATASET_PATH}/mav0/cam0/data"
CAM1_DIR="${DATASET_PATH}/mav0/cam1/data"

mkdir -p "${OUTPUT_DIR}"
TMP_DIR="$(mktemp -d "${OUTPUT_DIR}/.video_tmp_${SEQUENCE_NAME}_XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

link_sequence() {
  local src_dir="$1"
  local names_file="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"

  local index=0
  local name
  while IFS= read -r name; do
    index=$((index + 1))
    ln -s "${src_dir}/${name}" "${out_dir}/$(printf '%06d.png' "${index}")"
  done < "${names_file}"
}

find "${CAM0_DIR}" -maxdepth 1 -type f -name '*.png' -printf '%f\n' | sort > "${TMP_DIR}/cam0_names.txt"

if [[ "${MODE}" == "mono" ]]; then
  OUTPUT="${OUTPUT_DIR}/${SEQUENCE_NAME}_cam0_${FPS}fps.mp4"
  link_sequence "${CAM0_DIR}" "${TMP_DIR}/cam0_names.txt" "${TMP_DIR}/cam0"
  ffmpeg -y \
    -framerate "${FPS}" -i "${TMP_DIR}/cam0/%06d.png" \
    -vf "format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 18 \
    "${OUTPUT}"
elif [[ "${MODE}" == "stereo" ]]; then
  OUTPUT="${OUTPUT_DIR}/${SEQUENCE_NAME}_stereo_${FPS}fps.mp4"
  find "${CAM1_DIR}" -maxdepth 1 -type f -name '*.png' -printf '%f\n' | sort > "${TMP_DIR}/cam1_names.txt"
  comm -12 "${TMP_DIR}/cam0_names.txt" "${TMP_DIR}/cam1_names.txt" > "${TMP_DIR}/common_names.txt"
  link_sequence "${CAM0_DIR}" "${TMP_DIR}/common_names.txt" "${TMP_DIR}/cam0"
  link_sequence "${CAM1_DIR}" "${TMP_DIR}/common_names.txt" "${TMP_DIR}/cam1"
  ffmpeg -y \
    -framerate "${FPS}" -i "${TMP_DIR}/cam0/%06d.png" \
    -framerate "${FPS}" -i "${TMP_DIR}/cam1/%06d.png" \
    -filter_complex "[0:v][1:v]hstack=inputs=2,format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 18 \
    "${OUTPUT}"
else
  echo "Unknown MODE=${MODE}; use MODE=stereo or MODE=mono" >&2
  exit 1
fi

echo "${OUTPUT}"
