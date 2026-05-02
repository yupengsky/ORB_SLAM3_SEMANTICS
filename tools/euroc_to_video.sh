#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "${DATASET_PATH}" ]]; then
  echo "Set DATASET_PATH to a EuRoC ASL sequence path, for example:" >&2
  echo "  DATASET_PATH=<YOUR_EUROC_V1_01_EASY_ASL_PATH> MODE=stereo ./tools/euroc_to_video.sh" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/local/outputs/videos}"
FPS="${FPS:-20}"
MODE="${MODE:-stereo}"
SEQUENCE_NAME="${SEQUENCE_NAME:-$(basename "${DATASET_PATH}")}"

CAM0_DIR="${DATASET_PATH}/mav0/cam0/data"
CAM1_DIR="${DATASET_PATH}/mav0/cam1/data"

if [[ ! -d "${CAM0_DIR}" ]]; then
  echo "Missing EuRoC cam0 image directory: ${CAM0_DIR}" >&2
  exit 2
fi
if [[ "${MODE}" == "stereo" && ! -d "${CAM1_DIR}" ]]; then
  echo "Missing EuRoC cam1 image directory: ${CAM1_DIR}" >&2
  exit 2
fi

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
