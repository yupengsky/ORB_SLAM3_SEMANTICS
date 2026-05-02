# ORB_SLAM3_SEMANTICS Demo Commands

This file is safe to commit. It uses placeholder paths and assumes each developer keeps real paths in `local/dataset_config.json`, which is ignored by Git.

## 1. Enter The Repository

```bash
cd <ORB_SLAM3_SEMANTICS_REPO>
```

## 2. Prepare Local Configuration

```bash
mkdir -p local
cp semantics/scripts/dataset_config.json local/dataset_config.json
${EDITOR:-nano} local/dataset_config.json
```

Fill at least:

- `datasets.euroc_v101.path`
- `external_output_root`
- `yolo_model`
- Optional: `pangolin_prefix` when Pangolin is installed in a custom prefix.

## 3. Dry-Run Dataset Detection

```bash
./semantics/scripts/run_dataset.sh --dataset-key euroc_v101 --dry-run
```

Expected capabilities for EuRoC V1_01:

```text
has_monocular: True
has_stereo: True
has_imu_file: True
candidates=['stereo_imu', 'stereo', 'mono_imu', 'mono']
```

## 4. Run A Short Semantic Smoke Demo

```bash
python3 semantics/scripts/offline_pipeline.py \
  --config local/dataset_config.json \
  --dataset-key euroc_v101 \
  --slam-mode stereo_imu \
  --result-dir local/outputs/demo_euroc_v101_stereo_imu \
  --final-json-dir local/results/demo_euroc_v101_stereo_imu \
  --force-chunked \
  --chunk-size 450 \
  --chunk-overlap 0 \
  --chunk-min-frames 100 \
  --chunk-max-count 1 \
  --min-keyframes 5 \
  --min-map-points 50 \
  --min-observations 50 \
  --offline-max-keyframes 20 \
  --yolo-device cpu
```

Successful output should include:

```text
local/results/demo_euroc_v101_stereo_imu/scene.json
local/results/demo_euroc_v101_stereo_imu/scene_sketch.txt
local/results/demo_euroc_v101_stereo_imu/navigation_llm_view.json
```

## 5. Render A Semantic Flight Video

```bash
python3 semantics/scripts/render_semantic_flight_video.py \
  --run-dir local/outputs/demo_euroc_v101_stereo_imu \
  --result-dir local/results/demo_euroc_v101_stereo_imu \
  --output local/outputs/videos/demo_euroc_v101_stereo_imu_semantic_short.mp4 \
  --width 960 \
  --height 540 \
  --fps 12 \
  --frames 30 \
  --title "ORB-SLAM3 Semantics Demo - EuRoC V1_01 Stereo-IMU"
```

Validate the video:

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  local/outputs/videos/demo_euroc_v101_stereo_imu_semantic_short.mp4
```

## 6. Render Raw EuRoC Stereo Input Video

```bash
DATASET_PATH="<YOUR_EUROC_V1_01_EASY_ASL_PATH>" \
SEQUENCE_NAME="EuRoC_V101_raw" \
MODE=stereo \
FPS=20 \
./tools/euroc_to_video.sh
```

The default output directory is `local/outputs/videos`.
