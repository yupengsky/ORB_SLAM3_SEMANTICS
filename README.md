# ORB_SLAM3_SEMANTICS

这是一个基于 ORB-SLAM3 的离线语义建图工程。C++ 侧仍负责视觉/视觉惯性 SLAM；语义侧在 SLAM 结束后读取导出的 KeyFrame、MapPoint 和 Observation，使用 YOLO-seg 给稀疏 3D 点云投票打标签，并生成面向导航理解的 JSON。

本仓库已经把本地路径、数据集、模型权重和生成结果隔离到 Git 忽略目录。提交版只保留源码、模板配置和说明文档。

## 目录约定

- `Examples/`: ORB-SLAM3 官方风格示例入口，覆盖 Monocular、Stereo、RGB-D、IMU、EuRoC、KITTI、TUM、TUM-VI 等。
- `semantics/scripts/`: 语义建图调度、自动降级、ADVIO 打包、视频渲染脚本。
- `semantics/scripts/dataset_config.json`: 可提交的配置模板，不应写真实本机路径。
- `local/dataset_config.json`: 每个开发者自己的真实配置，被 Git 忽略。
- `semantics/checkpoints/`: 本地模型权重目录，被 Git 忽略。
- `local/outputs/`、`local/results/`: 推荐的本地输出目录，被 Git 忽略。

## 干净 Ubuntu 从零安装

以下以 Ubuntu 22.04 LTS 为例。20.04/24.04 也可用，但 Pangolin 和 OpenCV 版本可能需要按本机环境微调。

1. 安装系统依赖：

```bash
sudo apt update
sudo apt install -y \
  git build-essential cmake pkg-config \
  libopencv-dev libeigen3-dev \
  libboost-dev libboost-serialization-dev libssl-dev \
  libgl1-mesa-dev libglew-dev \
  python3 python3-venv python3-pip \
  ffmpeg
```

2. 安装 Pangolin：

```bash
mkdir -p "$HOME/deps"
cd "$HOME/deps"
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout v0.6
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/pangolin"
cmake --build . -j"$(nproc)"
cmake --install .
```

3. Clone 本工程：

```bash
cd "$HOME"
git clone <THIS_REPOSITORY_URL> ORB_SLAM3_SEMANTICS
cd ORB_SLAM3_SEMANTICS
```

4. 编译 ORB-SLAM3 和示例程序：

```bash
export CMAKE_PREFIX_PATH="$HOME/.local/pangolin:${CMAKE_PREFIX_PATH:-}"
BUILD_JOBS="$(nproc)" ./build.sh
```

编译成功后应看到：

```text
lib/libORB_SLAM3.so
Examples/Stereo/stereo_euroc
Examples/Stereo-Inertial/stereo_inertial_euroc
Examples/Monocular/mono_euroc
Examples/RGB-D/rgbd_tum
```

5. 创建 Python 语义环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r semantics/requirements.txt
```

如需 GPU 版 PyTorch，请按你的 CUDA 版本先安装 PyTorch，再安装 `semantics/requirements.txt`。

6. 准备本地配置：

```bash
mkdir -p local
cp semantics/scripts/dataset_config.json local/dataset_config.json
${EDITOR:-nano} local/dataset_config.json
```

至少填写：

- `datasets.euroc_v101.path`: EuRoC V1_01_easy 的 ASL 格式目录。
- `external_output_root`: 推荐填 `local/outputs` 或一个本机大容量目录。
- `yolo_model`: 本地 YOLO segmentation 权重路径，例如 `semantics/checkpoints/yolo26l-seg.pt`。
- `pangolin_prefix`: 如果 Pangolin 安装在自定义位置，填 `$HOME/.local/pangolin`；系统路径可留空。

7. 准备模型权重：

```bash
mkdir -p semantics/checkpoints
```

把你的 YOLO-seg `.pt` 权重放到 `semantics/checkpoints/`，并在 `local/dataset_config.json` 的 `yolo_model` 中填写对应路径。权重文件不会被 Git 提交。

8. 准备数据集：

EuRoC 需要 ASL 格式目录，结构类似：

```text
<YOUR_EUROC_SEQUENCE>/
  mav0/cam0/data/*.png
  mav0/cam1/data/*.png
  mav0/imu0/data.csv
```

TUM RGB-D、KITTI、TUM-VI 也可通过 `Examples/` 下的原始 ORB-SLAM3 示例运行。语义自动调度脚本当前主要面向 EuRoC 风格数据和打包后的 ADVIO。

## 快速验证

1. 验证 C++ 示例能启动：

```bash
./Examples/Stereo/stereo_euroc
```

预期输出用法说明，而不是动态库加载错误。

2. 验证本地配置和数据集能力：

```bash
./semantics/scripts/run_dataset.sh --dataset-key euroc_v101 --dry-run
```

EuRoC V1_01 正常时应看到：

```text
has_monocular: True
has_stereo: True
has_imu_file: True
candidates=['stereo_imu', 'stereo', 'mono_imu', 'mono']
```

3. 运行短流程语义 smoke demo：

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

成功后应生成：

```text
local/results/demo_euroc_v101_stereo_imu/scene.json
local/results/demo_euroc_v101_stereo_imu/scene_sketch.txt
local/results/demo_euroc_v101_stereo_imu/navigation_llm_view.json
```

## 常用运行命令

自动检测数据集能力并按 `stereo_imu -> stereo -> mono_imu -> mono` 降级：

```bash
./semantics/scripts/run_dataset.sh --dataset-key euroc_v101
```

运行配置里的多数据集测试：

```bash
./semantics/scripts/run_test_suite.sh
```

手动运行 EuRoC stereo-inertial：

```bash
./Examples/Stereo-Inertial/stereo_inertial_euroc \
  Vocabulary/ORBvoc.txt \
  Examples/Stereo-Inertial/EuRoC.yaml \
  <YOUR_EUROC_SEQUENCE> \
  Examples/Stereo-Inertial/EuRoC_TimeStamps/V101.txt \
  output_euroc_v101_stereo_imu
```

生成 EuRoC 原始双目视频：

```bash
DATASET_PATH="<YOUR_EUROC_SEQUENCE>" \
MODE=stereo \
FPS=20 \
./tools/euroc_to_video.sh
```

## ADVIO 打包

如果使用 ADVIO，需要先把原始 iPhone 视频和 IMU 打包成 EuRoC-like 单目惯性格式：

```bash
python3 semantics/scripts/package_advio.py \
  --source-seq "<YOUR_RAW_ADVIO_SEQUENCE>" \
  --raw-out "local/outputs/advio/raw/advio-15" \
  --package-out "local/outputs/advio/euroc_like/advio-15-iphone-stride3" \
  --frame-stride 3
```

然后把 `local/dataset_config.json` 中 `datasets.advio_15.path` 指向 `package-out`。

## Git 提交边界

应该提交：

- C++/Python/shell 源码。
- `semantics/scripts/dataset_config.json` 这种模板配置。
- README 和其他通用说明。

不应该提交：

- `local/`
- 数据集目录
- `semantics/checkpoints/*.pt`、`*.pth`
- `semantics/results/` 里的生成结果
- `build/`、`lib/`、示例二进制

## 引用

本工程基于 ORB-SLAM3。使用 SLAM 主体算法时请引用：

```bibtex
@article{ORBSLAM3_TRO,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual-Inertial SLAM},
  author={Campos, Carlos and Elvira, Richard and Gomez Rodriguez, Juan J. and Montiel, Jose M. M. and Tardos, Juan D.},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={6},
  pages={1874--1890},
  year={2021}
}
```
