# ORB_SLAM3_SEMANTICS Agent Handoff

本目录面向后续接手工程的 coding agent。

目标不是解释 ORB-SLAM3 理论，而是让新 agent 在最短时间内回答下面几个问题：

1. 这个工程现在到底能做什么。
2. 入口脚本在哪里。
3. 语义建图与导航 JSON 是怎么产出的。
4. 哪些数据集已经验证过，哪些还不稳定。
5. 继续开发时应该优先看哪些文件。

## 当前工程定位

当前工程已经不是“原始 ORB-SLAM3 仓库”，而是一个带有离线语义建图导出能力的 ORB-SLAM3 变体：

- SLAM 核心仍然是 ORB-SLAM3。
- 语义部分是离线后处理，不是在线改 Tracking / LocalMapping 主流程。
- 最终目标是产出适合导航场景理解的 JSON。
- 当前设计优先保证“完整地图优先”，其次才允许功能降级。

## 现在的主目录约定

- `semantics/checkpoints/`
  - 语义模型权重目录。权重本身被 Git 忽略。
- `semantics/results/`
  - 生成结果目录。结果文件被 Git 忽略。
- `semantics/scripts/`
  - 所有正式脚本、配置、调度逻辑。
- `agent/`
  - 本交接文档目录。

注意：

- `semantics/` 根目录不应再继续堆脚本。
- 正式脚本入口都已经收敛到 `semantics/scripts/`。

## 推荐阅读顺序

1. `agent/README.md`
2. `agent/PIPELINE.md`
3. `agent/DATASETS_AND_RESULTS.md`
4. `agent/DEVELOPMENT_NOTES.md`

## 正式入口

单个数据集自动检测 + 功能降级：

- `semantics/scripts/run_dataset.sh`

多数据集调度测试：

- `semantics/scripts/run_test_suite.sh`

提交版配置模板：

- `semantics/scripts/dataset_config.json`

真实本地配置：

- `local/dataset_config.json`

兼容性 wrapper 仍然存在于仓库根目录：

- `off-line_test`
- `off-line_test_advio`
- `off-line_auto_degrade`

但规范入口仍然建议使用：

- `semantics/scripts/run_dataset.sh`
- `semantics/scripts/run_test_suite.sh`

## 一句话现状

截至当前交接时：

- EuRoC 风格数据集的 `stereo_imu` 与 `mono_imu` 已完整跑通。
- ADVIO 的自动降级已跑通。
- ADVIO `mono_imu` 仍不可靠，自动降级后会落到 `mono`。
- 导航 JSON 顶层已经明确写入物理尺度状态：
  - `"physical_scale_unit": "米"`
  - 或 `"physical_scale_unit": "不确定"`
