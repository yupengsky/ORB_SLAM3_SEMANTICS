# Development Notes

## 接手时先注意什么

1. 当前 worktree 是脏的。
2. 不要随意回滚用户已有修改。
3. 语义正式脚本已经迁移到 `semantics/scripts/`，不要再把新脚本直接堆到 `semantics/` 根下。

## 当前最关键的文件

### Python / shell

- `semantics/scripts/run_dataset.sh`
  - 单数据集全过程入口
- `semantics/scripts/run_test_suite.sh`
  - 多数据集调度入口
- `semantics/scripts/dataset_config.json`
  - 所有地址、数据集键、测试列表
- `semantics/scripts/offline_auto_degrade.py`
  - 自动检测 + 模式降级
- `semantics/scripts/offline_pipeline.py`
  - full_map / chunked_map 编排
- `semantics/scripts/offline_semantic_mapper.py`
  - YOLO 投票到 MapPoint
- `semantics/scripts/navigation_scene_builder.py`
  - 生成导航 JSON
- `semantics/scripts/package_advio.py`
  - ADVIO 打包为 EuRoC-like

### C++

- `Examples/Stereo/stereo_euroc.cc`
- `Examples/Monocular/mono_euroc.cc`
- `Examples/Monocular-Inertial/mono_inertial_euroc.cc`
- `Examples/Stereo-Inertial/stereo_inertial_euroc.cc`

这些入口现在承担“跑 SLAM + 导出语义地图中间结果”的责任。

## 已知风险

### 1. ADVIO mono_imu 仍不稳定

这是当前最重要的技术风险。

已知现象：

- 全图经常失败
- 切块也经常失败
- 偶尔会留下极小地图，但不适合导航

当前结论：

- 不应把 ADVIO `mono_imu` 当成稳定链路
- 自动降级到 `mono` 是当前工程的正确行为

### 2. `semantics/results/` 里可能同时有成功产物和失败模式残留

后续 agent 不要只看“有 JSON 文件”就判定模式成功。

更可靠的检查方式：

- 先看 `auto_degrade_report.json`
- 再看 `selected.mode`
- 再看导航图里的 `physical_scale_unit`

### 3. 重新编译问题

如果修改了 C++ 示例入口，需要重新编译对应 target，例如：

```bash
cmake --build build --target stereo_euroc -j 8
cmake --build build --target mono_euroc -j 8
cmake --build build --target mono_inertial_euroc -j 8
cmake --build build --target stereo_inertial_euroc -j 8
```

## 推荐操作顺序

### 想快速确认当前工程还活着

跑 dry-run：

```bash
DATASET_KEY=euroc_v101 semantics/scripts/run_dataset.sh --dry-run
DATASET_KEY=advio_15 semantics/scripts/run_dataset.sh --dry-run
```

### 想真正产出一个结果

```bash
DATASET_KEY=euroc_v101 semantics/scripts/run_dataset.sh
```

### 想重跑批量测试

```bash
semantics/scripts/run_test_suite.sh
```

## 后续值得继续做的事情

1. 把 ADVIO `mono_imu` 的失败原因继续往“时间偏移 / 外参 / IMU 噪声 / 图像方向”方向定位。
2. 视需要继续收敛 `semantics/results/` 的结果命名与清理策略。
3. 如果后续要走在线语义建图，再单独设计在线模块，不要直接破坏当前离线稳定链路。
