# Pipeline Notes

## 当前管线的设计原则

当前语义建图方案是“离线方案”，而不是把语义强耦合进 ORB-SLAM3 在线建图主循环。

总流程是：

```text
数据集输入
    ->
ORB-SLAM3 跑完整地图
    ->
Shutdown()
    ->
导出 final KeyFrame / MapPoint / Observation
    ->
离线跑 YOLO-seg
    ->
用 2D observation 给 3D MapPoint 投票打标签
    ->
生成本地输出目录中的中间语义图 semantic_map.json
    ->
生成最终 scene.json、scene_sketch.txt 与 navigation_llm_view.json
```

## 自动检测与降级逻辑

单数据集入口在：

- `semantics/scripts/run_dataset.sh`
- 核心逻辑：`semantics/scripts/offline_auto_degrade.py`

自动检测的数据集能力是：

- 是否有 `mav0/cam0/data`
- 是否有 `mav0/cam1/data`
- 是否有 `mav0/imu0/data.csv`

降级优先级由配置文件控制，当前默认是：

```text
stereo_imu -> stereo -> mono_imu -> mono
```

原则是：

- 优先双目。
- 其次单目。
- 同类里优先带 IMU。
- IMU 实际不可用时，不能只看“文件存在”，还要看运行是否成功。

这正是为了处理 ADVIO 这种情况：

- 数据集形式上提供 `cam0 + imu0`
- 但 `mono_imu` 运行后会失败或只得到极小、不可导航地图
- 脚本需要继续降级到 `mono`

## full_map 与 chunked_map

每个候选模式内部都遵循：

```text
先尝试 full_map
如果失败 -> fallback 到 chunked_map
如果 chunked 也失败 -> 该模式判定失败，继续降级
```

full_map / chunked_map 的编排核心在：

- `semantics/scripts/offline_pipeline.py`

关键点：

- 默认优先完整地图。
- chunked_map 只是兜底，不是首选。
- chunked_map 生成的是多段独立地图，不能当成同一个全局坐标系。

## 语义投票

语义投票逻辑在：

- `semantics/scripts/offline_semantic_mapper.py`

做的事情：

1. 读取 `summary.json / keyframes.jsonl / map_points.jsonl / observations.jsonl`
2. 对关键帧图像跑 YOLO-seg
3. 通过 observation 中的 `(u, v)` 将 2D mask 与 3D MapPoint 关联
4. 对每个 MapPoint 的标签做投票
5. 输出本地中间文件 `semantic_map.json`

## 最终场景输出

导航场景构建在：

- `semantics/scripts/navigation_scene_builder.py`

主要产物：

- `scene.json`
- `scene_sketch.txt`
- `navigation_llm_view.json`
- `semantic_objects`
- `spatial_relations`
- `traversable_path_network`
- `path_nearby_objects`
- `scene_summary`

`navigation_llm_view.json` 是从完整 `scene.json` 中提炼出的 LLM 推荐入口，主要包含物体 footprint、路径距离、clearance 和风险等级。

当前 JSON 顶层已经加入：

```json
"physical_scale_unit": "米"
```

或：

```json
"physical_scale_unit": "不确定"
```

同时还有：

```json
"has_metric_scale": true/false
```

含义：

- `stereo_imu` / `stereo` / 稳定 `mono_imu` -> 米制
- `mono` -> 尺度不确定

## 代码改动点

为了让 ORB-SLAM3 示例程序支持语义导出，当前改过以下 C++ 示例入口：

- `Examples/Stereo-Inertial/stereo_inertial_euroc.cc`
- `Examples/Monocular-Inertial/mono_inertial_euroc.cc`
- `Examples/Monocular/mono_euroc.cc`
- `Examples/Stereo/stereo_euroc.cc`

这些入口现在支持：

- 通过环境变量 `ORB_SLAM3_SEMANTIC_EXPORT_DIR`
- 在 `Shutdown()` 后导出语义所需的最终地图数据

若这些文件再被修改，通常需要重新编译对应 target。
