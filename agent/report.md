# ORB_SLAM3_SEMANTICS 信号流报告

本文只说明 `semantics/` 里的离线语义建图脚本如何组织数据流，不展开 ORB-SLAM3 模型内部算法。

## 1. 总体信号流

整体管线是离线后处理，不把 YOLO 或语义模块嵌入 ORB-SLAM3 在线建图线程。

```text
数据集
  ->
ORB-SLAM3 示例程序运行
  ->
SLAM.Shutdown()
  ->
导出最终 KeyFrame / MapPoint / Observation
  ->
offline_semantic_mapper.py 给稀疏 MapPoint 投票添加语义
  ->
生成本地输出目录中的中间语义点云 semantic_map.json
  ->
navigation_scene_builder.py 聚类、建路径网、计算风险
  ->
生成最终 scene.json、scene_sketch.txt 和 navigation_llm_view.json
```

最终交付物保存在：

```text
semantics/results/<run_name>/scene.json
semantics/results/<run_name>/scene_sketch.txt
semantics/results/<run_name>/navigation_llm_view.json
```

其中 `scene.json` 是完整结构化场景地图，`scene_sketch.txt` 是导航优先的 ASCII 场景草图，`navigation_llm_view.json` 是专门给 LLM 读取的浓缩导航视图。

## 2. 脚本入口

单数据集入口：

```text
semantics/scripts/run_dataset.sh
```

它调用：

```text
semantics/scripts/offline_auto_degrade.py
```

作用是自动检测数据集能力并做功能降级。优先级由配置文件控制：

```text
stereo_imu -> stereo -> mono_imu -> mono
```

多数据集测试入口：

```text
semantics/scripts/run_test_suite.sh
```

它调用：

```text
semantics/scripts/offline_test_suite.py
```

作用是按配置文件轮流测试多个数据集和模式，例如 EuRoC V101/V102 的 `stereo_imu` 与 `mono_imu`。

统一配置文件：

```text
semantics/scripts/dataset_config.json
```

提交版文件只保存待填写模板。每个开发者应复制到 `local/dataset_config.json`，在 ignored 的本地配置里保存数据集路径、输出目录、YOLO 模型路径、Python 解释器路径和测试列表。

## 3. offline_pipeline.py 做什么

核心调度脚本是：

```text
semantics/scripts/offline_pipeline.py
```

它负责编排三个阶段：

```text
run_slam
run_yolo / semantic voting
run_navigation / scene building
```

第一阶段会设置环境变量：

```text
ORB_SLAM3_SEMANTIC_EXPORT_DIR=<slam_export_dir>
```

ORB-SLAM3 示例程序结束后会读取这个环境变量，把最终地图数据导出到本地输出目录。

导出的中间数据包括：

```text
summary.json
keyframes.jsonl
map_points.jsonl
observations.jsonl
```

含义如下：

```text
summary.json: 地图规模、关键帧数量、地图点数量、是否单地图
keyframes.jsonl: 关键帧位姿、时间戳、图像路径
map_points.jsonl: ORB-SLAM3 稀疏 3D MapPoint
observations.jsonl: MapPoint 在 KeyFrame 图像中的 2D 像素观测
```

如果 `full_map` 成功，就直接生成单张完整结果。如果失败，脚本支持 fallback 到 `chunked_map`。但当前最终结果优先使用 `full_map`，因为它保留一张完整全局地图。

## 4. 如何给稀疏点云添加语义

负责语义投票的脚本是：

```text
semantics/scripts/offline_semantic_mapper.py
```

输入是 ORB-SLAM3 导出的：

```text
keyframes.jsonl
map_points.jsonl
observations.jsonl
```

以及 YOLO-seg 模型：

```text
semantics/checkpoints/yolo26l-seg.pt
```

核心思想是利用 `observations.jsonl` 建立 2D 到 3D 的桥梁。

一条 observation 表示：

```text
某个 3D MapPoint 在某个 KeyFrame 图像上出现于像素坐标 (u, v)
```

脚本对每个关键帧图像运行 YOLO-seg，得到分割 mask。然后检查 observation 的 `(u, v)` 落在哪个 mask 内。

如果 `(u, v)` 落在 `bed` 的 mask 里，就给这个 3D MapPoint 投一票 `bed`。

一个 MapPoint 可能被多个关键帧看到，因此会收到多次投票。脚本最后对每个 MapPoint 统计所有标签得分，选择最高分标签。

简化表达如下：

```text
for keyframe:
    image -> YOLO-seg -> masks
    for observation in keyframe:
        point_id = observation.map_point_id
        pixel = (u, v)
        if pixel inside some mask:
            votes[point_id][label] += confidence

for map_point:
    label = argmax(votes[map_point])
```

输出是本地中间文件：

```text
semantic_map.json
```

它不是最终交付物，而是带语义标签的稀疏点云。每个点大致包含：

```text
map_point_id
position
label
score
semantic_observation_hits
orb_observations
label_scores
distance_m
```

这一步完成了：

```text
ORB-SLAM3 稀疏点云 -> 带语义标签的稀疏点云
```

## 5. 如何把语义点云转换成 scene.json

负责最终场景构建的脚本是：

```text
semantics/scripts/navigation_scene_builder.py
```

它读取：

```text
semantic_map.json
keyframes.jsonl
```

然后做四件事。

第一，按语义标签聚类 MapPoint。

同一标签的点会按 3D 空间距离做连通聚类。距离足够近的点会合成一个语义物体实例。

例如多个被标成 `bed` 且空间上相近的 MapPoint 会聚成：

```text
bed_001
```

第二，给每个语义物体计算几何信息。

对每个聚类出来的物体，脚本计算：

```text
center: 点云中心
bbox_3d.min: 3D 外包围盒最小坐标
bbox_3d.max: 3D 外包围盒最大坐标
extent: 3D 外包围盒尺寸
support_points: 支撑这个物体的语义点数量
```

这就是 scene.json 里物体尺寸和边界的来源。

需要注意：这里的边界是“语义稀疏点云外包围盒”，不是真实 CAD 边界，也不是稠密 mesh。

第三，从关键帧轨迹生成可行路径网。

脚本读取 `keyframes.jsonl` 里的关键帧位姿 `Twc`，取相机中心作为采集者轨迹点。相近的轨迹点会合并为路径节点，连续关键帧对应的节点会连成路径边。

得到：

```text
traversable_path_network.nodes
traversable_path_network.edges
path_network_length_m
recorded_trajectory_length_m
```

这张路径网表示“采集者实际走过的可行路径网络”。它不是完整自由空间地图，但对盲人导航 LLM 很有用，因为它提供了保守、已观察到的优先通行路线。

第四，计算物体到路径网的距离和风险。

脚本对每个物体计算：

```text
distance_to_path_network_m: 物体中心到路径网的最近距离
clearance_to_path_network_m: 物体 bbox 到路径网的最近距离
risk_level: high / medium / low
```

其中 `clearance_to_path_network_m` 比中心距离更适合导航，因为避障关心的是物体边界离路径有多近。

当前风险规则是：

```text
clearance <= 0.35 m -> high
0.35 m < clearance <= 0.75 m -> medium
clearance > 0.75 m -> low
```

最终写入完整地图：

```text
scene.json
```

主要字段包括：

```text
physical_scale_unit
has_metric_scale
scene_summary
global_bounds
semantic_objects
spatial_relations
traversable_path_network
path_nearby_objects
map_quality
```

同时脚本会额外写出：

```text
navigation_llm_view.json
```

这是专门给盲人导航 LLM 准备的紧凑视图，包含物体 footprint、bbox、height、clearance 和 risk。它不是新的观测信息，而是从 `scene.json` 中的完整地图结构提炼出来的 LLM 推荐入口。

## 6. 如何生成 scene_sketch.txt

`scene_sketch.txt` 也由：

```text
semantics/scripts/navigation_scene_builder.py
```

生成。

它不是原始点云图，而是从 `scene.json` 派生出的 ASCII 导航草图。

生成步骤如下：

第一，选择二维投影平面。

脚本根据 `global_bounds.extent` 找出范围最大的两个坐标轴作为草图平面。例如 EuRoC V101 stereo_imu 的草图是：

```text
horizontal = y
vertical = x
height = z
```

第二，把 3D 坐标映射到 ASCII 网格。

默认网格大小是：

```text
80 x 36
```

脚本根据全局边界把真实坐标线性映射为：

```text
row / col
```

第三，绘制障碍物 bbox 边框。

高风险物体绘制为：

```text
X
```

中风险物体绘制为：

```text
+
```

低风险物体不绘制在主图中，只保留在下方 legend 里，避免污染导航视图。

第四，最后绘制路径网。

路径网绘制为：

```text
#
```

并且路径最后绘制，所以不会被障碍物符号覆盖。这样可以保证可行路径视觉上连续。

起点和终点附近标为：

```text
S / E
```

第五，写出 legend。

草图下方会列出每个物体：

```text
X HIGH bed_001 label=bed footprint_y_by_x=5.63x4.29m height_z=1.68m dist_path=0.71m clearance=0.00m
```

这让 LLM 同时获得两类信息：

```text
主图：路径和风险区域的大致空间关系
legend：每个障碍物的类别、尺寸、距离和风险
```

## 7. 当前最终结果

当前工程内保留 5 组最终结果：

```text
semantics/results/EuRoC_V101_stereo_imu/
semantics/results/EuRoC_V101_mono_imu/
semantics/results/EuRoC_V102_stereo_imu/
semantics/results/EuRoC_V102_mono_imu/
semantics/results/ADVIO_15_iPhone_stride3_auto/
```

每个目录只保留：

```text
scene.json
scene_sketch.txt
navigation_llm_view.json
```

EuRoC 的 `stereo_imu` 和 `mono_imu` 结果是米制地图。

ADVIO 当前自动降级到 `mono`，因此结果可用但尺度为“不确定”。

## 8. 设计边界

这套管线的目标是：

```text
离线生成可供 LLM 理解的室内语义导航地图
```

它不解决：

```text
实时定位
实时避障
动态障碍物检测
YOLO 标签本身的准确率问题
稠密三维重建
真实墙体/自由空间占据栅格
```

因此给 LLM 使用时，应把它理解为：

```text
一张带语义物体、可行路径网、障碍物近路径风险的稀疏导航地图。
```

而不是完整、精确、可直接控制机器人运动的导航地图。
