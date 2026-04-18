# Datasets And Current Results

## 数据集配置文件

统一配置在：

- `semantics/scripts/dataset_config.json`

当前已配置的数据集键值：

- `euroc_v101`
- `euroc_v102`
- `advio_15`

## 已验证的 EuRoC 结果

测试调度结果保存在：

- `semantics/results/test_suite_summary.json`

当前确认通过的四轮测试：

| dataset | mode | status | map form |
|---|---|---|---|
| EuRoC_V101 | stereo_imu | ok | full_map |
| EuRoC_V101 | mono_imu | ok | full_map |
| EuRoC_V102 | stereo_imu | ok | full_map |
| EuRoC_V102 | mono_imu | ok | full_map |

当前已有结果目录：

- `semantics/results/EuRoC_V101_stereo_imu/`
- `semantics/results/EuRoC_V101_mono_imu/`
- `semantics/results/EuRoC_V102_stereo_imu/`
- `semantics/results/EuRoC_V102_mono_imu/`

这些目录下都应至少有：

- `semantic_map.json`
- `semantic_navigation_map.json`

## ADVIO 当前结论

ADVIO 相关结果目录：

- `semantics/results/ADVIO_15_iPhone_stride3_auto/`
- `semantics/results/ADVIO_15_iPhone_stride3_auto_mono/`
- `semantics/results/ADVIO_15_iPhone_stride3_auto_mono_imu/`

最重要的是：

- `semantics/results/ADVIO_15_iPhone_stride3_auto/auto_degrade_report.json`

当前工程事实：

- ADVIO 形式上是 `cam0 + imu0`
- 自动检测结果会给出候选：
  - `mono_imu`
  - `mono`
- 但 `mono_imu` 经常失败，或者只留下极小、不可导航的伪成功地图
- 自动降级最终应选择 `mono`

当前一次成功降级的结果是：

- `selected_mode = mono`
- `physical_scale_unit = 不确定`

这表示：

- ADVIO 目前可用于非米制语义地图与导航 JSON 生成
- ADVIO 目前不应被视为稳定可用的单目惯性数据源

## 对未来 agent 的判断建议

如果你接手时看到：

- `ADVIO_15_iPhone_stride3_auto_mono_imu/` 里也有 JSON

不要直接把它当成功。

先检查：

- `auto_degrade_report.json` 的 `selected.mode`
- `quality_status`
- 最终被选中的是否仍然是 `mono`

因为 `mono_imu` 可能偶尔“留下点东西”，但地图太小，不足以给导航使用。

