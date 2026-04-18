# Short Report For Advisor

本阶段已完成 ORB-SLAM3 离线语义建图与导航 JSON 管线整理。

结果如下：

- EuRoC `V101 / V102` 的 `stereo_imu` 与 `mono_imu` 共 4 轮测试已完成，均成功生成完整 `semantic_map.json` 与 `semantic_navigation_map.json`。
- ADVIO 已接入自动检测与功能降级流程。
- ADVIO 的 `mono_imu` 目前仍不稳定，系统已自动降级到 `mono`，成功生成非米制导航 JSON。
- 工程脚本已统一整理到 `semantics/scripts/`，地址配置统一整理到 `semantics/scripts/dataset_config.json`。
- 仓库已清理，仅保留功能代码、最终 JSON 结果和必要文档，便于后续提交与继续开发。
