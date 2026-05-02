# Short Report For Advisor

本阶段已完成 ORB-SLAM3 离线语义建图与导航 JSON 管线整理。

结果如下：

- EuRoC `V101 / V102` 的 `stereo_imu` 与 `mono_imu` 共 4 轮测试已完成，均成功生成最终 `scene.json`、ASCII `scene_sketch.txt` 与 `navigation_llm_view.json`。
- ADVIO 已接入自动检测与功能降级流程。
- ADVIO 的 `mono_imu` 目前仍不稳定，系统已自动降级到 `mono`，成功生成非米制场景 JSON、ASCII 草图与 LLM 导航视图。
- 工程脚本已统一整理到 `semantics/scripts/`，提交版地址配置保留为 `semantics/scripts/dataset_config.json` 模板。
- 真实本地路径应放在 ignored 的 `local/dataset_config.json`，便于后续提交与继续开发。
