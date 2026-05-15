目前的数据集地址：/media/yupeng/新加卷/0-3-Projects/Touvigation/自采数据集/
目前的功能选择：mono（单目）
SLAM建图复现的完整指令：BUILD_JOBS="$(nproc)" ./build.sh && ./tools/run_zero2w_mono_slam.sh
对点云进行回放的完整指令：python3 tools/replay_slam_pointcloud.py local/slam_pointcloud_mono_3m_5m/slam_pointcloud_timeline.ply --port 8765
