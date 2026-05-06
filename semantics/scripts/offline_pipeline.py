#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")
    return read_json(config_path)


def resolve_config_path(value, root_dir):
    if value is None or value == "":
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(root_dir) / path)


def config_get(config, name, default=None):
    value = config.get(name)
    return default if value is None or value == "" else value


def is_placeholder(value):
    return isinstance(value, str) and "<YOUR_" in value


def safe_rmtree(path, allowed_roots):
    path = Path(path).resolve()
    if not path.exists():
        return
    if str(path) in {"/", str(Path.home())}:
        raise RuntimeError(f"Refuse to remove unsafe path: {path}")
    allowed = any(path == root or root in path.parents for root in allowed_roots)
    if not allowed:
        raise RuntimeError(f"Refuse to remove path outside managed outputs: {path}")
    shutil.rmtree(path)


def run_command(cmd, cwd, log_path, env):
    # 统一封装外部进程调用：管线中的 C++ SLAM 和 Python 后处理脚本都通过这里运行。
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        # 先把完整命令写入日志，方便失败后复现同一条命令。
        log.write("[command] " + " ".join(str(part) for part in cmd) + "\n\n")
        log.flush()
        # 在指定工作目录和环境变量下启动子进程，并把 stdout/stderr 都收进同一个日志文件。
        result = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    # 不在这里抛异常，而是把退出码返回给上层，由 run_slam 等函数结合产物质量决定是否失败。
    return result.returncode


def validate_export(export_dir, require_single_map=True, min_keyframes=10, min_map_points=50, min_observations=50):
    summary_path = Path(export_dir) / "summary.json"
    if not summary_path.exists():
        return False, f"missing {summary_path}"

    try:
        summary = read_json(summary_path)
    except Exception as exc:
        return False, f"cannot read summary: {exc}"

    keyframes = int(summary.get("keyframes_exported", 0))
    map_points = int(summary.get("map_points_exported", 0))
    observations = int(summary.get("observations_exported", 0))
    maps_with_keyframes = int(summary.get("maps_with_keyframes", 0))

    if require_single_map and maps_with_keyframes != 1:
        return False, f"maps_with_keyframes={maps_with_keyframes}, expected 1"
    if keyframes < min_keyframes:
        return False, f"keyframes_exported={keyframes}, expected >= {min_keyframes}"
    if map_points < min_map_points:
        return False, f"map_points_exported={map_points}, expected >= {min_map_points}"
    if observations < min_observations:
        return False, f"observations_exported={observations}, expected >= {min_observations}"

    return True, "ok"


def split_timestamps(timestamps_path, output_dir, chunk_size, overlap, min_frames, max_chunks):
    timestamps_path = Path(timestamps_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [line.strip() for line in timestamps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No timestamps in {timestamps_path}")

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    min_frames = max(1, int(min_frames))
    step = max(1, chunk_size - overlap)

    ranges = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        if len(lines) - end < min_frames and end < len(lines):
            end = len(lines)
        if end - start >= min_frames or not ranges:
            ranges.append((start, end))
        if end >= len(lines):
            break
        start += step
        if max_chunks > 0 and len(ranges) >= max_chunks:
            break

    chunks = []
    for index, (start, end) in enumerate(ranges):
        chunk_id = f"chunk_{index:03d}"
        chunk_file = output_dir / f"{chunk_id}.txt"
        chunk_file.write_text("\n".join(lines[start:end]) + "\n", encoding="utf-8")
        chunks.append(
            {
                "id": chunk_id,
                "timestamps_file": str(chunk_file),
                "start_line": start,
                "end_line_exclusive": end,
                "frame_count": end - start,
                "first_timestamp": lines[start],
                "last_timestamp": lines[end - 1],
            }
        )

    manifest = {"source_timestamps": str(timestamps_path), "chunks": chunks}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def apply_config(args, config):
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent.parent
    root_dir = Path(args.root_dir or config_get(config, "root_dir", str(default_root))).expanduser()
    if not root_dir.is_absolute():
        root_dir = (default_root / root_dir).resolve()
    args.root_dir = str(root_dir.resolve())

    if args.dataset_key:
        datasets = config.get("datasets", {})
        if args.dataset_key not in datasets:
            raise RuntimeError(f"Dataset key not found in config: {args.dataset_key}")
        dataset = datasets[args.dataset_key]
        args.dataset_kind = args.dataset_kind or dataset.get("kind")
        args.dataset_name = args.dataset_name or dataset.get("name", args.dataset_key)
        args.dataset_path = args.dataset_path or dataset.get("path")
        args.timestamp_id = args.timestamp_id or dataset.get("timestamp_id", "V101")

    args.timestamp_id = args.timestamp_id or "V101"
    args.dataset_name = args.dataset_name or args.dataset_key or "dataset"

    output_root = Path(
        resolve_config_path(
            config_get(config, "external_output_root", "local/outputs"),
            args.root_dir,
        )
    )
    default_result_name = f"{args.dataset_name}_{args.slam_mode}"
    args.run_name = args.run_name or f"{args.dataset_name}_{args.slam_mode}_semantic"
    args.result_dir = args.result_dir or str(output_root / default_result_name)
    args.final_json_dir = args.final_json_dir or str(Path(args.root_dir) / "semantics" / "results" / default_result_name)
    if args.annotated_dir == "auto":
        args.annotated_dir = str(Path(args.result_dir) / "tmp")

    args.vocabulary = args.vocabulary or str(Path(args.root_dir) / "Vocabulary" / "ORBvoc.txt")
    args.semantic_script = args.semantic_script or str(script_dir / "offline_semantic_mapper.py")
    args.navigation_script = args.navigation_script or str(script_dir / "navigation_scene_builder.py")
    args.pangolin_prefix = args.pangolin_prefix or config_get(config, "pangolin_prefix", "")
    args.yolo_model = args.yolo_model or resolve_config_path(config_get(config, "yolo_model", ""), args.root_dir)
    args.semantics_python = args.semantics_python or config_get(config, "semantics_python", sys.executable)

    missing = []
    required_fields = ["dataset_kind", "dataset_path", "slam_mode"]
    if args.run_yolo:
        required_fields.append("yolo_model")
    for field in required_fields:
        if not getattr(args, field, None):
            missing.append(field)
    if missing:
        raise RuntimeError("Missing required configuration fields: " + ", ".join(missing))

    return args


def mode_config(args):
    root = Path(args.root_dir)
    timestamp_id = args.timestamp_id

    if args.dataset_kind == "euroc":
        configs = {
            "stereo_imu": {
                "exe": root / "Examples/Stereo-Inertial/stereo_inertial_euroc",
                "settings": root / "Examples/Stereo-Inertial/EuRoC.yaml",
                "timestamps": root / f"Examples/Stereo-Inertial/EuRoC_TimeStamps/{timestamp_id}.txt",
                "scale_mode": "metric",
                "required": ["mav0/cam0/data", "mav0/cam1/data", "mav0/imu0/data.csv"],
            },
            "stereo": {
                "exe": root / "Examples/Stereo/stereo_euroc",
                "settings": root / "Examples/Stereo/EuRoC.yaml",
                "timestamps": root / f"Examples/Stereo/EuRoC_TimeStamps/{timestamp_id}.txt",
                "scale_mode": "metric",
                "required": ["mav0/cam0/data", "mav0/cam1/data"],
            },
            "mono_imu": {
                "exe": root / "Examples/Monocular-Inertial/mono_inertial_euroc",
                "settings": root / "Examples/Monocular-Inertial/EuRoC.yaml",
                "timestamps": root / f"Examples/Monocular-Inertial/EuRoC_TimeStamps/{timestamp_id}.txt",
                "scale_mode": "metric",
                "required": ["mav0/cam0/data", "mav0/imu0/data.csv"],
            },
            "mono": {
                "exe": root / "Examples/Monocular/mono_euroc",
                "settings": root / "Examples/Monocular/EuRoC.yaml",
                "timestamps": root / f"Examples/Monocular/EuRoC_TimeStamps/{timestamp_id}.txt",
                "scale_mode": "arbitrary",
                "required": ["mav0/cam0/data"],
            },
        }
    else:
        configs = {
            "mono_imu": {
                "exe": root / "Examples/Monocular-Inertial/mono_inertial_euroc",
                "settings": Path(args.dataset_path) / "ADVIO_iphone_mono_inertial.yaml",
                "timestamps": Path(args.dataset_path) / "times.txt",
                "scale_mode": "metric",
                "required": ["mav0/cam0/data", "mav0/imu0/data.csv"],
            },
            "mono": {
                "exe": root / "Examples/Monocular/mono_euroc",
                "settings": Path(args.dataset_path) / "ADVIO_iphone_mono_inertial.yaml",
                "timestamps": Path(args.dataset_path) / "times.txt",
                "scale_mode": "arbitrary",
                "required": ["mav0/cam0/data"],
            },
        }

    if args.slam_mode not in configs:
        choices = ", ".join(sorted(configs))
        raise RuntimeError(f"Unsupported mode '{args.slam_mode}' for {args.dataset_kind}; choose one of: {choices}")
    return configs[args.slam_mode]


def check_required_paths(args, cfg):
    paths = []
    placeholders = []
    if args.run_slam:
        paths.extend(
            [
                Path(args.dataset_path),
                Path(args.vocabulary),
                Path(cfg["exe"]),
                Path(cfg["settings"]),
                Path(cfg["timestamps"]),
            ]
        )
        paths.extend(Path(args.dataset_path) / rel for rel in cfg["required"])
    elif args.run_yolo:
        paths.append(Path(cfg["settings"]))
    if args.run_yolo:
        paths.append(Path(args.yolo_model))

    for path in paths:
        if is_placeholder(str(path)):
            placeholders.append(str(path))
    if placeholders:
        raise RuntimeError(
            "Configuration still contains placeholder path(s). "
            "Copy semantics/scripts/dataset_config.json to local/dataset_config.json and fill them in:\n"
            + "\n".join(placeholders)
        )

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing required path:\n" + "\n".join(missing))

    if args.run_slam and not os.access(cfg["exe"], os.X_OK):
        raise RuntimeError(f"SLAM executable is not executable: {cfg['exe']}")


def maybe_package_advio(args, config):
    if args.dataset_kind != "advio" or not args.package_advio:
        return
    package_cfg = config.get("advio_packaging", {})
    required = ("source_seq", "raw_out", "package_out")
    missing = [name for name in required if not package_cfg.get(name)]
    if missing:
        raise RuntimeError("Missing advio_packaging fields in config: " + ", ".join(missing))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from package_advio import package_advio

    print("[pipeline] packaging ADVIO from config")
    summary = package_advio(
        SimpleNamespace(
            source_seq=package_cfg["source_seq"],
            raw_out=package_cfg["raw_out"],
            package_out=package_cfg["package_out"],
            frame_stride=int(package_cfg.get("frame_stride", 3)),
            force=bool(args.force_package),
        )
    )
    print(
        "[pipeline] package_advio finished: "
        f"{summary['frames']} frames packaged={summary['packaged_sequence']}"
    )


def build_env(args, export_dir):
    env = os.environ.copy()
    root = Path(args.root_dir)
    library_paths = [
        root / "lib",
        root / "Thirdparty/DBoW2/lib",
        root / "Thirdparty/g2o/lib",
    ]
    if args.pangolin_prefix:
        library_paths.append(Path(args.pangolin_prefix) / "lib")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_paths if path.exists())
    if existing:
        env["LD_LIBRARY_PATH"] += ":" + existing
    env["ORB_SLAM3_SEMANTIC_EXPORT_DIR"] = str(export_dir)
    return env


def run_slam(args, cfg, timestamps, export_dir, work_dir, run_name, log_path):
    # 如果关闭 SLAM 阶段，就不重新运行 ORB-SLAM3，只检查已有导出结果能否被后续语义阶段复用。
    if not args.run_slam:
        # 复用旧结果时仍然要求导出目录中存在单张可用地图，并满足关键帧、地图点和观测数阈值。
        ok, reason = validate_export(
            export_dir,
            require_single_map=True,
            min_keyframes=args.min_keyframes,
            min_map_points=args.min_map_points,
            min_observations=args.min_observations,
        )
        return 0 if ok else 1, reason

    # 重新运行 SLAM 前清理旧导出和旧工作目录，避免历史结果混入本次 full_map 判断。
    safe_rmtree(export_dir, args.allowed_roots)
    safe_rmtree(work_dir, args.allowed_roots)
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # 组装 ORB-SLAM3 命令：可执行文件、词袋、相机配置、数据集、时间戳和本次运行名。
    cmd = [cfg["exe"], args.vocabulary, cfg["settings"], args.dataset_path, timestamps, run_name]
    # 运行 ORB-SLAM3；build_env 会把导出目录写入环境变量，让 C++ 侧把地图数据导出到 export_dir。
    rc = run_command(cmd, work_dir, log_path, build_env(args, export_dir))
    # SLAM 进程结束后，不只看退出码，还要检查导出的地图内容是否足够支撑语义融合。
    ok, reason = validate_export(
        export_dir,
        require_single_map=True,
        min_keyframes=args.min_keyframes,
        min_map_points=args.min_map_points,
        min_observations=args.min_observations,
    )
    # 最严重的失败：SLAM 进程异常退出，并且导出地图也不可用。
    if rc != 0 and not ok:
        return rc, f"slam_exit={rc}; {reason}"
    # SLAM 进程可能正常退出，但导出的地图质量不足，此时 full_map 仍然不能继续。
    if not ok:
        return rc, reason
    # 有时地图已经成功导出，但 SLAM 进程返回非零退出码；保留该异常状态交给上层处理。
    if rc != 0:
        return rc, f"export ok, but slam_exit={rc}"
    # 进程退出正常，并且导出地图通过质量检查，full_map 的 SLAM 阶段才算真正成功。
    return 0, "ok"


def run_semantic_and_navigation(args, cfg, export_dir, semantic_json, semantic_ply, scene_json, sketch_path, llm_view_json, annotated_dir, map_form):
    # 语义阶段：读取 SLAM 导出的关键帧/地图点/观测数据，运行 YOLO，并把检测结果融合回 3D 地图。
    if args.run_yolo:
        # 调用 offline_semantic_mapper.py；它负责生成 semantic_map.json，以及可选的语义点云 PLY。
        cmd = [
            args.semantics_python,
            args.semantic_script,
            "--export-dir",
            export_dir,
            "--model",
            args.yolo_model,
            "--settings",
            cfg["settings"],
            "--output",
            semantic_json,
            "--semantic-ply",
            semantic_ply,
            "--imgsz",
            str(args.yolo_imgsz),
            "--conf",
            str(args.yolo_conf),
            "--max-keyframes",
            str(args.offline_max_keyframes),
            # full_map 要求导出的 SLAM 结果来自单张地图，避免把多个互不对齐的地图强行融合。
            "--require-single-map",
        ]
        # 如果用户指定 YOLO 运行设备，就把设备参数传给语义脚本，例如 cpu、cuda 或 cuda:0。
        if args.yolo_device:
            cmd.extend(["--device", args.yolo_device])
        # 如果需要保存 YOLO 标注图，则创建目录并让语义脚本把可视化结果写进去。
        if annotated_dir:
            Path(annotated_dir).mkdir(parents=True, exist_ok=True)
            cmd.extend(["--annotated-dir", annotated_dir])
        # 这里使用 check=True：语义阶段失败会直接抛异常，由 run_full/run_chunked 的上层逻辑处理。
        subprocess.run([str(part) for part in cmd], check=True)

    # 导航阶段：把 semantic_map.json 中的语义对象和 SLAM 几何信息组织成可导航的场景描述。
    if args.run_navigation:
        # 必须保证已有 semantic_json 可用。
        if not Path(semantic_json).exists():
            raise RuntimeError(f"Semantic JSON not found: {semantic_json}")
        # 调用 navigation_scene_builder.py，输出最终 scene.json、文本草图和 LLM 友好的导航视图。
        cmd = [
            args.semantics_python,
            args.navigation_script,
            "--semantic-map",
            semantic_json,
            "--export-dir",
            export_dir,
            "--output",
            scene_json,
            "--sketch-output",
            sketch_path,
            "--llm-view-output",
            llm_view_json,
            "--cluster-radius",
            str(args.nav_cluster_radius),
            "--min-object-points",
            str(args.nav_min_object_points),
            "--path-node-radius",
            str(args.nav_path_node_radius),
            "--node-nearby-radius",
            str(args.nav_node_nearby_radius),
            "--path-nearby-radius",
            str(args.nav_path_nearby_radius),
            "--spatial-relation-radius",
            str(args.nav_spatial_relation_radius),
            "--spatial-relation-neighbors",
            str(args.nav_spatial_relation_neighbors),
            "--scale-mode",
            cfg["scale_mode"],
            "--map-form",
            map_form,
            "--slam-mode",
            args.slam_mode,
            "--dataset-name",
            args.dataset_name,
        ]
        # 导航阶段失败同样直接抛异常，让 full_map 被拒绝或让分块片段记录失败原因。
        subprocess.run([str(part) for part in cmd], check=True)


def summarize_semantic_index(segment_records, output_path, map_form):
    segments = []
    totals = {
        "segments_total": len(segment_records),
        "segments_succeeded": 0,
        "map_points_total": 0,
        "map_points_labeled": 0,
        "semantic_observation_hits": 0,
    }
    for record in segment_records:
        item = {
            "id": record["id"],
            "status": record["status"],
            "frame_count": record.get("frame_count"),
        }
        if record["status"] == "ok" and Path(record["semantic_json"]).exists():
            semantic = read_json(record["semantic_json"])
            summary = semantic.get("scene_summary", {})
            item["scene_summary"] = summary
            totals["segments_succeeded"] += 1
            totals["map_points_total"] += int(summary.get("map_points_total", 0))
            totals["map_points_labeled"] += int(summary.get("map_points_labeled", 0))
            totals["semantic_observation_hits"] += int(summary.get("semantic_observation_hits", 0))
        else:
            item["reason"] = record.get("reason")
        segments.append(item)

    totals["segments_failed"] = totals["segments_total"] - totals["segments_succeeded"]
    write_json(output_path, {"map_form": map_form, "scene_summary": totals, "segments": segments})


def write_chunked_scene_sketch(scene, output_path):
    lines = [
        f"scene_sketch: {scene.get('dataset_name', '')} / {scene.get('slam_mode', '')}",
        f"map_form: {scene.get('map_form', '')}",
        "scale_unit: meter" if scene.get("has_metric_scale") else "scale_unit: uncertain",
        "projection: unavailable",
        "note: chunked maps do not share one global coordinate frame, so one ASCII grid would be misleading.",
        (
            "summary: "
            f"segments={scene.get('scene_summary', {}).get('segments_total', 0)}, "
            f"objects={scene.get('scene_summary', {}).get('semantic_objects_total', 0)}, "
            f"path_nodes={scene.get('scene_summary', {}).get('path_nodes_total', 0)}, "
            f"path_edges={scene.get('scene_summary', {}).get('path_edges_total', 0)}"
        ),
    ]
    for segment in scene.get("segments", []):
        segment_scene = segment.get("scene", {})
        segment_summary = segment_scene.get("scene_summary", {})
        lines.append(
            f"{segment.get('id')}: status={segment.get('status')} "
            f"frames={segment.get('frame_count')} "
            f"objects={segment_summary.get('semantic_objects_total', 0)} "
            f"nodes={segment_summary.get('path_nodes_total', 0)} "
            f"edges={segment_summary.get('path_edges_total', 0)}"
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chunked_llm_view(scene, output_path):
    summary = scene.get("scene_summary", {})
    view = {
        "purpose": "compact chunked map facts for indoor blind-navigation LLM reasoning",
        "scale_unit": "meter" if scene.get("has_metric_scale") else "uncertain",
        "map_form": scene.get("map_form"),
        "segments_share_global_frame": False,
        "interpretation_notes": [
            "This is a chunked fallback result; segment coordinates are not one shared global frame.",
            "Prefer full_map results for navigation when available.",
            "Use each segment as a local semantic navigation sketch, not as one merged floor plan.",
        ],
        "path_network_compact": {
            "segments_total": summary.get("segments_total", 0),
            "segments_succeeded": summary.get("segments_succeeded", 0),
            "path_nodes_total": summary.get("path_nodes_total", 0),
            "path_edges_total": summary.get("path_edges_total", 0),
        },
        "segments": [
            {
                "id": segment.get("id"),
                "status": segment.get("status"),
                "frame_count": segment.get("frame_count"),
                "scene_summary": segment.get("scene", {}).get("scene_summary", {}),
            }
            for segment in scene.get("segments", [])
        ],
    }
    Path(output_path).write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")


def combine_navigation_segments(segment_records, output_path, sketch_path, llm_view_path, args, cfg, map_form):
    # 合并 chunked 导航结果的入口；这里只做分段结果汇总，不把各 chunk 的坐标强行拼成一个全局地图。
    has_metric_scale = cfg["scale_mode"] == "metric"
    # segments 保存每个分块的状态和可用 scene，summary 汇总所有分块的导航/语义统计。
    segments = []
    summary = {
        "segments_total": len(segment_records),
        "segments_succeeded": 0,
        "segments_failed": 0,
        "semantic_objects_total": 0,
        "path_nodes_total": 0,
        "path_edges_total": 0,
        "map_points_total": 0,
        "map_points_labeled": 0,
    }

    # 遍历每个 chunk 的执行记录；成功分块会读入其 navigation_json 和 semantic_json，失败分块只记录原因。
    for record in segment_records:
        segment = {
            "id": record["id"],
            "status": record["status"],
            "frame_count": record.get("frame_count"),
            "first_timestamp": record.get("first_timestamp"),
            "last_timestamp": record.get("last_timestamp"),
        }
        # 只有状态为 ok 且导航 JSON 确实存在的分块，才会被纳入最终 chunked scene。
        if record["status"] == "ok" and Path(record["navigation_json"]).exists():
            scene = read_json(record["navigation_json"])
            semantic = read_json(record["semantic_json"]) if Path(record["semantic_json"]).exists() else {}
            scene_summary = scene.get("scene_summary", {})
            semantic_summary = semantic.get("scene_summary", {})
            # 保留该分块自己的局部 scene；注意它的坐标只在该 chunk 内部有意义。
            segment["scene"] = scene
            # 汇总成功分块的对象数量、路径图规模和语义点统计。
            summary["segments_succeeded"] += 1
            summary["semantic_objects_total"] += int(scene_summary.get("semantic_objects_total", 0))
            summary["path_nodes_total"] += int(scene_summary.get("path_nodes_total", 0))
            summary["path_edges_total"] += int(scene_summary.get("path_edges_total", 0))
            summary["map_points_total"] += int(semantic_summary.get("map_points_total", 0))
            summary["map_points_labeled"] += int(semantic_summary.get("map_points_labeled", 0))
        else:
            # 失败分块不阻塞整体合并，但会在最终 JSON 中保留失败原因，便于排查。
            segment["reason"] = record.get("reason")
            summary["segments_failed"] += 1
        segments.append(segment)

    # 如果没有任何分块产出可用导航 JSON，chunked_map 也不能作为最终导航结果。
    if summary["segments_succeeded"] == 0:
        raise RuntimeError("No chunk produced a usable navigation JSON.")

    # 组装最终 chunked scene；核心元信息明确说明各分块不共享全局坐标系。
    output = {
        "physical_scale_unit": "米" if has_metric_scale else "不确定",
        "has_metric_scale": has_metric_scale,
        "map_form": map_form,
        "dataset_name": args.dataset_name,
        "slam_mode": args.slam_mode,
        "coordinate_frame": {
            "unit": "meter" if has_metric_scale else "arbitrary_slam_unit",
            "segments_share_global_frame": False,
            "note": "Each chunk is an independent ORB-SLAM3 map; do not compare positions across chunks as one global frame.",
        },
        "scene_summary": summary,
        "segments": segments,
        "map_quality": {
            "map_type": "offline_sparse_semantic_map",
            "map_form": map_form,
            "single_complete_map": False,
            "chunked_fallback": True,
            "has_metric_scale": has_metric_scale,
            "semantic_source": "offline image segmentation projected to ORB-SLAM3 MapPoints",
            "path_network_source": "per-chunk ORB-SLAM3 keyframe trajectories clustered into traversable path networks",
            "free_space_is_exhaustive": False,
            "requires_online_localization": True,
            "requires_realtime_obstacle_avoidance": True,
        },
    }
    # 写出最终 chunked scene、分块草图和面向 LLM 的分块摘要视图。
    write_json(output_path, output)
    write_chunked_scene_sketch(output, sketch_path)
    write_chunked_llm_view(output, llm_view_path)


def run_chunked(args, cfg, map_form="chunked_map"):
    # chunked_map 的总调度入口：把长序列切成多个时间段，每段独立运行 SLAM 和语义后处理。
    print(f"[pipeline] map_form={map_form} dataset={args.dataset_name} slam_mode={args.slam_mode}")
    # 准备分块流程的工作目录、时间戳目录、每段语义结果目录和日志目录。
    chunk_root = Path(args.result_dir) / "chunked"
    timestamps_dir = chunk_root / "timestamps"
    segments_dir = Path(args.result_dir) / "intermediate" / "segments"
    logs_dir = Path(args.result_dir) / "logs"

    # 重新运行 SLAM 时清理旧 chunk 输出，避免旧地图导出影响本次分块结果。
    if args.run_slam:
        safe_rmtree(chunk_root, args.allowed_roots)
    # 重新运行 YOLO 或导航时清理旧 segment 结果，避免旧语义 JSON 被误复用。
    if args.run_yolo or args.run_navigation:
        safe_rmtree(segments_dir, args.allowed_roots)
    timestamps_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    # 按 chunk_size/overlap/min_frames 等参数切分原始时间戳文件，生成每个分块自己的时间戳子文件。
    manifest = split_timestamps(
        cfg["timestamps"],
        timestamps_dir,
        args.chunk_size,
        args.chunk_overlap,
        args.chunk_min_frames,
        args.chunk_max_count,
    )
    print(f"[pipeline] chunks={len(manifest['chunks'])}")

    segment_records = []
    # 逐个处理分块；每个 chunk 都有独立的 SLAM 导出、语义地图和导航 scene。
    for chunk in manifest["chunks"]:
        chunk_id = chunk["id"]
        # 准备当前分块各阶段共享的输入/输出位置。
        export_dir = chunk_root / chunk_id / "slam_export"
        work_dir = chunk_root / chunk_id / "slam_run"
        segment_dir = segments_dir / chunk_id
        semantic_json = segment_dir / "semantic_map.json"
        semantic_ply = chunk_root / chunk_id / "semantic_map.ply"
        scene_json = segment_dir / "scene.json"
        sketch_path = segment_dir / "scene_sketch.txt"
        llm_view_json = segment_dir / "navigation_llm_view.json"
        annotated_dir = Path(args.annotated_dir) / chunk_id if args.annotated_dir else None
        log_path = logs_dir / f"{chunk_id}_slam.log"

        print(f"[pipeline] segment={chunk_id} map_form={map_form}")
        # 第一阶段：只用当前 chunk 的时间戳运行 ORB-SLAM3，让每个分块形成独立局部地图。
        rc, reason = run_slam(args, cfg, chunk["timestamps_file"], export_dir, work_dir, f"{args.run_name}_{chunk_id}", log_path)
        # 先创建失败记录；只有后续 SLAM 和语义/导航都通过时才改成 ok。
        record = {
            **chunk,
            "status": "failed",
            "reason": reason,
            "export_dir": str(export_dir),
            "semantic_json": str(semantic_json),
            "navigation_json": str(scene_json),
        }
        # SLAM 正常成功，或虽然退出码异常但导出地图可用时，继续尝试语义和导航阶段。
        if rc == 0 or reason.startswith("export ok"):
            try:
                # 第二阶段：在当前 chunk 的局部地图上运行 YOLO 语义融合和导航 scene 生成。
                run_semantic_and_navigation(
                    args,
                    cfg,
                    str(export_dir),
                    str(semantic_json),
                    str(semantic_ply),
                    str(scene_json),
                    str(sketch_path),
                    str(llm_view_json),
                    str(annotated_dir) if annotated_dir else "",
                    map_form,
                )
                # 当前分块完整通过后，记录为成功；最终合并阶段只会消费成功 segment。
                record["status"] = "ok"
                record["reason"] = "ok"
            except Exception as exc:
                # 单个分块语义/导航失败不会中断整个 chunked 流程，只记录失败原因并继续处理后续分块。
                record["reason"] = f"semantic/navigation failed: {exc}"
        print(f"[pipeline] segment={chunk_id} status={record['status']} reason={record['reason']}")
        segment_records.append(record)

    # 写出所有分块的语义索引，记录哪些分块成功、哪些失败，以及每段的语义统计。
    summarize_semantic_index(segment_records, Path(args.result_dir) / "intermediate" / "semantic_index.json", map_form)
    if args.run_navigation:
        # 将成功分块的导航 scene 合并成最终 chunked scene；注意各 chunk 不共享一个全局坐标系。
        combine_navigation_segments(
            segment_records,
            Path(args.final_json_dir) / "scene.json",
            Path(args.final_json_dir) / "scene_sketch.txt",
            Path(args.final_json_dir) / "navigation_llm_view.json",
            args,
            cfg,
            map_form,
        )
    # 返回每个分块的执行记录，供调用方或测试进一步检查。
    return segment_records


def run_full(args, cfg):
    # full_map 的总调度入口：优先尝试一次性构建完整 SLAM 地图，再继续做语义融合和导航生成。
    print(f"[pipeline] map_form=full_map dataset={args.dataset_name} slam_mode={args.slam_mode}")
    # 准备全量流程各阶段共享的输入/输出位置，用于串联 SLAM、语义融合和导航结果。
    export_dir = Path(args.result_dir) / "slam_export"
    work_dir = Path(args.result_dir) / "slam_run"
    logs_dir = Path(args.result_dir) / "logs"
    semantic_json = Path(args.result_dir) / "intermediate" / "semantic_map.json"
    semantic_ply = Path(args.result_dir) / "semantic_map.ply"
    scene_json = Path(args.final_json_dir) / "scene.json"
    sketch_path = Path(args.final_json_dir) / "scene_sketch.txt"
    llm_view_json = Path(args.final_json_dir) / "navigation_llm_view.json"
    log_path = logs_dir / "full_slam.log"

    # 第一阶段：运行 ORB-SLAM3 全量建图，并导出后续语义阶段需要的关键帧、地图点和观测数据。
    # run_slam 内部还会检查导出地图是否达到最低质量要求；失败原因会通过 reason 返回。
    rc, reason = run_slam(args, cfg, cfg["timestamps"], export_dir, work_dir, args.run_name, log_path)
    if rc != 0:
        # 全量 SLAM 没有产出可用地图时，直接拒绝 full_map，并把原因交给 main() 的回退逻辑处理。
        print(f"[pipeline] full_map rejected: {reason}")
        return False, reason

    try:
        # 第二阶段：在可用的全量 SLAM 地图上运行语义和导航后处理。
        # 这里会按参数开关决定是否执行 YOLO，随后生成语义地图、场景 JSON、草图和 LLM 导航视图。
        run_semantic_and_navigation(
            args,
            cfg,
            str(export_dir),
            str(semantic_json),
            str(semantic_ply),
            str(scene_json),
            str(sketch_path),
            str(llm_view_json),
            args.annotated_dir,
            # 标记这批语义/导航结果来自 full_map，方便最终 JSON 和日志区分全量流程与分块流程。
            "full_map",
        )
    except Exception as exc:
        # 语义融合或导航生成任一步失败，都说明 full_map 不能作为最终结果使用。
        # 统一包装失败原因，返回给 main() 决定是否 fallback 到 chunked_map。
        reason = f"semantic/navigation failed: {exc}"
        print(f"[pipeline] full_map rejected: {reason}")
        return False, reason

    # 能执行到这里，表示 SLAM 导出、语义融合和导航生成全部成功，全量地图被接受。
    print("[pipeline] full_map accepted")
    return True, "ok"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ORB-SLAM3 + offline semantic mapping with full-map-first fallback.")
    # 数据集配置文件路径，默认读取脚本同目录下的 dataset_config.json。
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "dataset_config.json"))
    # 在配置文件中选择的具体数据集条目键名；为空时由其他数据集参数共同决定。
    parser.add_argument("--dataset-key", default="")
    # 数据集类型，用于区分不同目录结构和时间戳格式。
    parser.add_argument("--dataset-kind", choices=("euroc", "advio"), default="")
    # 数据集名称，用于生成运行标识、结果目录或从配置中匹配条目。
    parser.add_argument("--dataset-name", default="")
    # 数据集根路径，通常指向 EuRoC 序列目录或 ADVIO 原始数据目录。
    parser.add_argument("--dataset-path", default="")
    # 工程根目录；为空时根据当前脚本位置自动推断。
    parser.add_argument("--root-dir", default="")
    # 本次运行的中间结果和 SLAM 输出目录；为空时使用默认结果目录规则。
    parser.add_argument("--result-dir", default="")
    # 最终语义 JSON 的输出目录；为空时写入默认 final_json 目录。
    parser.add_argument("--final-json-dir", default="")
    # ORB-SLAM3 运行模式，例如单目、双目或 RGB-D；该参数必须显式提供。
    parser.add_argument("--slam-mode", required=True)
    # 本次运行名称，用于组织输出文件；为空时由数据集和时间戳生成。
    parser.add_argument("--run-name", default="")
    # 运行时间戳标识，用于保证输出目录和文件名唯一。
    parser.add_argument("--timestamp-id", default="")
    # ORB-SLAM3 词袋文件路径；为空时使用工程内默认 Vocabulary。
    parser.add_argument("--vocabulary", default="")
    # 离线语义融合脚本路径；为空时使用工程内默认语义脚本。
    parser.add_argument("--semantic-script", default="")
    # 导航图生成脚本路径；为空时使用工程内默认导航脚本。
    parser.add_argument("--navigation-script", default="")
    # YOLO 模型权重路径或模型名，用于离线目标检测。
    parser.add_argument("--yolo-model", default="")
    # Pangolin 安装前缀，用于定位运行 ORB-SLAM3 时需要的动态库。
    parser.add_argument("--pangolin-prefix", default="")
    # 运行语义相关 Python 脚本的解释器路径；为空时使用当前默认 Python。
    parser.add_argument("--semantics-python", default="")
    # YOLO 标注可视化图片输出目录；为空时按默认规则生成。
    parser.add_argument("--annotated-dir", default="")
    # YOLO 推理设备，例如 cpu、cuda 或 cuda:0；为空时由 YOLO 自动选择。
    parser.add_argument("--yolo-device", default="")
    # 是否将 ADVIO 数据打包转换为管线可直接读取的格式。
    parser.add_argument("--package-advio", action=argparse.BooleanOptionalAction, default=False)
    # 是否强制重新执行 ADVIO 打包，即使目标文件已经存在。
    parser.add_argument("--force-package", action=argparse.BooleanOptionalAction, default=False)
    # 是否运行 ORB-SLAM3 前端建图阶段；关闭时复用已有 SLAM 输出。
    parser.add_argument("--run-slam", action=argparse.BooleanOptionalAction, default=True)
    # 是否运行 YOLO 目标检测阶段；关闭时复用已有检测结果。
    parser.add_argument("--run-yolo", action=argparse.BooleanOptionalAction, default=True)
    # 是否运行最终导航图生成阶段。
    parser.add_argument("--run-navigation", action=argparse.BooleanOptionalAction, default=True)
    # 全量离线融合失败或质量不足时，是否自动退回到分块处理。
    parser.add_argument("--fallback-to-chunks", action=argparse.BooleanOptionalAction, default=True)
    # 是否跳过全量融合，直接使用分块离线语义融合流程。
    parser.add_argument("--force-chunked", action=argparse.BooleanOptionalAction, default=False)
    # 全量离线融合最多使用的关键帧数量；0 表示不限制。
    parser.add_argument("--offline-max-keyframes", type=int, default=0)
    # YOLO 输入图像尺寸。
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    # YOLO 检测置信度阈值。
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    # 分块处理时每个块包含的目标帧数。
    parser.add_argument("--chunk-size", type=int, default=450)
    # 相邻分块之间的重叠帧数，用于减少块边界处的信息丢失。
    parser.add_argument("--chunk-overlap", type=int, default=30)
    # 分块处理时允许形成有效块的最少帧数。
    parser.add_argument("--chunk-min-frames", type=int, default=100)
    # 分块处理最多生成的块数；0 表示不限制。
    parser.add_argument("--chunk-max-count", type=int, default=0)
    # 判断 SLAM 输出是否可用时要求的最少关键帧数。
    parser.add_argument("--min-keyframes", type=int, default=10)
    # 判断 SLAM 输出是否可用时要求的最少地图点数。
    parser.add_argument("--min-map-points", type=int, default=50)
    # 判断语义融合结果是否可用时要求的最少观测数量。
    parser.add_argument("--min-observations", type=int, default=50)
    # 导航图中语义对象聚类的空间半径。
    parser.add_argument("--nav-cluster-radius", type=float, default=0.75)
    # 生成导航对象时，每个对象聚类需要包含的最少点数。
    parser.add_argument("--nav-min-object-points", type=int, default=5)
    # 导航路径节点的合并或邻接半径。
    parser.add_argument("--nav-path-node-radius", type=float, default=0.5)
    # 查找节点附近语义对象时使用的半径。
    parser.add_argument("--nav-node-nearby-radius", type=float, default=2.0)
    # 查找路径附近语义对象时使用的半径。
    parser.add_argument("--nav-path-nearby-radius", type=float, default=1.25)
    # 推断对象空间关系时使用的搜索半径。
    parser.add_argument("--nav-spatial-relation-radius", type=float, default=3.0)
    # 每个对象最多记录的空间关系邻居数量。
    parser.add_argument("--nav-spatial-relation-neighbors", type=int, default=3)
    return parser.parse_args()


def main():
    # 解析命令行参数，得到用户显式传入的运行配置。
    args = parse_args()
    # 读取 JSON 配置文件，补充数据集、路径和模型等默认配置。
    config = load_config(args.config)
    # 将配置文件中的默认值合并到命令行参数中；命令行参数优先级更高。
    args = apply_config(args, config)

    # 统一把关键路径转换为绝对路径，避免后续子进程因工作目录不同找不到文件。
    args.root_dir = str(Path(args.root_dir).resolve())
    args.dataset_path = str(Path(args.dataset_path).resolve())
    args.result_dir = str(Path(args.result_dir).resolve())
    args.final_json_dir = str(Path(args.final_json_dir).resolve())
    args.vocabulary = str(Path(args.vocabulary).resolve())
    args.semantic_script = str(Path(args.semantic_script).resolve())
    args.navigation_script = str(Path(args.navigation_script).resolve())
    args.pangolin_prefix = str(Path(args.pangolin_prefix).expanduser().resolve()) if args.pangolin_prefix else ""
    args.yolo_model = str(Path(args.yolo_model).resolve()) if args.yolo_model else ""
    args.semantics_python = str(Path(args.semantics_python).resolve()) if args.semantics_python else sys.executable
    args.allowed_roots = [Path(args.result_dir).resolve(), Path(args.final_json_dir).resolve()]

    # ADVIO 数据集在进入 SLAM 管线前可能需要先打包成统一格式。
    maybe_package_advio(args, config)
    # 根据 slam-mode 选择 ORB-SLAM3 可执行文件、相机配置和时间戳等模式相关配置。
    cfg = mode_config(args)
    # 检查数据集、词袋、脚本、配置文件等关键输入是否存在。
    check_required_paths(args, cfg)

    # 创建 SLAM 和语义管线的输出目录。
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    if args.run_navigation:
        # 重新生成导航结果前清理旧的 final_json 输出，清理范围受 allowed_roots 限制。
        safe_rmtree(args.final_json_dir, args.allowed_roots)
    Path(args.final_json_dir).mkdir(parents=True, exist_ok=True)

    if args.force_chunked:
        # 用户指定强制分块时，跳过全量地图流程，直接按时间段分块建图与融合。
        run_chunked(args, cfg)
        return

    # 默认优先尝试全量地图流程：运行 SLAM、YOLO、语义融合和导航图生成。
    ok, reason = run_full(args, cfg)
    if ok:
        return

    if not args.fallback_to_chunks:
        raise RuntimeError(reason)

    print(f"[pipeline] fallback_to_chunks=1 reason={reason}")
    # 全量流程失败或质量不足时，按配置退回到分块流程以提高完成率。
    run_chunked(args, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pipeline] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
