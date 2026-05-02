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
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(str(part) for part in cmd) + "\n\n")
        log.flush()
        result = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
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
    if not args.run_slam:
        ok, reason = validate_export(
            export_dir,
            require_single_map=True,
            min_keyframes=args.min_keyframes,
            min_map_points=args.min_map_points,
            min_observations=args.min_observations,
        )
        return 0 if ok else 1, reason

    safe_rmtree(export_dir, args.allowed_roots)
    safe_rmtree(work_dir, args.allowed_roots)
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    cmd = [cfg["exe"], args.vocabulary, cfg["settings"], args.dataset_path, timestamps, run_name]
    rc = run_command(cmd, work_dir, log_path, build_env(args, export_dir))
    ok, reason = validate_export(
        export_dir,
        require_single_map=True,
        min_keyframes=args.min_keyframes,
        min_map_points=args.min_map_points,
        min_observations=args.min_observations,
    )
    if rc != 0 and not ok:
        return rc, f"slam_exit={rc}; {reason}"
    if not ok:
        return rc, reason
    if rc != 0:
        return rc, f"export ok, but slam_exit={rc}"
    return 0, "ok"


def run_semantic_and_navigation(args, cfg, export_dir, semantic_json, semantic_ply, scene_json, sketch_path, llm_view_json, annotated_dir, map_form):
    if args.run_yolo:
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
            "--require-single-map",
        ]
        if args.yolo_device:
            cmd.extend(["--device", args.yolo_device])
        if annotated_dir:
            Path(annotated_dir).mkdir(parents=True, exist_ok=True)
            cmd.extend(["--annotated-dir", annotated_dir])
        subprocess.run([str(part) for part in cmd], check=True)

    if args.run_navigation:
        if not Path(semantic_json).exists():
            raise RuntimeError(f"Semantic JSON not found: {semantic_json}")
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
    has_metric_scale = cfg["scale_mode"] == "metric"
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

    for record in segment_records:
        segment = {
            "id": record["id"],
            "status": record["status"],
            "frame_count": record.get("frame_count"),
            "first_timestamp": record.get("first_timestamp"),
            "last_timestamp": record.get("last_timestamp"),
        }
        if record["status"] == "ok" and Path(record["navigation_json"]).exists():
            scene = read_json(record["navigation_json"])
            semantic = read_json(record["semantic_json"]) if Path(record["semantic_json"]).exists() else {}
            scene_summary = scene.get("scene_summary", {})
            semantic_summary = semantic.get("scene_summary", {})
            segment["scene"] = scene
            summary["segments_succeeded"] += 1
            summary["semantic_objects_total"] += int(scene_summary.get("semantic_objects_total", 0))
            summary["path_nodes_total"] += int(scene_summary.get("path_nodes_total", 0))
            summary["path_edges_total"] += int(scene_summary.get("path_edges_total", 0))
            summary["map_points_total"] += int(semantic_summary.get("map_points_total", 0))
            summary["map_points_labeled"] += int(semantic_summary.get("map_points_labeled", 0))
        else:
            segment["reason"] = record.get("reason")
            summary["segments_failed"] += 1
        segments.append(segment)

    if summary["segments_succeeded"] == 0:
        raise RuntimeError("No chunk produced a usable navigation JSON.")

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
    write_json(output_path, output)
    write_chunked_scene_sketch(output, sketch_path)
    write_chunked_llm_view(output, llm_view_path)


def run_chunked(args, cfg, map_form="chunked_map"):
    print(f"[pipeline] map_form={map_form} dataset={args.dataset_name} slam_mode={args.slam_mode}")
    chunk_root = Path(args.result_dir) / "chunked"
    timestamps_dir = chunk_root / "timestamps"
    segments_dir = Path(args.result_dir) / "intermediate" / "segments"
    logs_dir = Path(args.result_dir) / "logs"

    if args.run_slam:
        safe_rmtree(chunk_root, args.allowed_roots)
    if args.run_yolo or args.run_navigation:
        safe_rmtree(segments_dir, args.allowed_roots)
    timestamps_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

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
    for chunk in manifest["chunks"]:
        chunk_id = chunk["id"]
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
        rc, reason = run_slam(args, cfg, chunk["timestamps_file"], export_dir, work_dir, f"{args.run_name}_{chunk_id}", log_path)
        record = {
            **chunk,
            "status": "failed",
            "reason": reason,
            "export_dir": str(export_dir),
            "semantic_json": str(semantic_json),
            "navigation_json": str(scene_json),
        }
        if rc == 0 or reason.startswith("export ok"):
            try:
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
                record["status"] = "ok"
                record["reason"] = "ok"
            except Exception as exc:
                record["reason"] = f"semantic/navigation failed: {exc}"
        print(f"[pipeline] segment={chunk_id} status={record['status']} reason={record['reason']}")
        segment_records.append(record)

    summarize_semantic_index(segment_records, Path(args.result_dir) / "intermediate" / "semantic_index.json", map_form)
    if args.run_navigation:
        combine_navigation_segments(
            segment_records,
            Path(args.final_json_dir) / "scene.json",
            Path(args.final_json_dir) / "scene_sketch.txt",
            Path(args.final_json_dir) / "navigation_llm_view.json",
            args,
            cfg,
            map_form,
        )
    return segment_records


def run_full(args, cfg):
    print(f"[pipeline] map_form=full_map dataset={args.dataset_name} slam_mode={args.slam_mode}")
    export_dir = Path(args.result_dir) / "slam_export"
    work_dir = Path(args.result_dir) / "slam_run"
    logs_dir = Path(args.result_dir) / "logs"
    semantic_json = Path(args.result_dir) / "intermediate" / "semantic_map.json"
    semantic_ply = Path(args.result_dir) / "semantic_map.ply"
    scene_json = Path(args.final_json_dir) / "scene.json"
    sketch_path = Path(args.final_json_dir) / "scene_sketch.txt"
    llm_view_json = Path(args.final_json_dir) / "navigation_llm_view.json"
    log_path = logs_dir / "full_slam.log"

    rc, reason = run_slam(args, cfg, cfg["timestamps"], export_dir, work_dir, args.run_name, log_path)
    if rc != 0:
        print(f"[pipeline] full_map rejected: {reason}")
        return False, reason

    try:
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
            "full_map",
        )
    except Exception as exc:
        reason = f"semantic/navigation failed: {exc}"
        print(f"[pipeline] full_map rejected: {reason}")
        return False, reason

    print("[pipeline] full_map accepted")
    return True, "ok"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ORB-SLAM3 + offline semantic mapping with full-map-first fallback.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "dataset_config.json"))
    parser.add_argument("--dataset-key", default="")
    parser.add_argument("--dataset-kind", choices=("euroc", "advio"), default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--root-dir", default="")
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--final-json-dir", default="")
    parser.add_argument("--slam-mode", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--timestamp-id", default="")
    parser.add_argument("--vocabulary", default="")
    parser.add_argument("--semantic-script", default="")
    parser.add_argument("--navigation-script", default="")
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--pangolin-prefix", default="")
    parser.add_argument("--semantics-python", default="")
    parser.add_argument("--annotated-dir", default="")
    parser.add_argument("--yolo-device", default="")
    parser.add_argument("--package-advio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-package", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-slam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-yolo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-navigation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-to-chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-chunked", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offline-max-keyframes", type=int, default=0)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--chunk-overlap", type=int, default=30)
    parser.add_argument("--chunk-min-frames", type=int, default=100)
    parser.add_argument("--chunk-max-count", type=int, default=0)
    parser.add_argument("--min-keyframes", type=int, default=10)
    parser.add_argument("--min-map-points", type=int, default=50)
    parser.add_argument("--min-observations", type=int, default=50)
    parser.add_argument("--nav-cluster-radius", type=float, default=0.75)
    parser.add_argument("--nav-min-object-points", type=int, default=5)
    parser.add_argument("--nav-path-node-radius", type=float, default=0.5)
    parser.add_argument("--nav-node-nearby-radius", type=float, default=2.0)
    parser.add_argument("--nav-path-nearby-radius", type=float, default=1.25)
    parser.add_argument("--nav-spatial-relation-radius", type=float, default=3.0)
    parser.add_argument("--nav-spatial-relation-neighbors", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    args = apply_config(args, config)
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

    maybe_package_advio(args, config)
    cfg = mode_config(args)
    check_required_paths(args, cfg)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    if args.run_navigation:
        safe_rmtree(args.final_json_dir, args.allowed_roots)
    Path(args.final_json_dir).mkdir(parents=True, exist_ok=True)

    if args.force_chunked:
        run_chunked(args, cfg)
        return

    ok, reason = run_full(args, cfg)
    if ok:
        return

    if not args.fallback_to_chunks:
        raise RuntimeError(reason)

    print(f"[pipeline] fallback_to_chunks=1 reason={reason}")
    run_chunked(args, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pipeline] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
