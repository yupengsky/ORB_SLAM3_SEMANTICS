#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bool_arg(enabled, name):
    return f"--{name}" if enabled else f"--no-{name}"


def count_images(path):
    path = Path(path)
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.suffix.lower() in {".png", ".jpg", ".jpeg"})


def imu_rows(path):
    path = Path(path)
    if not path.is_file():
        return 0
    rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows += 1
    return rows


def detect_capabilities(dataset_path):
    dataset_path = Path(dataset_path)
    cam0_dir = dataset_path / "mav0" / "cam0" / "data"
    cam1_dir = dataset_path / "mav0" / "cam1" / "data"
    imu_file = dataset_path / "mav0" / "imu0" / "data.csv"
    cam0_images = count_images(cam0_dir)
    cam1_images = count_images(cam1_dir)
    imu_count = imu_rows(imu_file)
    return {
        "dataset_path": str(dataset_path),
        "has_monocular": cam0_images > 0,
        "has_stereo": cam0_images > 0 and cam1_images > 0,
        "has_imu_file": imu_count > 0,
        "cam0_images": cam0_images,
        "cam1_images": cam1_images,
        "imu_rows": imu_count,
    }


def candidate_modes(config, dataset_kind, capabilities):
    priority = config.get("auto_degrade_priority", ["stereo_imu", "stereo", "mono_imu", "mono"])
    available = set()
    if capabilities["has_stereo"] and capabilities["has_imu_file"]:
        available.add("stereo_imu")
    if capabilities["has_stereo"]:
        available.add("stereo")
    if capabilities["has_monocular"] and capabilities["has_imu_file"]:
        available.add("mono_imu")
    if capabilities["has_monocular"]:
        available.add("mono")

    if dataset_kind == "advio":
        available.discard("stereo_imu")
        available.discard("stereo")

    return [mode for mode in priority if mode in available]


def default_paths(config, dataset_name, mode, suffix):
    output_root = Path(config["external_output_root"])
    run_name = f"{dataset_name}_{suffix}_{mode}"
    return {
        "result_dir": str(output_root / run_name),
        "final_json_dir": str(output_root / run_name / "final_scene"),
    }


def run_pipeline(args, config, dataset_key, dataset_name, mode):
    paths = default_paths(config, dataset_name, mode, args.output_suffix)
    pipeline = Path(__file__).resolve().parent / "offline_pipeline.py"
    cmd = [
        sys.executable,
        str(pipeline),
        "--config",
        args.config,
        "--dataset-key",
        dataset_key,
        "--slam-mode",
        mode,
        "--result-dir",
        paths["result_dir"],
        "--final-json-dir",
        paths["final_json_dir"],
        bool_arg(args.run_slam, "run-slam"),
        bool_arg(args.run_yolo, "run-yolo"),
        bool_arg(args.run_navigation, "run-navigation"),
        bool_arg(args.fallback_to_chunks, "fallback-to-chunks"),
        bool_arg(args.force_chunked, "force-chunked"),
        "--offline-max-keyframes",
        str(args.offline_max_keyframes),
    ]
    if args.annotated_dir:
        cmd.extend(["--annotated-dir", args.annotated_dir])
    if args.yolo_device:
        cmd.extend(["--yolo-device", args.yolo_device])
    if args.package_advio:
        cmd.append("--package-advio")
    if args.force_package:
        cmd.append("--force-package")

    print(f"[auto_degrade] trying mode={mode}")
    rc = subprocess.run(cmd, check=False).returncode
    scene_json = Path(paths["final_json_dir"]) / "scene.json"
    scene_sketch = Path(paths["final_json_dir"]) / "scene_sketch.txt"
    llm_view_json = Path(paths["final_json_dir"]) / "navigation_llm_view.json"
    ok = rc == 0 and scene_json.exists() and scene_sketch.exists() and llm_view_json.exists()
    return {
        "mode": mode,
        "status": "ok" if ok else "failed",
        "return_code": rc,
        "result_dir": paths["result_dir"],
        "final_json_dir": paths["final_json_dir"],
        "scene_json": str(scene_json),
        "scene_sketch": str(scene_sketch),
        "navigation_llm_view": str(llm_view_json),
    }


def copy_selected_outputs(selected, final_dir):
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("scene_json", "scene_sketch", "navigation_llm_view"):
        source = Path(selected[name])
        if source.exists():
            shutil.copy2(source, final_dir / source.name)


def validate_attempt_quality(attempt, args):
    scene_path = Path(attempt["scene_json"])
    sketch_path = Path(attempt["scene_sketch"])
    llm_view_path = Path(attempt["navigation_llm_view"])
    if not scene_path.exists() or not sketch_path.exists() or not llm_view_path.exists():
        return False, "missing scene JSON, ASCII sketch, or LLM view JSON"

    scene = read_json(scene_path)
    summary = scene.get("scene_summary", {})

    keyframes = int(summary.get("keyframes_processed", summary.get("path_nodes_total", 0)))
    points = int(summary.get("map_points_total", 0))
    nodes = int(summary.get("path_nodes_total", 0))
    edges = int(summary.get("path_edges_total", 0))

    if keyframes < args.min_semantic_keyframes:
        return False, f"semantic keyframes too few: {keyframes} < {args.min_semantic_keyframes}"
    if points < args.min_semantic_points:
        return False, f"map points too few: {points} < {args.min_semantic_points}"
    if nodes < args.min_path_nodes:
        return False, f"path nodes too few: {nodes} < {args.min_path_nodes}"
    if edges < args.min_path_edges:
        return False, f"path edges too few: {edges} < {args.min_path_edges}"

    return True, "ok"


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-detect dataset modality and run ORB-SLAM3 semantic mapping with mode fallback.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "dataset_config.json"))
    parser.add_argument("--dataset-key", default="advio_15")
    parser.add_argument("--output-suffix", default="auto")
    parser.add_argument("--run-slam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-yolo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-navigation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-to-chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-chunked", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--package-advio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-package", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offline-max-keyframes", type=int, default=0)
    parser.add_argument("--min-semantic-keyframes", type=int, default=20)
    parser.add_argument("--min-semantic-points", type=int, default=500)
    parser.add_argument("--min-path-nodes", type=int, default=3)
    parser.add_argument("--min-path-edges", type=int, default=1)
    parser.add_argument("--annotated-dir", default="")
    parser.add_argument("--yolo-device", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(args.config)
    datasets = config.get("datasets", {})
    if args.dataset_key not in datasets:
        raise RuntimeError(f"Dataset key not found in config: {args.dataset_key}")

    dataset = datasets[args.dataset_key]
    dataset_name = dataset.get("name", args.dataset_key)
    dataset_kind = dataset.get("kind", "euroc")
    capabilities = detect_capabilities(dataset["path"])
    modes = candidate_modes(config, dataset_kind, capabilities)

    final_dir = Path(config["root_dir"]) / "semantics" / "results" / f"{dataset_name}_{args.output_suffix}"
    report_path = Path(config["external_output_root"]) / f"{dataset_name}_{args.output_suffix}" / "auto_degrade_report.json"
    if final_dir.exists() and not args.dry_run:
        shutil.rmtree(final_dir)
    report = {
        "dataset_key": args.dataset_key,
        "dataset_name": dataset_name,
        "dataset_kind": dataset_kind,
        "capabilities": capabilities,
        "candidate_modes": modes,
        "attempts": [],
        "selected": None,
        "final_json_dir": str(final_dir),
        "report_path": str(report_path),
    }

    print(f"[auto_degrade] dataset={args.dataset_key}")
    print(f"[auto_degrade] capabilities={capabilities}")
    print(f"[auto_degrade] candidates={modes}")

    if args.dry_run:
        write_json(report_path, report)
        return

    for mode in modes:
        attempt = run_pipeline(args, config, args.dataset_key, dataset_name, mode)
        if attempt["status"] == "ok":
            quality_ok, quality_reason = validate_attempt_quality(attempt, args)
            attempt["quality_status"] = "ok" if quality_ok else "failed"
            attempt["quality_reason"] = quality_reason
            if not quality_ok:
                attempt["status"] = "failed"
        report["attempts"].append(attempt)
        print(f"[auto_degrade] mode={mode} status={attempt['status']}")
        if attempt["status"] == "ok":
            report["selected"] = attempt
            copy_selected_outputs(attempt, final_dir)
            break

    if report["selected"] is None:
        write_json(report_path, report)
        raise RuntimeError("All detected modes failed; no usable navigation JSON was produced.")

    write_json(report_path, report)
    print(f"[auto_degrade] selected_mode={report['selected']['mode']}")
    print(f"[auto_degrade] final_json_dir={final_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[auto_degrade] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
