#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bool_arg(enabled, name):
    return f"--{name}" if enabled else f"--no-{name}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the configured ORB-SLAM3 semantic test suite.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "dataset_config.json"))
    parser.add_argument("--run-slam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-yolo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-navigation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-to-chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-chunked", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline-max-keyframes", type=int, default=0)
    return parser.parse_args()


def default_output_paths(config, dataset_name, mode):
    root_dir = Path(config["root_dir"])
    output_root = Path(config["external_output_root"])
    name = f"{dataset_name}_{mode}"
    return {
        "result_dir": str(output_root / name),
        "final_json_dir": str(root_dir / "semantics" / "results" / name),
        "navigation_json": str(root_dir / "semantics" / "results" / name / "semantic_navigation_map.json"),
    }


def main():
    args = parse_args()
    config = read_json(args.config)
    suite = config.get("test_suite", [])
    if not suite:
        raise RuntimeError("Config has no test_suite entries.")

    pipeline = Path(__file__).resolve().parent / "offline_pipeline.py"
    results = []
    total = sum(len(item.get("modes", [])) for item in suite)
    index = 0

    for item in suite:
        dataset_key = item["dataset_key"]
        dataset = config["datasets"][dataset_key]
        for mode in item.get("modes", []):
            index += 1
            print(f"[suite] run {index}/{total}: dataset={dataset_key} mode={mode}")
            cmd = [
                sys.executable,
                str(pipeline),
                "--config",
                args.config,
                "--dataset-key",
                dataset_key,
                "--slam-mode",
                mode,
                bool_arg(args.run_slam, "run-slam"),
                bool_arg(args.run_yolo, "run-yolo"),
                bool_arg(args.run_navigation, "run-navigation"),
                bool_arg(args.fallback_to_chunks, "fallback-to-chunks"),
                bool_arg(args.force_chunked, "force-chunked"),
                "--offline-max-keyframes",
                str(args.offline_max_keyframes),
            ]
            rc = subprocess.run(cmd, check=False).returncode
            paths = default_output_paths(config, dataset.get("name", dataset_key), mode)
            result = {
                "dataset_key": dataset_key,
                "dataset_name": dataset.get("name", dataset_key),
                "slam_mode": mode,
                "status": "ok" if rc == 0 else "failed",
                "return_code": rc,
                **paths,
            }
            results.append(result)
            if rc != 0 and not args.keep_going:
                break
        if results and results[-1]["return_code"] != 0 and not args.keep_going:
            break

    summary = {
        "runs_total": len(results),
        "runs_succeeded": sum(1 for item in results if item["status"] == "ok"),
        "runs_failed": sum(1 for item in results if item["status"] != "ok"),
        "runs": results,
    }
    summary_path = Path(config["root_dir"]) / "semantics" / "results" / "test_suite_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[suite] summary: {summary_path}")

    if summary["runs_failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
