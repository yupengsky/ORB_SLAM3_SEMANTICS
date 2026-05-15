#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path
from statistics import median

import cv2
import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_pts_ms(path):
    values = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    return values


def find_camchain(kalibr_dir):
    kalibr_dir = Path(kalibr_dir)
    candidates = sorted(kalibr_dir.glob("camchain-imucam*.yaml"))
    if candidates:
        return candidates[0]
    path = kalibr_dir / "camchain.yaml"
    if path.exists():
        return path
    raise FileNotFoundError(f"No camchain YAML found in {kalibr_dir}")


def camera_calibration(kalibr_dir):
    camchain = read_yaml(find_camchain(kalibr_dir))
    cam0 = camchain["cam0"]
    return {
        "intrinsics": [float(value) for value in cam0["intrinsics"]],
        "distortion": [float(value) for value in cam0["distortion_coeffs"]],
        "resolution": [int(value) for value in cam0["resolution"]],
    }


def camera_records(source_dir):
    source_dir = Path(source_dir)
    metadata_path = source_dir / "camera_metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        records = []
        for index, item in enumerate(metadata):
            timestamp = item.get("SensorTimestamp")
            if timestamp is not None:
                records.append({"frame_index": index, "timestamp_ns": int(timestamp)})
        if records:
            return records

    pts = read_pts_ms(source_dir / "camera_pts.txt")
    return [
        {"frame_index": index, "timestamp_ns": int(round(value * 1_000_000.0))}
        for index, value in enumerate(pts)
    ]


def selected_records(records, start_s, duration_s, frame_stride):
    if not records:
        return []
    base = records[0]["timestamp_ns"]
    start_ns = base + int(round(start_s * 1_000_000_000))
    end_ns = start_ns + int(round(duration_s * 1_000_000_000))
    window = [
        record
        for record in records
        if start_ns <= record["timestamp_ns"] < end_ns
    ]
    stride = max(1, int(frame_stride))
    return [record for index, record in enumerate(window) if index % stride == 0]


def reset_dir(path):
    path = Path(path).expanduser().resolve()
    if str(path) in {"/", str(Path.home())}:
        raise RuntimeError(f"Refusing to reset unsafe path: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path, output_dir, records):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {record["frame_index"]: record for record in records}
    if not selected:
        raise RuntimeError("No camera frames selected.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    max_index = max(selected)
    index = 0
    while index <= max_index:
        ok, frame = cap.read()
        if not ok:
            break
        record = selected.get(index)
        if record is not None:
            output = output_dir / f"{record['timestamp_ns']}.png"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Could not write image: {output}")
            saved += 1
        index += 1

    cap.release()
    return [
        record
        for record in records
        if (output_dir / f"{record['timestamp_ns']}.png").exists()
    ], len(records) - saved


def estimate_fps(records):
    deltas = [
        (right["timestamp_ns"] - left["timestamp_ns"]) / 1_000_000_000.0
        for left, right in zip(records, records[1:])
        if right["timestamp_ns"] > left["timestamp_ns"]
    ]
    if not deltas:
        return 1.0
    return 1.0 / median(deltas)


def write_times(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(f"{record['timestamp_ns']}\n")


def write_settings(path, calibration, fps):
    fx, fy, cx, cy = calibration["intrinsics"]
    k1, k2, p1, p2 = calibration["distortion"][:4]
    width, height = calibration["resolution"]
    lines = [
        "%YAML:1.0",
        "",
        'File.version: "1.0"',
        'Camera.type: "PinHole"',
        "",
        f"Camera1.fx: {fx:.12g}",
        f"Camera1.fy: {fy:.12g}",
        f"Camera1.cx: {cx:.12g}",
        f"Camera1.cy: {cy:.12g}",
        f"Camera1.k1: {k1:.12g}",
        f"Camera1.k2: {k2:.12g}",
        f"Camera1.p1: {p1:.12g}",
        f"Camera1.p2: {p2:.12g}",
        "",
        f"Camera.width: {width}",
        f"Camera.height: {height}",
        f"Camera.fps: {max(1, int(round(fps)))}",
        "Camera.RGB: 1",
        "",
        "ORBextractor.nFeatures: 1500",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20",
        "ORBextractor.minThFAST: 7",
        "",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -3.5",
        "Viewer.ViewpointF: 500.0",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Package the Zero2W H.264 camera stream for mono ORB-SLAM3.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--kalibr-dir", required=True)
    parser.add_argument("--package-out", required=True)
    parser.add_argument("--start", type=float, default=180.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    package_out = Path(args.package_out).expanduser().resolve()
    reset_dir(package_out)

    records = selected_records(camera_records(source_dir), args.start, args.duration, args.frame_stride)
    records, missing = extract_frames(source_dir / "camera.h264", package_out / "mav0" / "cam0" / "data", records)
    if not records:
        raise RuntimeError("No decodable frames were packaged.")

    write_times(package_out / "times.txt", records)
    write_settings(package_out / "ORB_SLAM3_mono.yaml", camera_calibration(args.kalibr_dir), estimate_fps(records))
    duration = (records[-1]["timestamp_ns"] - records[0]["timestamp_ns"]) / 1_000_000_000.0
    print(f"[package_zero2w_mono] frames={len(records)} missing_tail_frames={missing} duration_s={duration:.3f}")


if __name__ == "__main__":
    main()
