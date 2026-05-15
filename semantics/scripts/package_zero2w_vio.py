#!/usr/bin/env python3
import argparse
import csv
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


def safe_remove_tree(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return
    if str(path) in {"/", str(Path.home())}:
        raise RuntimeError(f"Refusing to remove unsafe path: {path}")
    shutil.rmtree(path)


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(value)
    return number


def invert_rigid_transform(matrix):
    rotation = [row[:3] for row in matrix[:3]]
    translation = [matrix[row][3] for row in range(3)]
    rotation_t = [[rotation[col][row] for col in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(rotation_t[row][col] * translation[col] for col in range(3))
        for row in range(3)
    ]
    return [
        [*rotation_t[0], inverse_translation[0]],
        [*rotation_t[1], inverse_translation[1]],
        [*rotation_t[2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def flatten_matrix(matrix):
    return [value for row in matrix for value in row]


def find_default_camchain(kalibr_dir):
    kalibr_dir = Path(kalibr_dir)
    candidates = sorted(kalibr_dir.glob("camchain-imucam*.yaml"))
    if candidates:
        return candidates[0]
    fallback = kalibr_dir / "camchain.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No camchain YAML found in {kalibr_dir}")


def load_calibration(kalibr_dir, camchain_path=None, imu_path=None):
    kalibr_dir = Path(kalibr_dir).expanduser().resolve()
    camchain_path = Path(camchain_path).expanduser().resolve() if camchain_path else find_default_camchain(kalibr_dir)
    imu_path = Path(imu_path).expanduser().resolve() if imu_path else kalibr_dir / "imu.yaml"

    camchain = read_yaml(camchain_path)
    imu = read_yaml(imu_path)
    cam0 = camchain["cam0"]
    intrinsics = [float(value) for value in cam0["intrinsics"]]
    distortion = [float(value) for value in cam0["distortion_coeffs"]]
    resolution = [int(value) for value in cam0["resolution"]]
    t_cam_imu = [[float(value) for value in row] for row in cam0["T_cam_imu"]]
    t_imu_cam = invert_rigid_transform(t_cam_imu)
    timeshift = float(cam0.get("timeshift_cam_imu", 0.0))

    return {
        "camchain_path": str(camchain_path),
        "imu_path": str(imu_path),
        "intrinsics": intrinsics,
        "distortion": distortion,
        "resolution": resolution,
        "T_cam_imu": t_cam_imu,
        "T_imu_cam": t_imu_cam,
        "timeshift_cam_imu_s": timeshift,
        "imu_frequency_hz": float(imu.get("update_rate", 0.0) or 0.0),
        "noise_gyro": float(imu.get("gyroscope_noise_density", 0.001)),
        "noise_acc": float(imu.get("accelerometer_noise_density", 0.02)),
        "gyro_walk": float(imu.get("gyroscope_random_walk", 5.0e-05)),
        "acc_walk": float(imu.get("accelerometer_random_walk", 0.0005)),
    }


def read_pts_ms(path):
    values = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line))
    return values


def read_camera_records(source_dir):
    source_dir = Path(source_dir)
    metadata_path = source_dir / "camera_metadata.json"
    pts_path = source_dir / "camera_pts.txt"
    pts_ms = read_pts_ms(pts_path) if pts_path.exists() else []

    if metadata_path.exists():
        metadata = read_json(metadata_path)
        records = []
        for index, item in enumerate(metadata):
            timestamp = item.get("SensorTimestamp")
            if timestamp is None:
                timestamp = item.get("FrameWallClock")
            if timestamp is None:
                continue
            record = {
                "frame_index": index,
                "timestamp_ns_raw": int(timestamp),
            }
            if index < len(pts_ms):
                record["pts_ms"] = pts_ms[index]
            records.append(record)
        if records:
            return records, "camera_metadata.json SensorTimestamp"

    if not pts_ms:
        raise FileNotFoundError(f"No camera metadata or PTS found in {source_dir}")

    records = [
        {
            "frame_index": index,
            "timestamp_ns_raw": int(round(value * 1_000_000.0)),
            "pts_ms": value,
        }
        for index, value in enumerate(pts_ms)
    ]
    return records, "camera_pts.txt relative ms"


def select_camera_records(records, start_s, duration_s, frame_stride):
    first_timestamp_ns = records[0]["timestamp_ns_raw"]
    start_ns = first_timestamp_ns + int(round(start_s * 1_000_000_000))
    end_ns = None if duration_s <= 0 else start_ns + int(round(duration_s * 1_000_000_000))
    candidates = [
        record
        for record in records
        if record["timestamp_ns_raw"] >= start_ns and (end_ns is None or record["timestamp_ns_raw"] < end_ns)
    ]
    stride = max(1, int(frame_stride))
    return [record for index, record in enumerate(candidates) if index % stride == 0]


def extract_selected_frames(video_path, output_dir, selected_records, force):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_by_index = {int(record["frame_index"]): record for record in selected_records}
    expected = len(selected_by_index)
    if expected == 0:
        raise RuntimeError("No frames selected for extraction.")

    existing = list(output_dir.glob("*.png"))
    if len(existing) == expected and not force:
        return len(existing), 0, "reused"

    for path in existing:
        path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    decoded = 0
    max_index = max(selected_by_index)
    frame_index = 0
    while frame_index <= max_index:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        record = selected_by_index.get(frame_index)
        if record is not None:
            output_path = output_dir / f"{record['timestamp_ns']}.png"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not write frame: {output_path}")
            saved += 1
        frame_index += 1

    cap.release()
    missing = expected - saved
    if saved == 0:
        raise RuntimeError(f"No selected frames were decoded from {video_path}")
    return saved, missing, f"decoded_until_frame_index={frame_index - 1}, decoded={decoded}"


def write_timestamps(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(f"{record['timestamp_ns']}\n")


def write_camera_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("#timestamp [ns],filename\n")
        for record in records:
            f.write(f"{record['timestamp_ns']},{record['timestamp_ns']}.png\n")


def write_imu(source_csv, output_csv, start_ns, end_ns, padding_s):
    source_csv = Path(source_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pad_ns = int(round(padding_s * 1_000_000_000))
    lower = start_ns - pad_ns
    upper = end_ns + pad_ns

    rows_written = 0
    rows_total = 0
    rows_nonfinite = 0
    rows_outside = 0
    last_timestamp = None

    with source_csv.open("r", encoding="utf-8") as src, output_csv.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        dst.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for row in reader:
            rows_total += 1
            try:
                timestamp_ns = int(row["monotonic_time_ns"])
                acc_x = finite_float(row["acc_x_mps2"])
                acc_y = finite_float(row["acc_y_mps2"])
                acc_z = finite_float(row["acc_z_mps2"])
                gyro_x = finite_float(row["gyro_x_rads"])
                gyro_y = finite_float(row["gyro_y_rads"])
                gyro_z = finite_float(row["gyro_z_rads"])
            except (KeyError, TypeError, ValueError):
                rows_nonfinite += 1
                continue

            if timestamp_ns < lower or timestamp_ns > upper:
                rows_outside += 1
                continue
            if timestamp_ns == last_timestamp:
                continue

            dst.write(
                f"{timestamp_ns},{gyro_x:.12g},{gyro_y:.12g},{gyro_z:.12g},"
                f"{acc_x:.12g},{acc_y:.12g},{acc_z:.12g}\n"
            )
            last_timestamp = timestamp_ns
            rows_written += 1

    return {
        "source_rows": rows_total,
        "written_rows": rows_written,
        "skipped_nonfinite_rows": rows_nonfinite,
        "skipped_outside_window_rows": rows_outside,
        "padding_s": padding_s,
    }


def write_settings(path, calibration, fps):
    fx, fy, cx, cy = calibration["intrinsics"]
    k1, k2, p1, p2 = calibration["distortion"][:4]
    width, height = calibration["resolution"]
    matrix = ", ".join(f"{value:.12g}" for value in flatten_matrix(calibration["T_imu_cam"]))
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
        "IMU.T_b_c1: !!opencv-matrix",
        "   rows: 4",
        "   cols: 4",
        "   dt: f",
        f"   data: [{matrix}]",
        "",
        f"IMU.NoiseGyro: {calibration['noise_gyro']:.12g}",
        f"IMU.NoiseAcc: {calibration['noise_acc']:.12g}",
        f"IMU.GyroWalk: {calibration['gyro_walk']:.12g}",
        f"IMU.AccWalk: {calibration['acc_walk']:.12g}",
        f"IMU.Frequency: {calibration['imu_frequency_hz']:.12g}",
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


def estimate_fps(records):
    if len(records) < 2:
        return 1.0
    deltas = [
        (records[index]["timestamp_ns"] - records[index - 1]["timestamp_ns"]) / 1_000_000_000.0
        for index in range(1, len(records))
        if records[index]["timestamp_ns"] > records[index - 1]["timestamp_ns"]
    ]
    if not deltas:
        return 1.0
    return 1.0 / median(deltas)


def package_zero2w_vio(args):
    source_dir = Path(args.source_dir).expanduser().resolve()
    kalibr_dir = Path(args.kalibr_dir).expanduser().resolve()
    package_out = Path(args.package_out).expanduser().resolve()
    if args.force:
        safe_remove_tree(package_out)
    package_out.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(kalibr_dir, args.camchain, args.imu_yaml)
    camera_records, timestamp_source = read_camera_records(source_dir)
    selected = select_camera_records(camera_records, args.start, args.duration, args.frame_stride)
    shift_ns = int(round(calibration["timeshift_cam_imu_s"] * 1_000_000_000)) if args.apply_timeshift else 0
    for record in selected:
        record["timestamp_ns"] = int(record["timestamp_ns_raw"]) + shift_ns

    cam_data_dir = package_out / "mav0" / "cam0" / "data"
    saved, missing, decode_note = extract_selected_frames(source_dir / "camera.h264", cam_data_dir, selected, args.force)
    selected = [
        record
        for record in selected
        if (cam_data_dir / f"{record['timestamp_ns']}.png").exists()
    ]
    if not selected:
        raise RuntimeError("No packaged frames exist after extraction.")

    write_timestamps(package_out / "times.txt", selected)
    write_camera_csv(package_out / "mav0" / "cam0" / "data.csv", selected)
    imu_stats = write_imu(
        source_dir / "imu.csv",
        package_out / "mav0" / "imu0" / "data.csv",
        selected[0]["timestamp_ns"],
        selected[-1]["timestamp_ns"],
        args.imu_padding,
    )
    fps = estimate_fps(selected)
    write_settings(package_out / "ADVIO_iphone_mono_inertial.yaml", calibration, fps)

    duration_s = (selected[-1]["timestamp_ns"] - selected[0]["timestamp_ns"]) / 1_000_000_000.0 if len(selected) > 1 else 0.0
    summary = {
        "source_dir": str(source_dir),
        "kalibr_dir": str(kalibr_dir),
        "packaged_sequence": str(package_out),
        "format": "euroc_like_mono_inertial",
        "timestamp_source": timestamp_source,
        "timeshift_cam_imu_s": calibration["timeshift_cam_imu_s"],
        "timeshift_applied_to_camera_timestamps": bool(args.apply_timeshift),
        "frame_stride": int(args.frame_stride),
        "start_s": float(args.start),
        "requested_duration_s": float(args.duration),
        "packaged_duration_s": duration_s,
        "camera_records_total": len(camera_records),
        "selected_records_before_decode": saved + missing,
        "frames": len(selected),
        "missing_selected_video_frames": missing,
        "decode_note": decode_note,
        "image_width": calibration["resolution"][0],
        "image_height": calibration["resolution"][1],
        "estimated_fps": fps,
        "times_file": "times.txt",
        "settings_file": "ADVIO_iphone_mono_inertial.yaml",
        "imu_file": "mav0/imu0/data.csv",
        "imu": imu_stats,
        "calibration": {
            "camchain_path": calibration["camchain_path"],
            "imu_path": calibration["imu_path"],
            "T_b_c1_source": "inverse of Kalibr T_cam_imu",
        },
    }
    (package_out / "packaging_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Package Raspberry Pi Zero 2 W VIO H.264 + IMU data for ORB-SLAM3 mono-inertial examples.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--kalibr-dir", required=True)
    parser.add_argument("--package-out", required=True)
    parser.add_argument("--camchain", default="")
    parser.add_argument("--imu-yaml", default="")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in camera stream seconds.")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds; 0 means until the end.")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--imu-padding", type=float, default=1.0)
    parser.add_argument("--apply-timeshift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    summary = package_zero2w_vio(parse_args())
    print(
        "[package_zero2w_vio] finished: "
        f"{summary['frames']} frames, imu={summary['imu']['written_rows']} rows, "
        f"packaged={summary['packaged_sequence']}"
    )


if __name__ == "__main__":
    main()
