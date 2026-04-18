#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
from bisect import bisect_right
from pathlib import Path


DEFAULT_T_B_C1 = (
    (0.999976337909, -0.004066386342, -0.005548704675, 0.009253976345),
    (-0.004079205043, -0.999989033012, -0.002300856704, 0.075519914257),
    (-0.005539287650, 0.002323436565, -0.999981958805, -0.005770993238),
    (0.0, 0.0, 0.0, 1.0),
)


def read_csv_floats(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                rows.append([float(value) for value in row])
    return rows


def copy_tree_subset(source, target):
    source = Path(source)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    for relative in (
        "iphone/frames.mov",
        "iphone/frames.csv",
        "iphone/gyro.csv",
        "iphone/accelerometer.csv",
        "iphone/arkit.csv",
        "ground-truth/pose.csv",
        "ground-truth/fixpoints.csv",
        "tango/frames.mov",
        "tango/frames.csv",
    ):
        src = source / relative
        if not src.exists():
            continue
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)


def sec_to_ns(seconds):
    return int(round(float(seconds) * 1_000_000_000))


def write_settings(path, fps):
    matrix = ", ".join(f"{value:.12g}" for row in DEFAULT_T_B_C1 for value in row)
    text = "\n".join(
        [
            "%YAML:1.0",
            "",
            'File.version: "1.0"',
            'Camera.type: "PinHole"',
            "",
            "Camera1.fx: 1082.4",
            "Camera1.fy: 1084.4",
            "Camera1.cx: 364.6778",
            "Camera1.cy: 643.3080",
            "Camera1.k1: 0.0366",
            "Camera1.k2: 0.0803",
            "Camera1.k3: 0.0",
            "Camera1.p1: 0.000783",
            "Camera1.p2: -0.000215",
            "",
            "Camera.width: 720",
            "Camera.height: 1280",
            "Camera.newWidth: 720",
            "Camera.newHeight: 1280",
            f"Camera.fps: {int(round(fps))}",
            "Camera.RGB: 1",
            "",
            "IMU.T_b_c1: !!opencv-matrix",
            "   rows: 4",
            "   cols: 4",
            "   dt: f",
            f"   data: [{matrix}]",
            "",
            "IMU.NoiseGyro: 0.0024",
            "IMU.NoiseAcc: 0.0048",
            "IMU.GyroWalk: 0.000051",
            "IMU.AccWalk: 0.00021",
            "IMU.Frequency: 100.0",
            "",
            "ORBextractor.nFeatures: 1200",
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
    )
    Path(path).write_text(text, encoding="utf-8")


def extract_frames(video_path, output_dir, all_frame_rows, selected_frame_numbers, force=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_frame_numbers = set(selected_frame_numbers)
    expected = len(selected_frame_numbers)
    existing = list(output_dir.glob("*.png"))
    if len(existing) == expected and not force:
        return False

    if force and output_dir.exists():
        for path in output_dir.glob("*.png"):
            path.unlink()

    tmp_dir = output_dir.parent / ".extract_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-start_number",
            "1",
            str(tmp_dir / "frame_%06d.png"),
        ],
        check=True,
    )

    extracted = sorted(tmp_dir.glob("frame_*.png"))
    if len(extracted) != len(all_frame_rows):
        raise RuntimeError(f"Extracted {len(extracted)} frames, expected {len(all_frame_rows)}.")

    for frame_number, (src, row) in enumerate(zip(extracted, all_frame_rows), start=1):
        if frame_number not in selected_frame_numbers:
            continue
        timestamp_ns = sec_to_ns(row[0])
        src.rename(output_dir / f"{timestamp_ns}.png")

    shutil.rmtree(tmp_dir)
    return True


def write_timestamps(path, frame_rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in frame_rows:
            f.write(f"{sec_to_ns(row[0])}\n")


def interpolate_value(rows, timestamp):
    times = [row[0] for row in rows]
    index = bisect_right(times, timestamp)
    if index <= 0:
        return rows[0][1:4]
    if index >= len(rows):
        return rows[-1][1:4]

    left = rows[index - 1]
    right = rows[index]
    denom = right[0] - left[0]
    if abs(denom) < 1e-12:
        return left[1:4]
    alpha = (timestamp - left[0]) / denom
    return [left[i] + alpha * (right[i] - left[i]) for i in range(1, 4)]


def write_imu(path, gyro_rows, acc_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        last_timestamp_ns = None
        for gyro in gyro_rows:
            timestamp = gyro[0]
            timestamp_ns = sec_to_ns(timestamp)
            if timestamp_ns == last_timestamp_ns:
                continue
            acc = interpolate_value(acc_rows, timestamp)
            f.write(
                f"{timestamp_ns},{gyro[1]:.9f},{gyro[2]:.9f},{gyro[3]:.9f},"
                f"{acc[0]:.9f},{acc[1]:.9f},{acc[2]:.9f}\n"
            )
            last_timestamp_ns = timestamp_ns


def package_advio(args):
    source_seq = Path(args.source_seq).expanduser().resolve()
    raw_out = Path(args.raw_out).expanduser().resolve()
    package_out = Path(args.package_out).expanduser().resolve()
    if not source_seq.is_dir():
        raise FileNotFoundError(f"ADVIO source sequence not found: {source_seq}")

    copy_tree_subset(source_seq, raw_out)
    source_for_packaging = raw_out

    frame_stride = max(1, int(args.frame_stride))
    all_frame_rows = read_csv_floats(source_for_packaging / "iphone" / "frames.csv")
    frame_rows = [row for index, row in enumerate(all_frame_rows) if index % frame_stride == 0]
    selected_frame_numbers = [index + 1 for index in range(len(all_frame_rows)) if index % frame_stride == 0]
    gyro_rows = read_csv_floats(source_for_packaging / "iphone" / "gyro.csv")
    acc_rows = read_csv_floats(source_for_packaging / "iphone" / "accelerometer.csv")

    cam_data_dir = package_out / "mav0" / "cam0" / "data"
    extract_frames(source_for_packaging / "iphone" / "frames.mov", cam_data_dir, all_frame_rows, selected_frame_numbers, force=args.force)
    write_timestamps(package_out / "times.txt", frame_rows)
    write_imu(package_out / "mav0" / "imu0" / "data.csv", gyro_rows, acc_rows)
    write_settings(package_out / "ADVIO_iphone_mono_inertial.yaml", fps=60.0 / frame_stride)

    summary = {
        "source_sequence": str(source_seq),
        "raw_advio_copy": str(raw_out),
        "packaged_sequence": str(package_out),
        "camera": "iphone",
        "format": "euroc_like_mono_inertial",
        "frame_stride": frame_stride,
        "source_frames": len(all_frame_rows),
        "frames": len(frame_rows),
        "gyro_rows": len(gyro_rows),
        "accelerometer_rows": len(acc_rows),
        "image_width": 720,
        "image_height": 1280,
        "times_file": "times.txt",
        "settings_file": "ADVIO_iphone_mono_inertial.yaml",
        "imu_file": "mav0/imu0/data.csv",
    }
    (package_out / "packaging_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Package ADVIO iPhone video + IMU as an ORB-SLAM3 EuRoC-like mono-inertial sequence.")
    parser.add_argument("--source-seq", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--package-out", required=True)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    summary = package_advio(parse_args())
    print(
        "[package_advio] finished: "
        f"{summary['frames']} frames, raw={summary['raw_advio_copy']}, packaged={summary['packaged_sequence']}"
    )


if __name__ == "__main__":
    main()
