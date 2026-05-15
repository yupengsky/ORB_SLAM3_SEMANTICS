#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def color_for_label(label):
    if label == "unknown":
        return 160, 160, 160
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return 80 + digest[0] % 176, 80 + digest[1] % 176, 80 + digest[2] % 176


def camera_center_from_keyframe(keyframe):
    twc = keyframe.get("Twc")
    if not twc or len(twc) != 16:
        return None
    return [float(twc[3]), float(twc[7]), float(twc[11])]


def load_points(export_dir, semantic_map):
    if semantic_map and Path(semantic_map).exists():
        semantic = read_json(semantic_map)
        return semantic.get("points", []), str(Path(semantic_map))

    points = []
    for point in read_jsonl(Path(export_dir) / "map_points.jsonl"):
        points.append(
            {
                "map_point_id": int(point["map_point_id"]),
                "position": point["position"],
                "label": "unknown",
                "score": 0.0,
                "semantic_observation_hits": 0,
                "orb_observations": int(point.get("observations", 0)),
            }
        )
    return points, None


def build_history(export_dir, semantic_map=None, metadata=None):
    export_dir = Path(export_dir)
    summary = read_json(export_dir / "summary.json") if (export_dir / "summary.json").exists() else {}
    keyframes = read_jsonl(export_dir / "keyframes.jsonl")
    observations = read_jsonl(export_dir / "observations.jsonl")
    map_points = read_jsonl(export_dir / "map_points.jsonl")

    keyframes = sorted(keyframes, key=lambda item: (float(item.get("timestamp", 0.0)), int(item.get("keyframe_id", 0))))
    keyframe_index = {int(kf["keyframe_id"]): index for index, kf in enumerate(keyframes)}
    keyframes_by_id = {int(kf["keyframe_id"]): kf for kf in keyframes}

    frames = []
    for index, kf in enumerate(keyframes):
        frames.append(
            {
                "frame_index": index,
                "keyframe_id": int(kf["keyframe_id"]),
                "frame_id": int(kf.get("frame_id", -1)),
                "timestamp": float(kf.get("timestamp", 0.0)),
                "timestamp_ns": int(float(kf.get("timestamp_ns", 0))),
                "image_name": kf.get("image_name", ""),
                "image_path": kf.get("image_path", ""),
                "camera_center": camera_center_from_keyframe(kf),
            }
        )

    first_seen = {}
    observation_counts = defaultdict(int)
    for obs in observations:
        point_id = int(obs["map_point_id"])
        kf_id = int(obs["keyframe_id"])
        if kf_id not in keyframe_index:
            continue
        index = keyframe_index[kf_id]
        observation_counts[point_id] += 1
        if point_id not in first_seen or index < first_seen[point_id]["frame_index"]:
            first_seen[point_id] = {
                "frame_index": index,
                "keyframe_id": kf_id,
            }

    fallback_first_kf = {
        int(point["map_point_id"]): int(point.get("first_keyframe_id", -1))
        for point in map_points
    }
    source_points, semantic_source = load_points(export_dir, semantic_map)
    history_points = []
    for point in source_points:
        point_id = int(point["map_point_id"])
        first = first_seen.get(point_id)
        if first is None:
            kf_id = fallback_first_kf.get(point_id, -1)
            if kf_id in keyframe_index:
                first = {"frame_index": keyframe_index[kf_id], "keyframe_id": kf_id}
            else:
                first = {"frame_index": 0, "keyframe_id": frames[0]["keyframe_id"] if frames else -1}
        frame = frames[first["frame_index"]] if frames else {}
        label = str(point.get("label", "unknown"))
        r, g, b = color_for_label(label)
        history_points.append(
            {
                "map_point_id": point_id,
                "position": [float(value) for value in point["position"]],
                "first_seen_frame_index": int(first["frame_index"]),
                "first_seen_keyframe_id": int(first["keyframe_id"]),
                "first_seen_timestamp": float(frame.get("timestamp", 0.0)),
                "first_seen_timestamp_ns": int(frame.get("timestamp_ns", 0)),
                "label": label,
                "score": float(point.get("score", 0.0)),
                "semantic_observation_hits": int(point.get("semantic_observation_hits", 0)),
                "orb_observations": int(point.get("orb_observations", point.get("observations", observation_counts.get(point_id, 0)))),
                "color_rgb": [r, g, b],
            }
        )

    history_points.sort(key=lambda point: (point["first_seen_frame_index"], point["map_point_id"]))
    return {
        "schema": "orb_slam3_semantics_pointcloud_history_v1",
        "description": "Final surviving ORB-SLAM3 map points annotated with the earliest exported keyframe that observed each point.",
        "export_dir": str(export_dir),
        "semantic_map": semantic_source,
        "metadata": metadata or {},
        "summary": {
            "frames_total": len(frames),
            "points_total": len(history_points),
            "observations_total": len(observations),
            "source_keyframes_exported": int(summary.get("keyframes_exported", len(frames))),
            "source_map_points_exported": int(summary.get("map_points_exported", len(history_points))),
            "source_observations_exported": int(summary.get("observations_exported", len(observations))),
        },
        "frames": frames,
        "points": history_points,
    }


def write_history_ply(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int first_seen_frame_index\n")
        f.write("end_header\n")
        for point in points:
            x, y, z = point["position"]
            r, g, b = point["color_rgb"]
            f.write(f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b} {point['first_seen_frame_index']}\n")


def parse_metadata(values):
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Metadata must be key=value, got: {value}")
        key, raw = value.split("=", 1)
        metadata[key] = raw
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Build a replayable point-cloud history from ORB-SLAM3 semantic export files.")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--semantic-map", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-ply", default="")
    parser.add_argument("--metadata", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    history = build_history(args.export_dir, args.semantic_map, parse_metadata(args.metadata))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.history_ply:
        write_history_ply(args.history_ply, history["points"])
    print(
        "[pointcloud_history] finished: "
        f"{history['summary']['frames_total']} frames, "
        f"{history['summary']['points_total']} points, output={output}"
    )


if __name__ == "__main__":
    main()
