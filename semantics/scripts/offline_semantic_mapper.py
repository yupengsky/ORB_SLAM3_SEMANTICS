#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_summary(export_dir):
    path = Path(export_dir) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def color_for_label(label):
    if label == "unknown":
        return 160, 160, 160
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return 80 + digest[0] % 176, 80 + digest[1] % 176, 80 + digest[2] % 176


def safe_stem(path):
    stem = Path(path).stem if path else "image"
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)


class StereoRectifier:
    def __init__(self, map_x, map_y):
        self.map_x = map_x
        self.map_y = map_y

    @staticmethod
    def _node_real(fs, name, default=None):
        node = fs.getNode(name)
        if node.empty():
            return default
        return float(node.real())

    @staticmethod
    def _node_int(fs, name, default=None):
        value = StereoRectifier._node_real(fs, name, default)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _distortion(fs, prefix):
        values = []
        for name in ("k1", "k2", "p1", "p2", "k3"):
            value = StereoRectifier._node_real(fs, f"{prefix}.{name}", None)
            if value is not None:
                values.append(value)
        return np.array(values, dtype=np.float64)

    @classmethod
    def from_settings(cls, settings_path):
        if settings_path is None:
            return None

        fs = cv2.FileStorage(str(settings_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Could not open settings file: {settings_path}")

        camera_type = fs.getNode("Camera.type").string()
        has_second_camera = not fs.getNode("Camera2.fx").empty()
        if camera_type != "PinHole" or not has_second_camera:
            fs.release()
            return None

        width = cls._node_int(fs, "Camera.width")
        height = cls._node_int(fs, "Camera.height")
        if width is None or height is None:
            fs.release()
            return None

        k1 = np.array(
            [
                [cls._node_real(fs, "Camera1.fx"), 0.0, cls._node_real(fs, "Camera1.cx")],
                [0.0, cls._node_real(fs, "Camera1.fy"), cls._node_real(fs, "Camera1.cy")],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        k2 = np.array(
            [
                [cls._node_real(fs, "Camera2.fx"), 0.0, cls._node_real(fs, "Camera2.cx")],
                [0.0, cls._node_real(fs, "Camera2.fy"), cls._node_real(fs, "Camera2.cy")],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        d1 = cls._distortion(fs, "Camera1")
        d2 = cls._distortion(fs, "Camera2")
        # Match this ORB-SLAM3 checkout's stereo rectification path exactly.
        if d1.shape == d2.shape:
            d2 = d1.copy()
        t_c1_c2 = fs.getNode("Stereo.T_c1_c2").mat().astype(np.float64)
        fs.release()

        t_inv = np.linalg.inv(t_c1_c2)
        r12 = t_inv[:3, :3]
        t12 = t_inv[:3, 3]
        image_size = (width, height)
        r1, _, p1, _, _, _, _ = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            image_size,
            r12,
            t12,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
            newImageSize=image_size,
        )
        map_x, map_y = cv2.initUndistortRectifyMap(k1, d1, r1, p1[:3, :3], image_size, cv2.CV_32F)
        return cls(map_x, map_y)

    def rectify(self, image):
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)


def load_image(path, rectifier):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if rectifier is not None:
        image = rectifier.rectify(image)
    return image


def class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if 0 <= int(class_id) < len(names):
        return str(names[int(class_id)])
    return str(class_id)


def camera_center_from_keyframe(keyframe):
    twc = keyframe.get("Twc")
    if not twc or len(twc) != 16:
        return None
    return np.array([float(twc[3]), float(twc[7]), float(twc[11])], dtype=np.float64)


def map_point_distances(point, observations_by_point, camera_centers_by_kf):
    position = np.array(point["position"], dtype=np.float64)
    distance_from_origin = float(np.linalg.norm(position))
    distances = []
    for obs in observations_by_point.get(int(point["map_point_id"]), []):
        center = camera_centers_by_kf.get(int(obs["keyframe_id"]))
        if center is not None:
            distances.append(float(np.linalg.norm(position - center)))

    if not distances:
        return {
            "distance_m": None,
            "distance_from_world_origin_m": distance_from_origin,
            "observed_distance_min_m": None,
            "observed_distance_mean_m": None,
            "observed_distance_max_m": None,
            "observed_distance_count": 0,
        }

    return {
        "distance_m": float(np.mean(distances)),
        "distance_from_world_origin_m": distance_from_origin,
        "observed_distance_min_m": float(np.min(distances)),
        "observed_distance_mean_m": float(np.mean(distances)),
        "observed_distance_max_m": float(np.max(distances)),
        "observed_distance_count": len(distances),
    }


def vote_observations(result, observations, names, votes, hit_counts):
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return 0

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    mask_h, mask_w = masks.shape[-2:]
    orig_h, orig_w = result.orig_shape
    hits = 0

    for obs in observations:
        u = float(obs["u"])
        v = float(obs["v"])
        if u < 0 or v < 0 or u >= orig_w or v >= orig_h:
            continue

        mx = min(max(int(round(u * (mask_w - 1) / max(orig_w - 1, 1))), 0), mask_w - 1)
        my = min(max(int(round(v * (mask_h - 1) / max(orig_h - 1, 1))), 0), mask_h - 1)

        best_index = None
        best_conf = -1.0
        for idx in range(masks.shape[0]):
            if masks[idx, my, mx] > 0.5 and float(confidences[idx]) > best_conf:
                best_index = idx
                best_conf = float(confidences[idx])

        if best_index is None:
            continue

        label = class_name(names, classes[best_index])
        point_id = int(obs["map_point_id"])
        votes[point_id][label] += best_conf
        hit_counts[point_id] += 1
        hits += 1

    return hits


def write_semantic_ply(path, semantic_points):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(semantic_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in semantic_points:
            r, g, b = color_for_label(point["label"])
            x, y, z = point["position"]
            f.write(f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}\n")


def save_annotated_result(result, keyframe, output_dir):
    annotated = result.plot(boxes=True, masks=True, labels=True, conf=True)
    filename = f"kf_{int(keyframe['keyframe_id']):06d}_{safe_stem(keyframe.get('image_path', 'image'))}.jpg"
    output_path = Path(output_dir) / filename
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not write annotated image: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Offline YOLO-seg voting for ORB-SLAM3 MapPoints.")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-ply", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-keyframes", type=int, default=0)
    parser.add_argument("--annotated-dir", default=None)
    parser.add_argument("--require-single-map", action="store_true")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    summary = read_summary(export_dir)
    if args.require_single_map and summary.get("maps_with_keyframes") != 1:
        raise RuntimeError(f"Expected one complete map, got summary: {summary}")

    keyframes = read_jsonl(export_dir / "keyframes.jsonl")
    map_points = read_jsonl(export_dir / "map_points.jsonl")
    observations = read_jsonl(export_dir / "observations.jsonl")

    keyframes_by_id = {int(kf["keyframe_id"]): kf for kf in keyframes}
    camera_centers_by_kf = {
        int(kf["keyframe_id"]): camera_center
        for kf in keyframes
        for camera_center in [camera_center_from_keyframe(kf)]
        if camera_center is not None
    }
    observations_by_kf = defaultdict(list)
    observations_by_point = defaultdict(list)
    for obs in observations:
        observations_by_kf[int(obs["keyframe_id"])].append(obs)
        observations_by_point[int(obs["map_point_id"])].append(obs)

    process_keyframes = [
        keyframes_by_id[kf_id]
        for kf_id in sorted(observations_by_kf)
        if kf_id in keyframes_by_id and keyframes_by_id[kf_id].get("image_path")
    ]
    if args.max_keyframes > 0:
        process_keyframes = process_keyframes[: args.max_keyframes]

    rectifier = StereoRectifier.from_settings(args.settings)
    model = YOLO(args.model)
    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
        "retina_masks": True,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else None
    manifest = None
    if annotated_dir is not None:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        manifest = (annotated_dir / "manifest.jsonl").open("w", encoding="utf-8")

    votes = defaultdict(lambda: defaultdict(float))
    hit_counts = defaultdict(int)
    processed = 0
    semantic_hits = 0

    try:
        for kf in process_keyframes:
            image = load_image(kf["image_path"], rectifier)
            result = model.predict(source=image, **predict_kwargs)[0]
            if annotated_dir is not None:
                annotated_path = save_annotated_result(result, kf, annotated_dir)
                manifest.write(
                    json.dumps(
                        {
                            "keyframe_id": int(kf["keyframe_id"]),
                            "frame_id": int(kf["frame_id"]),
                            "timestamp": float(kf["timestamp"]),
                            "image_path": kf["image_path"],
                            "annotated_image": str(annotated_path),
                            "detections": int(len(result.boxes) if result.boxes is not None else 0),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            kf_observations = observations_by_kf[int(kf["keyframe_id"])]
            semantic_hits += vote_observations(result, kf_observations, model.names, votes, hit_counts)
            processed += 1
            if processed % 10 == 0:
                print(f"[semantic] processed {processed}/{len(process_keyframes)} keyframes")
    finally:
        if manifest is not None:
            manifest.close()

    semantic_points = []
    labeled_points = 0
    for point in map_points:
        point_id = int(point["map_point_id"])
        label_scores = dict(votes.get(point_id, {}))
        distance_fields = map_point_distances(point, observations_by_point, camera_centers_by_kf)
        if label_scores:
            label = max(label_scores, key=label_scores.get)
            score = float(label_scores[label])
            labeled_points += 1
        else:
            label = "unknown"
            score = 0.0

        semantic_points.append(
            {
                "map_point_id": point_id,
                "position": point["position"],
                **distance_fields,
                "label": label,
                "score": score,
                "semantic_observation_hits": int(hit_counts.get(point_id, 0)),
                "orb_observations": int(point.get("observations", 0)),
                "label_scores": label_scores,
            }
        )

    output = {
        "scene_summary": {
            "keyframes_processed": processed,
            "keyframes_available": len(process_keyframes),
            "partial": args.max_keyframes > 0,
            "map_points_total": len(semantic_points),
            "map_points_labeled": labeled_points,
            "semantic_observation_hits": semantic_hits,
            "single_complete_map": bool(summary.get("single_complete_map", False)),
        },
        "points": semantic_points,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.semantic_ply:
        write_semantic_ply(args.semantic_ply, semantic_points)

    print(
        "[semantic] finished: "
        f"{processed} keyframes, {len(semantic_points)} points, "
        f"{labeled_points} labeled points, {semantic_hits} observation hits"
    )


if __name__ == "__main__":
    main()
