#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


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


def round_float(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def round_vec(values, digits=4):
    return [round_float(value, digits) for value in values]


def as_vec(values):
    return np.array(values, dtype=np.float64)


def safe_id(label):
    value = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return value or "object"


def camera_center_from_keyframe(keyframe):
    twc = keyframe.get("Twc")
    if not twc or len(twc) != 16:
        return None
    return as_vec([twc[3], twc[7], twc[11]])


def vector_norm(values):
    return float(np.linalg.norm(values))


def unit_vector(values):
    norm = vector_norm(values)
    if norm <= 1e-9:
        return [0.0, 0.0, 0.0]
    return round_vec(values / norm)


def map_bounds(vectors):
    if not vectors:
        return None
    coords = np.vstack(vectors)
    return {
        "min": round_vec(coords.min(axis=0)),
        "max": round_vec(coords.max(axis=0)),
        "extent": round_vec(coords.max(axis=0) - coords.min(axis=0)),
    }


def label_counts(points):
    counts = defaultdict(int)
    for point in points:
        counts[str(point.get("label", "unknown"))] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def grid_cell(position, radius):
    return tuple(np.floor(position / radius).astype(np.int64).tolist())


def neighbor_cells(cell):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield cell[0] + dx, cell[1] + dy, cell[2] + dz


def connected_components(points, radius):
    if not points:
        return []

    positions = [as_vec(point["position"]) for point in points]
    cells = defaultdict(list)
    for index, position in enumerate(positions):
        cells[grid_cell(position, radius)].append(index)

    visited = [False] * len(points)
    components = []
    for start in range(len(points)):
        if visited[start]:
            continue

        visited[start] = True
        queue = deque([start])
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)
            current_position = positions[current]
            for cell in neighbor_cells(grid_cell(current_position, radius)):
                for candidate in cells.get(cell, []):
                    if visited[candidate]:
                        continue
                    if vector_norm(current_position - positions[candidate]) <= radius:
                        visited[candidate] = True
                        queue.append(candidate)

        components.append(component)

    return components


def build_semantic_objects(semantic_points, cluster_radius, min_object_points, max_objects_per_label):
    points_by_label = defaultdict(list)
    for point in semantic_points:
        label = str(point.get("label", "unknown"))
        if label == "unknown":
            continue
        if not point.get("position") or len(point["position"]) != 3:
            continue
        points_by_label[label].append(point)

    objects = []
    discarded_clusters = 0
    per_label_index = defaultdict(int)

    for label in sorted(points_by_label):
        label_points = points_by_label[label]
        components = connected_components(label_points, cluster_radius)
        components.sort(key=len, reverse=True)

        kept_for_label = 0
        for component in components:
            if len(component) < min_object_points:
                discarded_clusters += 1
                continue
            if kept_for_label >= max_objects_per_label:
                discarded_clusters += 1
                continue

            member_points = [label_points[index] for index in component]
            coords = np.vstack([as_vec(point["position"]) for point in member_points])
            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)
            semantic_hits = sum(int(point.get("semantic_observation_hits", 0)) for point in member_points)
            point_confidences = []
            for point in member_points:
                hits = max(int(point.get("semantic_observation_hits", 0)), 1)
                point_confidences.append(min(1.0, float(point.get("score", 0.0)) / hits))

            per_label_index[label] += 1
            kept_for_label += 1
            objects.append(
                {
                    "id": f"{safe_id(label)}_{per_label_index[label]:03d}",
                    "label": label,
                    "center": round_vec(coords.mean(axis=0)),
                    "bbox_3d": {
                        "min": round_vec(bbox_min),
                        "max": round_vec(bbox_max),
                    },
                    "extent": round_vec(bbox_max - bbox_min),
                    "confidence": round_float(float(np.mean(point_confidences)) if point_confidences else 0.0),
                    "support_points": len(member_points),
                    "semantic_observation_hits": semantic_hits,
                }
            )

    objects.sort(key=lambda item: (item["label"], item["id"]))
    return objects, discarded_clusters


def build_traversable_path_network(keyframes, node_radius, nearby_radius, objects):
    valid = []
    for keyframe in sorted(keyframes, key=lambda kf: (float(kf.get("timestamp", 0.0)), int(kf["keyframe_id"]))):
        center = camera_center_from_keyframe(keyframe)
        if center is not None:
            valid.append(center)

    if not valid:
        return {
            "nodes": [],
            "edges": [],
            "path_network_length_m": 0.0,
            "recorded_trajectory_length_m": 0.0,
        }

    recorded_trajectory_length = sum(vector_norm(curr - prev) for prev, curr in zip(valid, valid[1:]))
    node_accumulators = []
    assignments = []

    for position in valid:
        best_index = None
        best_distance = math.inf
        for index, node in enumerate(node_accumulators):
            center = node["sum"] / node["count"]
            distance = vector_norm(position - center)
            if distance <= node_radius and distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is None:
            best_index = len(node_accumulators)
            node_accumulators.append({"sum": position.copy(), "count": 1})
        else:
            node_accumulators[best_index]["sum"] += position
            node_accumulators[best_index]["count"] += 1
        assignments.append(best_index)

    node_positions = [node["sum"] / node["count"] for node in node_accumulators]
    edge_stats = {}
    observed_directions = defaultdict(set)
    for previous, current in zip(assignments, assignments[1:]):
        if previous == current:
            continue
        pair = tuple(sorted((previous, current)))
        edge_stats[pair] = edge_stats.get(pair, 0) + 1
        observed_directions[pair].add((previous, current))

    object_centers = [(obj["id"], obj["label"], as_vec(obj["center"])) for obj in objects]
    nodes = []
    for index, position in enumerate(node_positions):
        nearby_objects = []
        for object_id, label, object_center in object_centers:
            distance = vector_norm(object_center - position)
            if distance <= nearby_radius:
                nearby_objects.append(
                    {
                        "object_id": object_id,
                        "label": label,
                        "distance_m": round_float(distance),
                    }
                )
        nearby_objects.sort(key=lambda item: (item["distance_m"], item["object_id"]))
        nodes.append(
            {
                "id": f"path_node_{index + 1:03d}",
                "position": round_vec(position),
                "observed_keyframes_count": int(node_accumulators[index]["count"]),
                "nearby_objects": nearby_objects[:12],
            }
        )

    edges = []
    for edge_index, ((left, right), traversal_count) in enumerate(sorted(edge_stats.items()), start=1):
        left_position = node_positions[left]
        right_position = node_positions[right]
        delta = right_position - left_position
        directions = observed_directions[(left, right)]
        edges.append(
            {
                "id": f"path_edge_{edge_index:03d}",
                "from": nodes[left]["id"],
                "to": nodes[right]["id"],
                "distance_m": round_float(vector_norm(delta)),
                "heading_vector": unit_vector(delta),
                "traversal_count": traversal_count,
                "bidirectional_observed": len(directions) > 1,
            }
        )

    path_network_length = sum(edge["distance_m"] for edge in edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "path_network_length_m": round_float(path_network_length),
        "recorded_trajectory_length_m": round_float(recorded_trajectory_length),
    }


def point_segment_distance(point, start, end):
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return vector_norm(point - start)
    t = float(np.dot(point - start, segment) / denom)
    t = min(max(t, 0.0), 1.0)
    projection = start + t * segment
    return vector_norm(point - projection)


def point_aabb_distance(point, bbox_min, bbox_max):
    delta = np.maximum(np.maximum(bbox_min - point, 0.0), point - bbox_max)
    return vector_norm(delta)


def aabb_segment_distance(bbox_min, bbox_max, start, end, samples=11):
    best = math.inf
    for index in range(samples):
        t = index / max(samples - 1, 1)
        point = start + t * (end - start)
        best = min(best, point_aabb_distance(point, bbox_min, bbox_max))
    return best


def nearest_path_distance(point, path_network):
    node_positions = {node["id"]: as_vec(node["position"]) for node in path_network["nodes"]}
    best_distance = math.inf
    best_node_id = None
    best_edge_id = None

    for edge in path_network["edges"]:
        start = node_positions[edge["from"]]
        end = node_positions[edge["to"]]
        distance = point_segment_distance(point, start, end)
        if distance < best_distance:
            best_distance = distance
            best_edge_id = edge["id"]
            best_node_id = edge["from"] if vector_norm(point - start) <= vector_norm(point - end) else edge["to"]

    if not path_network["edges"]:
        for node_id, position in node_positions.items():
            distance = vector_norm(point - position)
            if distance < best_distance:
                best_distance = distance
                best_node_id = node_id

    if best_distance == math.inf:
        return None, None, None
    return best_distance, best_node_id, best_edge_id


def nearest_path_clearance(bbox_min, bbox_max, path_network):
    node_positions = {node["id"]: as_vec(node["position"]) for node in path_network["nodes"]}
    best_distance = math.inf
    best_node_id = None
    best_edge_id = None

    for edge in path_network["edges"]:
        start = node_positions[edge["from"]]
        end = node_positions[edge["to"]]
        distance = aabb_segment_distance(bbox_min, bbox_max, start, end)
        if distance < best_distance:
            best_distance = distance
            best_edge_id = edge["id"]
            midpoint = (start + end) / 2.0
            best_node_id = edge["from"] if vector_norm(midpoint - start) <= vector_norm(midpoint - end) else edge["to"]

    if not path_network["edges"]:
        for node_id, position in node_positions.items():
            distance = point_aabb_distance(position, bbox_min, bbox_max)
            if distance < best_distance:
                best_distance = distance
                best_node_id = node_id

    if best_distance == math.inf:
        return None, None, None
    return best_distance, best_node_id, best_edge_id


def annotate_objects_with_path(objects, path_network):
    for obj in objects:
        center = as_vec(obj["center"])
        bbox_min = as_vec(obj["bbox_3d"]["min"])
        bbox_max = as_vec(obj["bbox_3d"]["max"])
        distance, node_id, edge_id = nearest_path_distance(center, path_network)
        clearance, clearance_node_id, clearance_edge_id = nearest_path_clearance(bbox_min, bbox_max, path_network)
        obj["distance_to_path_network_m"] = round_float(distance)
        obj["clearance_to_path_network_m"] = round_float(clearance)
        obj["nearest_path_node_id"] = clearance_node_id or node_id
        obj["nearest_path_edge_id"] = clearance_edge_id or edge_id


def relation_from_distance(distance):
    if distance <= 1.0:
        return "near"
    if distance <= 2.0:
        return "medium"
    return "far"


def risk_from_clearance(clearance):
    if clearance is None:
        return "unknown"
    if clearance <= 0.35:
        return "high"
    if clearance <= 0.75:
        return "medium"
    return "low"


def build_spatial_relations(objects, max_neighbors, max_distance):
    relations = {}
    centers = [(obj["id"], as_vec(obj["center"])) for obj in objects]
    for object_id, center in centers:
        distances = []
        for other_id, other_center in centers:
            if other_id == object_id:
                continue
            distance = vector_norm(other_center - center)
            if distance <= max_distance:
                distances.append((distance, other_id))
        distances.sort(key=lambda item: (item[0], item[1]))
        for distance, other_id in distances[:max_neighbors]:
            left, right = sorted((object_id, other_id))
            relations[(left, right)] = round_float(distance)

    return [
        {
            "from": left,
            "to": right,
            "distance_m": distance,
            "relation": relation_from_distance(distance),
        }
        for (left, right), distance in sorted(relations.items(), key=lambda item: (item[1], item[0]))
    ]


def build_path_nearby_objects(objects, path_nearby_radius):
    nearby = []
    for obj in objects:
        clearance = obj.get("clearance_to_path_network_m")
        if clearance is None or clearance > path_nearby_radius:
            continue
        nearby.append(
            {
                "object_id": obj["id"],
                "label": obj["label"],
                "nearest_path_node_id": obj.get("nearest_path_node_id"),
                "nearest_path_edge_id": obj.get("nearest_path_edge_id"),
                "distance_to_path_network_m": obj.get("distance_to_path_network_m"),
                "clearance_to_path_network_m": clearance,
                "relation_to_path": "near_path",
                "risk_level": risk_from_clearance(clearance),
            }
        )

    nearby.sort(key=lambda item: (item["clearance_to_path_network_m"], item["object_id"]))
    return nearby


def object_label_counts(objects):
    counts = defaultdict(int)
    for obj in objects:
        counts[obj["label"]] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_global_bounds(objects, path_network, max_object_path_distance):
    vectors = [as_vec(node["position"]) for node in path_network["nodes"]]
    for obj in objects:
        distance = obj.get("distance_to_path_network_m")
        if distance is not None and distance > max_object_path_distance:
            continue
        vectors.append(as_vec(obj["bbox_3d"]["min"]))
        vectors.append(as_vec(obj["bbox_3d"]["max"]))
    return map_bounds(vectors)


def build_scene(args):
    semantic_map = read_json(args.semantic_map)
    semantic_points = semantic_map.get("points", [])
    export_dir = Path(args.export_dir)
    keyframes = read_jsonl(export_dir / "keyframes.jsonl")

    semantic_objects, discarded_clusters = build_semantic_objects(
        semantic_points,
        args.cluster_radius,
        args.min_object_points,
        args.max_objects_per_label,
    )
    traversable_path_network = build_traversable_path_network(
        keyframes,
        args.path_node_radius,
        args.node_nearby_radius,
        semantic_objects,
    )
    annotate_objects_with_path(semantic_objects, traversable_path_network)
    semantic_objects.sort(
        key=lambda item: (
            item.get("distance_to_path_network_m") is None,
            item.get("distance_to_path_network_m", math.inf),
            item["label"],
            item["id"],
        )
    )

    spatial_relations = build_spatial_relations(
        semantic_objects,
        args.spatial_relation_neighbors,
        args.spatial_relation_radius,
    )
    path_nearby_objects = build_path_nearby_objects(semantic_objects, args.path_nearby_radius)
    global_bounds = build_global_bounds(semantic_objects, traversable_path_network, args.bounds_object_path_radius)

    has_metric_scale = args.scale_mode == "metric"
    scene = {
        "physical_scale_unit": "米" if has_metric_scale else "不确定",
        "has_metric_scale": has_metric_scale,
        "map_form": args.map_form,
        "dataset_name": args.dataset_name,
        "slam_mode": args.slam_mode,
        "coordinate_frame": {
            "unit": "meter" if has_metric_scale else "arbitrary_slam_unit",
            "origin": "ORB-SLAM3 world origin",
            "axes": "ORB-SLAM3 world frame",
        },
        "scene_summary": {
            "map_points_total": len(semantic_points),
            "semantic_objects_total": len(semantic_objects),
            "path_nodes_total": len(traversable_path_network["nodes"]),
            "path_edges_total": len(traversable_path_network["edges"]),
            "path_network_length_m": traversable_path_network["path_network_length_m"],
            "recorded_trajectory_length_m": traversable_path_network["recorded_trajectory_length_m"],
            "semantic_objects_by_label": object_label_counts(semantic_objects),
            "map_points_by_label": label_counts(semantic_points),
        },
        "global_bounds": global_bounds,
        "semantic_objects": semantic_objects,
        "spatial_relations": spatial_relations,
        "traversable_path_network": traversable_path_network,
        "path_nearby_objects": path_nearby_objects,
        "map_quality": {
            "map_type": "offline_sparse_semantic_map",
            "has_metric_scale": has_metric_scale,
            "semantic_source": "offline image segmentation projected to ORB-SLAM3 MapPoints",
            "path_network_source": "ORB-SLAM3 keyframe trajectory clustered into a traversable path network",
            "free_space_is_exhaustive": False,
            "requires_online_localization": True,
            "requires_realtime_obstacle_avoidance": True,
            "discarded_low_support_semantic_clusters": discarded_clusters,
        },
    }
    return scene


def parse_args():
    parser = argparse.ArgumentParser(description="Build a semantic navigation map from offline ORB-SLAM3 semantic outputs.")
    parser.add_argument("--semantic-map", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-radius", type=float, default=0.75)
    parser.add_argument("--min-object-points", type=int, default=5)
    parser.add_argument("--max-objects-per-label", type=int, default=40)
    parser.add_argument("--path-node-radius", type=float, default=0.5)
    parser.add_argument("--node-nearby-radius", type=float, default=2.0)
    parser.add_argument("--path-nearby-radius", type=float, default=1.25)
    parser.add_argument("--spatial-relation-neighbors", type=int, default=3)
    parser.add_argument("--spatial-relation-radius", type=float, default=3.0)
    parser.add_argument("--bounds-object-path-radius", type=float, default=8.0)
    parser.add_argument("--scale-mode", choices=("metric", "arbitrary"), default="metric")
    parser.add_argument("--map-form", choices=("full_map", "chunked_map"), default="full_map")
    parser.add_argument("--slam-mode", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--graph-stride", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    scene = build_scene(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[semantic_navigation_map] finished: "
        f"{scene['scene_summary']['semantic_objects_total']} objects, "
        f"{scene['scene_summary']['path_nodes_total']} path nodes, "
        f"{scene['scene_summary']['path_edges_total']} path edges, "
        f"{len(scene['path_nearby_objects'])} path-nearby objects"
    )


if __name__ == "__main__":
    main()
