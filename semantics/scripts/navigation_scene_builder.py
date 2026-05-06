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
    # 把 3D 空间划分成边长为 radius 的立方体网格，返回一个整数坐标表示 position 落在哪个网格里。
    return tuple(np.floor(position / radius).astype(np.int64).tolist())


def neighbor_cells(cell):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield cell[0] + dx, cell[1] + dy, cell[2] + dz


def connected_components(points, radius):
    # 没有输入点时没有任何连通簇。
    if not points:
        return []

    # 提前取出所有 3D 坐标；后续只在坐标数组上做距离计算。
    positions = [as_vec(point["position"]) for point in points]
    # 用半径大小划分 3D 网格，把点放进网格桶里，避免每个点都和全体点暴力比较。
    cells = defaultdict(list)
    for index, position in enumerate(positions):
        cells[grid_cell(position, radius)].append(index)

    # visited 标记点是否已经被归入某个簇；components 保存每个簇包含的点索引。
    visited = [False] * len(points)
    components = []
    for start in range(len(points)):
        if visited[start]:
            continue

        # 从一个未访问点开始做 BFS，所有在 radius 内连通的点都会被归入同一簇。
        visited[start] = True
        queue = deque([start])
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)
            current_position = positions[current]
            # 只检查当前点所在网格及周围 26 个相邻网格；半径内邻居不可能出现在更远网格。
            for cell in neighbor_cells(grid_cell(current_position, radius)):
                for candidate in cells.get(cell, []):
                    if visited[candidate]:
                        continue
                    # 两点距离小于等于 radius，就认为它们在空间上连通，并加入 BFS 队列继续扩张。
                    if vector_norm(current_position - positions[candidate]) <= radius:
                        visited[candidate] = True
                        queue.append(candidate)

        # component 存的是 points 的索引列表，上层会用这些索引取回真实 MapPoint。
        components.append(component)

    return components


def build_semantic_objects(semantic_points, cluster_radius, min_object_points, max_objects_per_label):
    # 先按语义类别分桶；unknown 点和缺少 3D 坐标的点无法形成可导航物体实例。
    points_by_label = defaultdict(list)
    for point in semantic_points:
        label = str(point.get("label", "unknown"))
        if label == "unknown":
            continue
        if not point.get("position") or len(point["position"]) != 3:
            continue
        points_by_label[label].append(point)

    # objects 是最终保留下来的语义物体实例，discarded_clusters 记录被过滤掉的低可信簇数量。
    objects = []
    discarded_clusters = 0
    per_label_index = defaultdict(int)

    # 每个类别单独聚类，避免把不同语义标签的点合并成同一个物体。
    for label in sorted(points_by_label):
        label_points = points_by_label[label]
        # 使用空间连通性把同类 MapPoint 聚成候选物体实例。
        components = connected_components(label_points, cluster_radius)
        # 大簇通常更稳定，优先保留；后续 max_objects_per_label 也会先留下支撑点更多的对象。
        components.sort(key=len, reverse=True)

        kept_for_label = 0
        for component in components:
            # 点数太少的簇容易是误检或噪声，不进入最终导航对象。
            if len(component) < min_object_points:
                discarded_clusters += 1
                continue
            # 限制每个类别输出对象数量，防止某一类产生过多小对象让 scene.json 失控。
            if kept_for_label >= max_objects_per_label:
                discarded_clusters += 1
                continue

            # 取出该聚类里的所有 MapPoint，并基于它们的 3D 坐标计算对象几何属性。
            member_points = [label_points[index] for index in component]
            coords = np.vstack([as_vec(point["position"]) for point in member_points])
            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)
            # 聚合语义证据：命中次数表示这个对象被多少次 2D mask 观测支持。
            semantic_hits = sum(int(point.get("semantic_observation_hits", 0)) for point in member_points)
            point_confidences = []
            for point in member_points:
                # 单点置信度用累计语义得分除以命中次数，避免多次观测的点天然分数更大。
                hits = max(int(point.get("semantic_observation_hits", 0)), 1)
                point_confidences.append(min(1.0, float(point.get("score", 0.0)) / hits))

            # 给同一类别下的对象稳定编号，并写入中心、包围盒、尺寸、置信度和支撑点数量。
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

    # 输出前按 label/id 排序，保证同一输入下 scene.json 的对象顺序稳定。
    objects.sort(key=lambda item: (item["label"], item["id"]))
    return objects, discarded_clusters


def build_traversable_path_network(keyframes, node_radius, nearby_radius, objects):
    # 从关键帧中提取相机中心；这些中心表示 SLAM 实际走过/拍摄过的轨迹点。
    valid = []
    for keyframe in sorted(keyframes, key=lambda kf: (float(kf.get("timestamp", 0.0)), int(kf["keyframe_id"]))):
        center = camera_center_from_keyframe(keyframe)
        if center is not None:
            valid.append(center)

    # 如果没有有效相机位姿，就无法构造路径网络，返回空图结构。
    if not valid:
        return {
            "nodes": [],
            "edges": [],
            "path_network_length_m": 0.0,
            "recorded_trajectory_length_m": 0.0,
        }

    # 原始轨迹长度按相邻关键帧相机中心的折线距离累加，保留 SLAM 记录的实际运动长度。
    recorded_trajectory_length = sum(vector_norm(curr - prev) for prev, curr in zip(valid, valid[1:]))
    # node_accumulators 用来把相近关键帧合并成一个路径节点，assignments 记录每个关键帧归属哪个节点。
    node_accumulators = []
    assignments = []

    # 顺着时间顺序遍历相机轨迹，把距离已有节点中心不超过 node_radius 的关键帧合并到该节点。
    for position in valid:
        best_index = None
        best_distance = math.inf
        for index, node in enumerate(node_accumulators):
            center = node["sum"] / node["count"]
            distance = vector_norm(position - center)
            if distance <= node_radius and distance < best_distance:
                best_index = index
                best_distance = distance

        # 没有足够近的已有节点时创建新节点；否则更新已有节点的坐标均值累加器。
        if best_index is None:
            best_index = len(node_accumulators)
            node_accumulators.append({"sum": position.copy(), "count": 1})
        else:
            node_accumulators[best_index]["sum"] += position
            node_accumulators[best_index]["count"] += 1
        assignments.append(best_index)

    # 每个路径节点的位置取其包含关键帧相机中心的平均值。
    node_positions = [node["sum"] / node["count"] for node in node_accumulators]
    # 根据关键帧归属序列统计节点之间的实际转移，形成路径边和被观测到的通行方向。
    edge_stats = {}
    observed_directions = defaultdict(set)
    for previous, current in zip(assignments, assignments[1:]):
        if previous == current:
            continue
        pair = tuple(sorted((previous, current)))
        edge_stats[pair] = edge_stats.get(pair, 0) + 1
        observed_directions[pair].add((previous, current))

    # 预取语义对象中心，用于给每个路径节点标注附近可作为路标的对象。
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
        # 每个节点只保留最近的一批对象，避免 scene.json 中单个路径节点过于臃肿。
        nearby_objects.sort(key=lambda item: (item["distance_m"], item["object_id"]))
        nodes.append(
            {
                "id": f"path_node_{index + 1:03d}",
                "position": round_vec(position),
                "observed_keyframes_count": int(node_accumulators[index]["count"]),
                "nearby_objects": nearby_objects[:12],
            }
        )

    # 把节点转移统计写成边；边表示相机轨迹中确实从一个路径节点移动到过另一个路径节点。
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

    # 路径网络长度是压缩后的节点边长度总和，通常小于或等于原始关键帧折线轨迹长度。
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
    # 给每个语义对象补充它与路径网络的关系，方便后续判断哪些对象靠近可通行轨迹。
    for obj in objects:
        # center 用于估计对象中心到路径的距离，bbox 用于估计对象实体边界到路径的 clearance。
        center = as_vec(obj["center"])
        bbox_min = as_vec(obj["bbox_3d"]["min"])
        bbox_max = as_vec(obj["bbox_3d"]["max"])
        # center 距离描述对象中心离路径多远，适合做排序和“靠近路径”判断。
        distance, node_id, edge_id = nearest_path_distance(center, path_network)
        # clearance 距离描述对象包围盒离路径多近，更接近避障意义上的最小间隙。
        clearance, clearance_node_id, clearance_edge_id = nearest_path_clearance(bbox_min, bbox_max, path_network)
        # 将路径相关属性直接写回对象，后续 scene、LLM view 和草图都会复用这些字段。
        obj["distance_to_path_network_m"] = round_float(distance)
        obj["clearance_to_path_network_m"] = round_float(clearance)
        # 最近节点/边优先使用 clearance 的结果，因为它考虑了对象体积；没有时退回中心距离结果。
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
    # 构建语义对象之间的近邻关系；结果用于描述场景中物体彼此的空间邻接。
    relations = {}
    # 只用对象中心点估计对象间距离，避免把完整 bbox 几何引入关系计算。
    centers = [(obj["id"], as_vec(obj["center"])) for obj in objects]
    # 对每个对象单独寻找 max_distance 范围内最近的若干邻居。
    for object_id, center in centers:
        distances = []
        for other_id, other_center in centers:
            if other_id == object_id:
                continue
            distance = vector_norm(other_center - center)
            if distance <= max_distance:
                distances.append((distance, other_id))
        # 先按距离、再按对象 id 排序，保证近邻选择稳定可复现。
        distances.sort(key=lambda item: (item[0], item[1]))
        for distance, other_id in distances[:max_neighbors]:
            # 空间关系是无向的，A-B 和 B-A 只保留一条记录。
            left, right = sorted((object_id, other_id))
            relations[(left, right)] = round_float(distance)

    # 将去重后的关系字典转换成 JSON 友好的列表，并附上粗粒度 near/medium/far 关系标签。
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
    # 抽取离路径网络足够近的对象；这些对象更可能作为导航路标或潜在障碍物。
    nearby = []
    for obj in objects:
        # clearance 使用对象包围盒到路径的最小距离，比中心点距离更适合判断是否贴近路径。
        clearance = obj.get("clearance_to_path_network_m")
        if clearance is None or clearance > path_nearby_radius:
            continue
        # 只保留导航需要的精简字段，避免重复塞入完整 semantic_objects。
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

    # 越贴近路径的对象越靠前；同距离时按 id 排序保证输出稳定。
    nearby.sort(key=lambda item: (item["clearance_to_path_network_m"], item["object_id"]))
    return nearby


def object_label_counts(objects):
    counts = defaultdict(int)
    for obj in objects:
        counts[obj["label"]] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_global_bounds(objects, path_network, max_object_path_distance):
    # 全局边界先包含路径节点，保证场景范围至少覆盖相机实际走过的轨迹。
    vectors = [as_vec(node["position"]) for node in path_network["nodes"]]
    # 再加入路径附近对象的 3D 包围盒，避免离导航路径很远的误检对象把整体边界拉得过大。
    for obj in objects:
        distance = obj.get("distance_to_path_network_m")
        if distance is not None and distance > max_object_path_distance:
            continue
        # 用 bbox 的 min/max 两个角点扩展边界，就能覆盖该对象的完整空间范围。
        vectors.append(as_vec(obj["bbox_3d"]["min"]))
        vectors.append(as_vec(obj["bbox_3d"]["max"]))
    # 返回 min/max/extent，供草图投影、LLM 视图和 scene 元信息复用。
    return map_bounds(vectors)


def projection_axes_from_bounds(bounds):
    extent = bounds.get("extent") if bounds else None
    if not extent or len(extent) != 3:
        return 0, 1
    ranked = sorted(range(3), key=lambda index: float(extent[index]), reverse=True)
    return ranked[0], ranked[1]


def build_object_footprint(obj, axes):
    x_axis, y_axis = axes
    height_axis = next(axis for axis in range(3) if axis not in axes)
    bbox_min = as_vec(obj["bbox_3d"]["min"])
    bbox_max = as_vec(obj["bbox_3d"]["max"])
    center = as_vec(obj["center"])
    footprint_size = [float(bbox_max[x_axis] - bbox_min[x_axis]), float(bbox_max[y_axis] - bbox_min[y_axis])]
    clearance = obj.get("clearance_to_path_network_m")
    return {
        "object_id": obj["id"],
        "label": obj["label"],
        "center_3d": round_vec(center),
        "projection_axes": {
            "horizontal": axis_name(x_axis),
            "vertical": axis_name(y_axis),
            "height": axis_name(height_axis),
        },
        "footprint_2d": {
            "min": round_vec([bbox_min[x_axis], bbox_min[y_axis]]),
            "max": round_vec([bbox_max[x_axis], bbox_max[y_axis]]),
            "size": round_vec(footprint_size),
            "area": round_float(max(footprint_size[0], 0.0) * max(footprint_size[1], 0.0)),
        },
        "height": round_float(float(bbox_max[height_axis] - bbox_min[height_axis])),
        "bbox_3d": obj["bbox_3d"],
        "extent_3d": obj["extent"],
        "distance_to_path_network_m": obj.get("distance_to_path_network_m"),
        "clearance_to_path_network_m": clearance,
        "risk_level": risk_from_clearance(clearance),
        "support_points": obj.get("support_points", 0),
    }


def build_navigation_llm_view(scene):
    # 根据全局边界选择跨度最大的两个轴做 2D 投影，便于把 3D 对象压缩成平面 footprint。
    axes = projection_axes_from_bounds(scene.get("global_bounds") or {})
    # 从完整 scene 中取出路径网络和语义对象；这里不重新构图，只生成精简视图。
    path = scene.get("traversable_path_network", {})
    objects = scene.get("semantic_objects", [])
    # 将每个 3D 语义对象转换成包含 2D footprint、路径 clearance 和风险等级的紧凑表示。
    footprints = [build_object_footprint(obj, axes) for obj in objects]
    # 优先排列最贴近路径的对象，因为它们对导航提示和障碍风险最重要。
    footprints.sort(
        key=lambda item: (
            item.get("clearance_to_path_network_m") is None,
            item.get("clearance_to_path_network_m", math.inf),
            item["label"],
            item["object_id"],
        )
    )
    # 返回面向 LLM/下游应用的轻量地图事实：投影轴、解释说明、路径摘要和对象 footprint。
    return {
        "purpose": "compact map facts for indoor blind-navigation LLM reasoning",
        "scale_unit": "meter" if scene.get("has_metric_scale") else "uncertain",
        "projection_axes": {
            "horizontal": axis_name(axes[0]),
            "vertical": axis_name(axes[1]),
            "height": axis_name(next(axis for axis in range(3) if axis not in axes)),
        },
        "interpretation_notes": [
            "Object sizes come from semantic ORB-SLAM3 sparse MapPoint bounding boxes, not dense CAD geometry.",
            "Path network is the observed traversable camera trajectory network; it is useful as a conservative preferred route.",
            "clearance_to_path_network_m is more useful than center distance for obstacle warnings.",
        ],
        "path_network_compact": {
            "nodes_total": len(path.get("nodes", [])),
            "edges_total": len(path.get("edges", [])),
            "path_network_length_m": path.get("path_network_length_m"),
            "recorded_trajectory_length_m": path.get("recorded_trajectory_length_m"),
        },
        "object_footprints": footprints,
    }


def build_scene(args):
    # 读取语义融合阶段的主产物：points 是带语义标签的 3D MapPoint，summary 是上游处理统计。
    semantic_map = read_json(args.semantic_map)
    semantic_points = semantic_map.get("points", [])
    semantic_summary = semantic_map.get("scene_summary", {})
    # 读取 SLAM 导出的关键帧轨迹，后续会把相机运动轨迹压缩成导航路径网络。
    export_dir = Path(args.export_dir)
    keyframes = read_jsonl(export_dir / "keyframes.jsonl")

    # 将离散的语义 MapPoint 按 label 和空间距离聚类成“物体实例”，并丢弃点数不足的低可信簇。
    semantic_objects, discarded_clusters = build_semantic_objects(
        semantic_points,
        args.cluster_radius,
        args.min_object_points,
        args.max_objects_per_label,
    )
    # 将关键帧相机中心聚合成路径节点/边，形成一个近似的可通行轨迹网络。
    traversable_path_network = build_traversable_path_network(
        keyframes,
        args.path_node_radius,
        args.node_nearby_radius,
        semantic_objects,
    )
    # 给每个语义对象补充它到路径网络的距离、最近路径节点和通行风险等导航相关属性。
    annotate_objects_with_path(semantic_objects, traversable_path_network)
    # 优先展示靠近路径的对象；这些对象通常更适合做导航路标或避障参考。
    semantic_objects.sort(
        key=lambda item: (
            item.get("distance_to_path_network_m") is None,
            item.get("distance_to_path_network_m", math.inf),
            item["label"],
            item["id"],
        )
    )

    # 构建对象之间的近邻空间关系，例如某个对象附近有哪些其他对象。
    spatial_relations = build_spatial_relations(
        semantic_objects,
        args.spatial_relation_neighbors,
        args.spatial_relation_radius,
    )
    # 抽取路径附近的对象列表，方便下游快速找到可作为导航参照物的语义对象。
    path_nearby_objects = build_path_nearby_objects(semantic_objects, args.path_nearby_radius)
    # 估计场景全局边界，用于投影草图、空间归一化和导航结果的整体范围描述。
    global_bounds = build_global_bounds(semantic_objects, traversable_path_network, args.bounds_object_path_radius)

    # scale_mode 决定输出距离能否被解释为米；单目 arbitrary scale 下只能表示相对尺度。
    has_metric_scale = args.scale_mode == "metric"
    # 组装最终导航场景：同时保留元信息、统计摘要、语义对象、空间关系和路径网络。
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
            "keyframes_processed": int(semantic_summary.get("keyframes_processed", len(keyframes))),
            "map_points_total": len(semantic_points),
            "map_points_labeled": int(
                semantic_summary.get(
                    "map_points_labeled",
                    sum(1 for point in semantic_points if str(point.get("label", "unknown")) != "unknown"),
                )
            ),
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
    # 在完整 scene 基础上派生一份更轻量的 LLM 导航视图，供 main() 选择内嵌或单独写出。
    scene["navigation_llm_view"] = build_navigation_llm_view(scene)
    return scene


def axis_name(axis):
    return ("x", "y", "z")[axis]


def choose_sketch_axes(scene):
    bounds = scene.get("global_bounds") or {}
    return projection_axes_from_bounds(bounds)


def sketch_positions(scene):
    positions = []
    for node in scene.get("traversable_path_network", {}).get("nodes", []):
        position = node.get("position")
        if position and len(position) == 3:
            positions.append(as_vec(position))
    for obj in scene.get("semantic_objects", []):
        center = obj.get("center")
        if center and len(center) == 3:
            positions.append(as_vec(center))
    return positions


def sketch_bounds(scene, axes, padding_ratio=0.08):
    bounds = scene.get("global_bounds") or {}
    bounds_min = bounds.get("min")
    bounds_max = bounds.get("max")
    if bounds_min and bounds_max and len(bounds_min) == 3 and len(bounds_max) == 3:
        low = as_vec(bounds_min)
        high = as_vec(bounds_max)
    else:
        positions = sketch_positions(scene)
        if not positions:
            low = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
            high = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        else:
            coords = np.vstack(positions)
            low = coords.min(axis=0)
            high = coords.max(axis=0)

    for axis in axes:
        span = float(high[axis] - low[axis])
        pad = max(span * padding_ratio, 0.25)
        low[axis] -= pad
        high[axis] += pad
        if abs(high[axis] - low[axis]) <= 1e-9:
            low[axis] -= 0.5
            high[axis] += 0.5
    return low, high


def make_projector(low, high, axes, width, height):
    x_axis, y_axis = axes
    x_span = max(float(high[x_axis] - low[x_axis]), 1e-9)
    y_span = max(float(high[y_axis] - low[y_axis]), 1e-9)

    def project(position):
        vector = as_vec(position)
        col = int(round((float(vector[x_axis] - low[x_axis]) / x_span) * (width - 1)))
        row = int(round((float(high[y_axis] - vector[y_axis]) / y_span) * (height - 1)))
        col = min(max(col, 0), width - 1)
        row = min(max(row, 0), height - 1)
        return row, col

    return project


def draw_cell(grid, row, col, char, overwrite=False):
    height = len(grid)
    width = len(grid[0]) if height else 0
    if row < 0 or row >= height or col < 0 or col >= width:
        return
    if overwrite:
        grid[row][col] = char
        return
    current = grid[row][col]
    if current == " " or current == char:
        grid[row][col] = char
        return

    priority = {" ": 0, ".": 1, "+": 2, "X": 3}
    if priority.get(char, 0) > priority.get(current, 0):
        grid[row][col] = char


def draw_line(grid, start, end, char, overwrite=False):
    row0, col0 = start
    row1, col1 = end
    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1
    error = d_col - d_row

    row, col = row0, col0
    while True:
        draw_cell(grid, row, col, char, overwrite=overwrite)
        if row == row1 and col == col1:
            break
        twice_error = 2 * error
        if twice_error > -d_row:
            error -= d_row
            col += step_col
        if twice_error < d_col:
            error += d_col
            row += step_row


def draw_rect(grid, top, left, bottom, right, char, fill=False):
    if top > bottom:
        top, bottom = bottom, top
    if left > right:
        left, right = right, left
    if bottom - top <= 1 and right - left <= 1:
        draw_cell(grid, top, left, char)
        return
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            is_border = row in (top, bottom) or col in (left, right)
            if fill or is_border:
                draw_cell(grid, row, col, char)


def projected_bbox(project, obj):
    bbox = obj.get("bbox_3d") or {}
    bbox_min = bbox.get("min")
    bbox_max = bbox.get("max")
    if not bbox_min or not bbox_max:
        return None
    corners = [
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ]
    cells = [project(corner) for corner in corners]
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), min(cols), max(rows), max(cols)


def object_code(index):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if index < len(alphabet):
        return alphabet[index]
    return "?"


def risk_symbol(clearance):
    risk = risk_from_clearance(clearance)
    if risk == "high":
        return "X"
    if risk == "medium":
        return "+"
    return "."


def build_ascii_grid(scene, width=80, height=36, max_objects=48):
    width = max(24, int(width))
    height = max(12, int(height))
    axes = choose_sketch_axes(scene)
    x_axis, y_axis = axes
    low, high = sketch_bounds(scene, axes)
    project = make_projector(low, high, axes, width, height)
    grid = [[" " for _ in range(width)] for _ in range(height)]

    path = scene.get("traversable_path_network", {})
    nodes = path.get("nodes", [])
    node_positions = {node["id"]: node.get("position") for node in nodes if node.get("position")}
    projected_nodes = {node_id: project(position) for node_id, position in node_positions.items()}

    objects = scene.get("semantic_objects", [])
    objects = sorted(
        objects,
        key=lambda item: (
            item.get("clearance_to_path_network_m") is None,
            item.get("clearance_to_path_network_m", math.inf),
            item.get("label", ""),
            item.get("id", ""),
        ),
    )[:max_objects]
    high_count = 0
    medium_count = 0
    low_count = 0
    for obj in objects:
        risk = risk_from_clearance(obj.get("clearance_to_path_network_m"))
        if risk == "low":
            low_count += 1
            continue
        bbox = projected_bbox(project, obj)
        if bbox:
            top, left, bottom, right = bbox
            symbol = risk_symbol(obj.get("clearance_to_path_network_m"))
            draw_rect(grid, top, left, bottom, right, symbol, fill=False)
            if risk == "high":
                high_count += 1
            elif risk == "medium":
                medium_count += 1

    # Draw the path last so the traversable route stays visually continuous.
    for edge in path.get("edges", []):
        start = projected_nodes.get(edge.get("from"))
        end = projected_nodes.get(edge.get("to"))
        if start and end:
            draw_line(grid, start, end, "#", overwrite=True)

    if nodes:
        start = projected_nodes.get(nodes[0].get("id"))
        end = projected_nodes.get(nodes[-1].get("id"))
        if start:
            draw_cell(grid, start[0], start[1], "S", overwrite=True)
        if end:
            draw_cell(grid, end[0], end[1], "E", overwrite=True)

    legend = []
    for obj in objects:
        risk = risk_from_clearance(obj.get("clearance_to_path_network_m"))
        distance = obj.get("distance_to_path_network_m")
        clearance = obj.get("clearance_to_path_network_m")
        distance_text = "?" if distance is None else f"{distance:.2f}"
        clearance_text = "?" if clearance is None else f"{clearance:.2f}"
        footprint = build_object_footprint(obj, axes)
        footprint_size = footprint["footprint_2d"]["size"]
        height_text = "?" if footprint["height"] is None else f"{footprint['height']:.2f}"
        symbol = risk_symbol(clearance)
        legend.append(
            f"{symbol} {risk.upper()} {obj.get('id')} label={obj.get('label')} "
            f"footprint_{axis_name(x_axis)}_by_{axis_name(y_axis)}={footprint_size[0]:.2f}x{footprint_size[1]:.2f}m "
            f"height_{footprint['projection_axes']['height']}={height_text}m "
            f"dist_path={distance_text}m clearance={clearance_text}m"
        )

    summary = scene.get("scene_summary", {})
    unit = "meter" if scene.get("has_metric_scale") else "uncertain"
    x_span = max(float(high[x_axis] - low[x_axis]), 1e-9)
    y_span = max(float(high[y_axis] - low[y_axis]), 1e-9)
    lines = [
        f"scene_sketch: {scene.get('dataset_name', '')} / {scene.get('slam_mode', '')}",
        f"map_form: {scene.get('map_form', '')}",
        f"scale_unit: {unit}",
        f"projection: horizontal={axis_name(x_axis)} vertical={axis_name(y_axis)}",
        f"grid_size: {width}x{height}",
        f"cell_size: {x_span / max(width - 1, 1):.3f} x {y_span / max(height - 1, 1):.3f}",
        "legend: #=traversable path network, S/E=start/end-ish, X=high-risk obstacle bbox border, +=medium-risk obstacle bbox border",
        "note: path is drawn last and kept continuous; low-risk objects are omitted from the grid and listed below",
        "note: obstacle boxes are sparse-SLAM semantic bounding boxes; use as rough navigation context, not exact collision geometry",
        (
            "summary: "
            f"objects={summary.get('semantic_objects_total', 0)}, "
            f"path_nodes={summary.get('path_nodes_total', 0)}, "
            f"path_edges={summary.get('path_edges_total', 0)}, "
            f"points={summary.get('map_points_total', 0)}, "
            f"grid_high={high_count}, grid_medium={medium_count}, grid_low_omitted={low_count}"
        ),
        "+" + "-" * width + "+",
    ]
    lines.extend("|" + "".join(row) + "|" for row in grid)
    lines.append("+" + "-" * width + "+")
    if legend:
        lines.append("objects:")
        lines.extend(legend)
    if len(scene.get("semantic_objects", [])) > len(objects):
        lines.append(f"objects_truncated: {len(scene.get('semantic_objects', [])) - len(objects)}")
    return "\n".join(lines) + "\n"


def parse_args():
    # 定义导航场景生成脚本的命令行参数入口。
    parser = argparse.ArgumentParser(description="Build a semantic navigation map from offline ORB-SLAM3 semantic outputs.")
    # 输入的语义地图 JSON，由 offline_semantic_mapper.py 生成。
    parser.add_argument("--semantic-map", required=True)
    # SLAM 导出目录，用于读取关键帧、地图点和轨迹等几何信息。
    parser.add_argument("--export-dir", required=True)
    # 完整导航场景 JSON 的输出路径。
    parser.add_argument("--output", required=True)
    # ASCII 场景草图输出路径；为空时不生成草图。
    parser.add_argument("--sketch-output", default="")
    # 面向 LLM 或下游应用的精简导航视图 JSON 输出路径；为空时嵌入主 scene。
    parser.add_argument("--llm-view-output", default="")
    # ASCII 草图宽度，单位是字符列数。
    parser.add_argument("--sketch-width", type=int, default=80)
    # ASCII 草图高度，单位是字符行数。
    parser.add_argument("--sketch-height", type=int, default=36)
    # ASCII 草图中最多绘制的语义对象数量，避免草图过密。
    parser.add_argument("--sketch-max-objects", type=int, default=48)
    # 同类语义点聚成一个物体实例时使用的空间半径。
    parser.add_argument("--cluster-radius", type=float, default=0.75)
    # 一个语义物体实例至少需要包含的 MapPoint 数量。
    parser.add_argument("--min-object-points", type=int, default=5)
    # 每个语义类别最多保留的物体实例数量，防止单类对象过多压垮导航 JSON。
    parser.add_argument("--max-objects-per-label", type=int, default=40)
    # 将相机关键帧轨迹压缩成导航路径节点时使用的合并半径。
    parser.add_argument("--path-node-radius", type=float, default=0.5)
    # 判断语义对象是否靠近某个路径节点时使用的半径。
    parser.add_argument("--node-nearby-radius", type=float, default=2.0)
    # 判断语义对象是否靠近整条导航路径时使用的半径。
    parser.add_argument("--path-nearby-radius", type=float, default=1.25)
    # 每个语义对象最多记录多少个空间邻居关系。
    parser.add_argument("--spatial-relation-neighbors", type=int, default=3)
    # 计算物体之间空间关系时的最大邻居搜索距离。
    parser.add_argument("--spatial-relation-radius", type=float, default=3.0)
    # 构建全局边界时，路径周围多远以内的对象会被纳入边界估计。
    parser.add_argument("--bounds-object-path-radius", type=float, default=8.0)
    # SLAM 地图尺度类型：metric 表示米制尺度，arbitrary 表示单目等不确定尺度。
    parser.add_argument("--scale-mode", choices=("metric", "arbitrary"), default="metric")
    # 地图形态标记，用于区分全量地图和分块地图生成的导航结果。
    parser.add_argument("--map-form", choices=("full_map", "chunked_map"), default="full_map")
    # 本次运行的 SLAM 模式，写入输出 JSON 作为元信息。
    parser.add_argument("--slam-mode", default="")
    # 数据集名称，写入输出 JSON 作为元信息。
    parser.add_argument("--dataset-name", default="")
    # 兼容旧参数的隐藏选项；当前构图逻辑不对用户展示。
    parser.add_argument("--graph-stride", type=int, default=None, help=argparse.SUPPRESS)
    # 返回解析后的命令行参数对象。
    return parser.parse_args()


def main():
    # 导航场景生成入口：读取语义地图和 SLAM 导出，聚类语义对象并构建路径/空间关系。
    args = parse_args()
    # build_scene 是核心构图函数，返回完整导航场景，包括 objects、path network、relations 和 LLM view。
    scene = build_scene(args)
    # LLM 视图是从完整 scene 中抽出的轻量结构，便于下游语言模型或导航应用直接消费。
    llm_view = scene.get("navigation_llm_view", {})
    scene_output = dict(scene)
    # 如果单独指定 llm_view_output，则主 scene.json 不再内嵌 navigation_llm_view，避免重复存储。
    if args.llm_view_output:
        scene_output.pop("navigation_llm_view", None)

    # 写出完整导航场景 JSON：这是后续程序读取的主产物。
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scene_output, ensure_ascii=False, indent=2), encoding="utf-8")
    # 如指定 LLM 视图路径，则额外写出一个面向语言模型的精简导航 JSON。
    if args.llm_view_output:
        llm_view_path = Path(args.llm_view_output)
        llm_view_path.parent.mkdir(parents=True, exist_ok=True)
        llm_view_path.write_text(json.dumps(llm_view, ensure_ascii=False, indent=2), encoding="utf-8")
    # 如指定草图路径，则把 3D 场景投影成 ASCII 网格，方便快速人工检查物体和路径布局。
    if args.sketch_output:
        sketch_path = Path(args.sketch_output)
        sketch_path.parent.mkdir(parents=True, exist_ok=True)
        sketch_path.write_text(
            build_ascii_grid(scene, args.sketch_width, args.sketch_height, args.sketch_max_objects),
            encoding="utf-8",
        )
    # 打印导航构图摘要，快速确认对象数量、路径节点/边数量和路径附近对象数量。
    print(
        "[scene_builder] finished: "
        f"{scene['scene_summary']['semantic_objects_total']} objects, "
        f"{scene['scene_summary']['path_nodes_total']} path nodes, "
        f"{scene['scene_summary']['path_edges_total']} path edges, "
        f"{len(scene['path_nearby_objects'])} path-nearby objects"
    )


if __name__ == "__main__":
    main()
