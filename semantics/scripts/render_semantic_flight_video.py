#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


def color_for_label(label):
    palette = {
        "unknown": (0.45, 0.47, 0.50),
        "chair": (0.16, 0.64, 0.95),
        "dining table": (0.95, 0.58, 0.18),
        "couch": (0.73, 0.42, 0.95),
        "potted plant": (0.18, 0.78, 0.36),
        "bed": (0.95, 0.25, 0.29),
        "book": (0.97, 0.80, 0.22),
        "tv": (0.30, 0.92, 0.85),
        "car": (0.96, 0.46, 0.16),
    }
    if label in palette:
        return palette[label]
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return tuple((80 + digest[i] % 176) / 255.0 for i in range(3))


def camera_center_from_keyframe(keyframe):
    twc = keyframe.get("Twc")
    if not twc or len(twc) != 16:
        return None
    return np.array([float(twc[3]), float(twc[7]), float(twc[11])], dtype=np.float64)


def padded_limits(points, padding=0.15):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pads = np.maximum(spans * padding, 1e-3)
    return [(mins[i] - pads[i], maxs[i] + pads[i]) for i in range(3)]


def scene_bounds_points(scene):
    bounds = scene.get("global_bounds") or {}
    mn = bounds.get("min")
    mx = bounds.get("max")
    if not mn or not mx or len(mn) != 3 or len(mx) != 3:
        return None
    return np.array([mn, mx], dtype=np.float64)


def robust_bounds_points(points, percentile):
    lower = np.percentile(points, percentile, axis=0)
    upper = np.percentile(points, 100.0 - percentile, axis=0)
    return np.array([lower, upper], dtype=np.float64)


def compute_focus_limits(scene, positions, camera_centers, args):
    focus_points = []
    mode = args.focus_bounds
    if mode == "scene":
        bounds = scene_bounds_points(scene)
        if bounds is not None:
            focus_points.append(bounds)
        else:
            mode = "robust"

    if mode == "robust":
        focus_points.append(robust_bounds_points(positions, args.robust_percentile))
    elif mode == "all":
        focus_points.append(positions)

    if len(camera_centers) > 0:
        focus_points.append(camera_centers)
    for obj in scene.get("semantic_objects", []):
        bbox = obj.get("bbox_3d")
        if bbox:
            focus_points.append(np.array([bbox["min"], bbox["max"]], dtype=np.float64))
    return padded_limits(np.vstack(focus_points), padding=args.padding)


def points_inside_limits(points, limits):
    mask = np.ones(len(points), dtype=bool)
    for axis, (lo, hi) in enumerate(limits):
        mask &= (points[:, axis] >= lo) & (points[:, axis] <= hi)
    return mask


def draw_bbox(ax, bbox, color, linewidth=1.2, alpha=0.75):
    mn = np.array(bbox["min"], dtype=np.float64)
    mx = np.array(bbox["max"], dtype=np.float64)
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for left, right in edges:
        xs, ys, zs = zip(corners[left], corners[right])
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=alpha)


def build_display_bboxes(scene, positions, labels, percentile, min_points):
    display_bboxes = {}
    label_array = np.array(labels)
    for obj in scene.get("semantic_objects", []):
        bbox = obj.get("bbox_3d")
        label = str(obj.get("label", "unknown"))
        if not bbox or label == "unknown":
            continue
        mn = np.array(bbox["min"], dtype=np.float64)
        mx = np.array(bbox["max"], dtype=np.float64)
        mask = label_array == label
        for axis in range(3):
            mask &= (positions[:, axis] >= mn[axis] - 1e-6) & (positions[:, axis] <= mx[axis] + 1e-6)
        member_positions = positions[mask]
        if len(member_positions) < min_points:
            display_bboxes[obj["id"]] = bbox
            continue
        lower = np.percentile(member_positions, percentile, axis=0)
        upper = np.percentile(member_positions, 100.0 - percentile, axis=0)
        # Percentile boxes are for visualization only; scene.json keeps the full sparse evidence box.
        display_bboxes[obj["id"]] = {
            "min": [float(value) for value in lower],
            "max": [float(value) for value in upper],
        }
    return display_bboxes


def object_risk(obj):
    risk = obj.get("risk_level")
    if risk:
        return risk
    clearance = obj.get("clearance_to_path_network_m")
    if clearance is None:
        return "unknown"
    if clearance <= 0.35:
        return "high"
    if clearance <= 0.75:
        return "medium"
    return "low"


def risk_color(risk):
    if risk == "high":
        return "#ff4d5e"
    if risk == "medium":
        return "#ffc857"
    if risk == "low":
        return "#6ee7b7"
    return "#cbd5e1"


def draw_video_observations(image, observations, point_by_id, label_color_bgr, semantic_enabled, max_points=1200):
    if image is None:
        return None
    canvas = image.copy()
    if not observations:
        return canvas
    stride = max(1, len(observations) // max_points)
    label_counts = Counter()
    for obs in observations[::stride]:
        point = point_by_id.get(int(obs["map_point_id"]))
        if point is None:
            continue
        label = point["label"] if semantic_enabled else "unknown"
        label_counts[label] += 1
        color = label_color_bgr.get(label, (170, 170, 170))
        u = int(round(float(obs["u"])))
        v = int(round(float(obs["v"])))
        if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
            cv2.circle(canvas, (u, v), 4, color, -1, lineType=cv2.LINE_AA)

    panel_lines = ["2D observations attached to SLAM points"]
    if semantic_enabled:
        panel_lines.extend(f"{label}: {count}" for label, count in label_counts.most_common(6) if label != "unknown")
    else:
        panel_lines.append("gray = ORB-SLAM3 observations before semantic voting")
    x0, y0 = 18, 34
    line_height = 28
    panel_width = min(canvas.shape[1] - 36, 560)
    panel_height = line_height * len(panel_lines) + 20
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 26), (x0 - 8 + panel_width, y0 - 26 + panel_height), (5, 7, 13), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    for index, line in enumerate(panel_lines):
        y = y0 + index * line_height
        if index > 0 and semantic_enabled:
            label = line.split(":", 1)[0]
            color = label_color_bgr.get(label, (230, 230, 230))
            cv2.circle(canvas, (x0 + 8, y - 5), 7, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, line, (x0 + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 248, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (245, 248, 255), 2, cv2.LINE_AA)
    return canvas


def image_to_rgb(path, observations, point_by_id, label_color_bgr, semantic_enabled):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((720, 480, 3), dtype=np.uint8)
    image = draw_video_observations(image, observations, point_by_id, label_color_bgr, semantic_enabled)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def setup_3d_axis(ax, limits, elev, azim):
    ax.set_facecolor("#05070d")
    ax.grid(True, alpha=0.12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.02, 0.03, 0.06, 1.0))
        axis._axinfo["grid"]["color"] = (0.35, 0.38, 0.45, 0.18)
        axis._axinfo["tick"]["color"] = (0.78, 0.80, 0.84, 1.0)
    ax.tick_params(colors="#cfd4df", labelsize=8)
    ax.set_xlabel("x", color="#d8dde8")
    ax.set_ylabel("y", color="#d8dde8")
    ax.set_zlabel("z", color="#d8dde8")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect(
        (
            limits[0][1] - limits[0][0],
            limits[1][1] - limits[1][0],
            limits[2][1] - limits[2][0],
        )
    )
    ax.view_init(elev=elev, azim=azim)


def phase_name(progress):
    if progress < 0.33:
        return "1/3 Video frames -> 2D observations attached to ORB-SLAM3 points"
    if progress < 0.66:
        return "2/3 Sparse point-cloud modeling -> camera trajectory + growing map"
    return "3/3 YOLO-seg semantic voting -> labeled 3D points + object boxes"


def make_label_legend(labels, max_labels=8):
    shown = [label for label in labels if label != "unknown"][:max_labels]
    if not shown:
        return ""
    return "  ".join(shown)


def semantic_panel_text(objects, max_items):
    lines = ["SEMANTIC OBJECT LABELS"]
    for obj in objects[:max_items]:
        risk = object_risk(obj).upper()
        clearance = obj.get("clearance_to_path_network_m")
        clearance_text = "?" if clearance is None else f"{clearance:.2f}m"
        lines.append(f"{risk:6s}  {obj.get('id')}  {obj.get('label')}  clear={clearance_text}")
    return "\n".join(lines)


def render(args):
    run_dir = Path(args.run_dir)
    result_dir = Path(args.result_dir)
    semantic_map = read_json(run_dir / "intermediate" / "semantic_map.json")
    keyframes = read_jsonl(run_dir / "slam_export" / "keyframes.jsonl")
    observations = read_jsonl(run_dir / "slam_export" / "observations.jsonl")
    scene = read_json(result_dir / "scene.json")

    points = semantic_map["points"]
    positions = np.array([point["position"] for point in points], dtype=np.float64)
    labels = [str(point.get("label", "unknown")) for point in points]
    unique_labels = sorted(set(labels), key=lambda item: (item == "unknown", item))
    colors = np.array([color_for_label(label) for label in labels], dtype=np.float64)
    gray_colors = np.array([color_for_label("unknown") for _ in labels], dtype=np.float64)
    point_by_id = {int(point["map_point_id"]): point for point in points}
    display_bboxes = build_display_bboxes(
        scene,
        positions,
        labels,
        percentile=args.bbox_percentile,
        min_points=args.bbox_min_points,
    )

    label_color_bgr = {
        label: tuple(int(round(component * 255)) for component in reversed(color_for_label(label)))
        for label in unique_labels
    }

    keyframes = sorted(keyframes, key=lambda item: (float(item.get("timestamp", 0.0)), int(item["keyframe_id"])))
    kf_index_by_id = {int(kf["keyframe_id"]): index for index, kf in enumerate(keyframes)}
    observations_by_kf = defaultdict(list)
    first_seen = np.full(len(points), len(keyframes) - 1, dtype=np.int64)
    point_index_by_id = {int(point["map_point_id"]): index for index, point in enumerate(points)}
    for obs in observations:
        keyframe_id = int(obs["keyframe_id"])
        observations_by_kf[keyframe_id].append(obs)
        point_index = point_index_by_id.get(int(obs["map_point_id"]))
        if point_index is not None:
            first_seen[point_index] = min(first_seen[point_index], kf_index_by_id.get(keyframe_id, len(keyframes) - 1))

    camera_centers = []
    valid_camera_indices = []
    for index, keyframe in enumerate(keyframes):
        center = camera_center_from_keyframe(keyframe)
        if center is not None:
            camera_centers.append(center)
            valid_camera_indices.append(index)
    camera_centers = np.array(camera_centers, dtype=np.float64)

    limits = compute_focus_limits(scene, positions, camera_centers, args)
    focus_mask = points_inside_limits(positions, limits) if args.clip_to_focus else np.ones(len(points), dtype=bool)
    clipped_points = int(len(points) - np.count_nonzero(focus_mask))
    print(
        "[render] focus_bounds="
        f"{args.focus_bounds} shown_points={int(np.count_nonzero(focus_mask))}/{len(points)} "
        f"clipped_outliers={clipped_points}"
    )

    nframes = int(args.frames)
    nframes = max(nframes, 2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100)
    fig.patch.set_facecolor("#05070d")
    grid = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58], wspace=0.02)
    ax_img = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")

    top_objects = sorted(
        scene.get("semantic_objects", []),
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2, "unknown": 3}.get(object_risk(item), 3),
            -int(item.get("support_points", 0)),
            item.get("id", ""),
        ),
    )[: args.max_object_labels]

    for frame_idx in range(nframes):
        progress = frame_idx / max(nframes - 1, 1)
        keyframe_index = int(round(progress * (len(keyframes) - 1)))
        keyframe = keyframes[keyframe_index]
        semantic_enabled = progress >= 0.33
        final_phase = progress >= 0.66

        ax_img.clear()
        ax3d.clear()
        fig.texts.clear()

        image_rgb = image_to_rgb(
            keyframe.get("image_path", ""),
            observations_by_kf.get(int(keyframe["keyframe_id"]), []),
            point_by_id,
            label_color_bgr,
            semantic_enabled,
        )
        ax_img.imshow(image_rgb)
        ax_img.axis("off")
        ax_img.set_title(
            f"Input video keyframe {keyframe_index + 1}/{len(keyframes)}",
            color="#eef3ff",
            fontsize=13,
            pad=10,
        )
        ax_img.text(
            0.02,
            0.96,
            "2D observations overlay\nsemantic color after YOLO voting",
            transform=ax_img.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="#eef3ff",
            bbox={"facecolor": "#05070d", "alpha": 0.55, "edgecolor": "#536074"},
        )

        elev = args.elev + args.elev_swing * math.sin(progress * math.tau)
        azim = args.azim_start + args.azim_sweep * progress
        setup_3d_axis(ax3d, limits, elev, azim)

        if final_phase:
            visible_mask = np.ones(len(points), dtype=bool)
        else:
            visible_mask = first_seen <= keyframe_index
        visible_mask &= focus_mask
        visible_positions = positions[visible_mask]
        visible_colors = colors[visible_mask] if semantic_enabled else gray_colors[visible_mask]
        visible_sizes = np.where(np.array(labels)[visible_mask] == "unknown", 4.0, 8.0 if semantic_enabled else 5.0)
        alpha = 0.86 if semantic_enabled else 0.48
        if len(visible_positions) > 0:
            ax3d.scatter(
                visible_positions[:, 0],
                visible_positions[:, 1],
                visible_positions[:, 2],
                c=visible_colors,
                s=visible_sizes,
                alpha=alpha,
                depthshade=False,
                linewidths=0,
            )

        if len(camera_centers) > 0:
            current_center = camera_center_from_keyframe(keyframe)
            path_until = [center for center, idx in zip(camera_centers, valid_camera_indices) if idx <= keyframe_index]
            if path_until:
                path_until = np.array(path_until)
                ax3d.plot(
                    path_until[:, 0],
                    path_until[:, 1],
                    path_until[:, 2],
                    color="#f8f1a6",
                    linewidth=2.2,
                    alpha=0.9,
                )
            if current_center is not None:
                ax3d.scatter(
                    [current_center[0]],
                    [current_center[1]],
                    [current_center[2]],
                    c=[(1.0, 1.0, 1.0)],
                    s=80,
                    marker="^",
                    edgecolors="#111111",
                    linewidths=0.8,
                    depthshade=False,
                )

        if final_phase:
            for obj in scene.get("semantic_objects", []):
                risk = object_risk(obj)
                color = risk_color(risk)
                bbox = display_bboxes.get(obj.get("id"), obj.get("bbox_3d"))
                if bbox:
                    draw_bbox(ax3d, bbox, color=color, linewidth=1.9 if risk == "high" else 1.3, alpha=0.94)
            for obj in top_objects:
                center = obj.get("center")
                if not center:
                    continue
                risk = object_risk(obj)
                ax3d.text(
                    center[0],
                    center[1],
                    center[2],
                    f"{obj.get('label')}\\n{obj.get('id')}\\n{risk.upper()}",
                    color=risk_color(risk),
                    fontsize=args.label_font_size,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "#05070d", "alpha": 0.72, "edgecolor": risk_color(risk), "pad": 3.2},
                )

        ax3d.set_title("3D semantic sparse map fly-through", color="#eef3ff", fontsize=14, pad=14)
        fig.text(
            0.5,
            0.965,
            f"{args.title}   |   {phase_name(progress)}",
            color="#f8fafc",
            fontsize=17,
            ha="center",
            va="top",
            weight="bold",
        )
        if final_phase:
            fig.text(
                0.985,
                0.92,
                semantic_panel_text(top_objects, args.label_panel_items),
                color="#eef3ff",
                fontsize=11,
                ha="right",
                va="top",
                family="monospace",
                bbox={"facecolor": "#05070d", "alpha": 0.78, "edgecolor": "#536074", "pad": 7.0},
            )
        elif semantic_enabled:
            fig.text(
                0.985,
                0.92,
                "SEMANTIC VOTING\ncolored 3D points = YOLO-seg labels\nsame MapPoint may receive votes\nfrom multiple keyframes",
                color="#eef3ff",
                fontsize=11,
                ha="right",
                va="top",
                bbox={"facecolor": "#05070d", "alpha": 0.72, "edgecolor": "#536074", "pad": 7.0},
            )
        else:
            fig.text(
                0.985,
                0.92,
                "POINT-CLOUD MODELING\ngray points grow as keyframes observe\nORB-SLAM3 MapPoints\nwhite triangle = current camera",
                color="#eef3ff",
                fontsize=11,
                ha="right",
                va="top",
                bbox={"facecolor": "#05070d", "alpha": 0.72, "edgecolor": "#536074", "pad": 7.0},
            )
        fig.text(
            0.5,
            0.028,
            (
                f"points={len(points)}  labeled={semantic_map['scene_summary'].get('map_points_labeled')}  "
                f"shown={int(np.count_nonzero(focus_mask))}  "
                f"keyframes={len(keyframes)}  objects={len(scene.get('semantic_objects', []))}  "
                f"labels: {make_label_legend(unique_labels)}"
            ),
            color="#cbd5e1",
            fontsize=10,
            ha="center",
            va="bottom",
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        if (frame_idx + 1) % 20 == 0 or frame_idx + 1 == nframes:
            print(f"[render] frame {frame_idx + 1}/{nframes}")

    writer.release()
    plt.close(fig)
    print(f"[render] wrote {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render a paper-style semantic point-cloud flight video.")
    parser.add_argument("--run-dir", required=True, help="External pipeline run dir containing slam_export/ and intermediate/.")
    parser.add_argument("--result-dir", required=True, help="Final semantics/results/<run_name> dir containing scene.json.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Video -> ORB-SLAM3 sparse map -> semantic navigation map")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument(
        "--focus-bounds",
        choices=["scene", "robust", "all"],
        default="scene",
        help="3D viewport policy: scene uses scene.json bounds, robust ignores raw MapPoint outliers, all shows everything.",
    )
    parser.add_argument("--robust-percentile", type=float, default=0.5)
    parser.add_argument(
        "--clip-to-focus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide raw MapPoint outliers outside the chosen viewport.",
    )
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--elev-swing", type=float, default=7.0)
    parser.add_argument("--azim-start", type=float, default=-60.0)
    parser.add_argument("--azim-sweep", type=float, default=300.0)
    parser.add_argument("--max-object-labels", type=int, default=8)
    parser.add_argument("--label-font-size", type=int, default=11)
    parser.add_argument("--label-panel-items", type=int, default=10)
    parser.add_argument(
        "--bbox-percentile",
        type=float,
        default=10.0,
        help="Use percentile object boxes for video display only; scene.json remains unchanged.",
    )
    parser.add_argument("--bbox-min-points", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
