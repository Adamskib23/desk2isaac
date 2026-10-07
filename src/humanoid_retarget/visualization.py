from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import list_rgb_frames
from humanoid_retarget.tracking import hsv_mask
from humanoid_retarget.types import Detection, ObjectSpec


def save_detection_video(
    episode_dir: str | Path,
    detections: list[Detection],
    specs: list[ObjectSpec],
    output_path: str | Path,
    fps: float,
) -> None:
    rgb_frames = list_rgb_frames(episode_dir)
    if not rgb_frames:
        return
    first = cv2.imread(str(rgb_frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        return
    h, w = first.shape[:2]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame: dict[int, list[Detection]] = {}
    for detection in detections:
        by_frame.setdefault(detection.frame_idx, []).append(detection)
    spec_by_name = {spec.name: spec for spec in specs}

    for idx, frame_path in enumerate(rgb_frames):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        for detection in by_frame.get(idx, []):
            spec = spec_by_name[detection.object_name]
            color = tuple(int(c) for c in spec.color_bgr)
            x, y = [int(round(v)) for v in detection.pixel_xy]
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 14, color, 2)
            label = f"{detection.object_name} {detection.confidence:.2f}"
            cv2.putText(frame, label, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def save_workspace_overlay(
    image_bgr: np.ndarray,
    calibration: WorkspaceCalibration,
    output_path: str | Path,
) -> None:
    frame = image_bgr.copy()
    outline = calibration.workspace_outline_m or calibration.desk_points_m
    pts = np.round([calibration.desk_to_pixel_xy(p) for p in outline]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    for idx, pt in enumerate(pts.reshape(-1, 2)):
        cv2.circle(frame, tuple(int(v) for v in pt), 5, (0, 255, 255), -1)
        cv2.putText(frame, str(idx), tuple(int(v) for v in pt + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)


def rectify_workspace(
    image_bgr: np.ndarray,
    calibration: WorkspaceCalibration,
    width_px: int = 1200,
) -> np.ndarray:
    """Render a rectangular desk plane top-down, including lens correction."""
    outline = np.asarray(calibration.workspace_outline_m, dtype=np.float64)
    if outline.shape != (4, 2) or width_px < 2:
        raise ValueError("Top-down workspace requires a four-corner rectangle and width >= 2")
    if image_bgr.shape[:2] != calibration.image_size_px[::-1]:
        raise ValueError("Image dimensions do not match the workspace calibration")
    edges = np.roll(outline, -1, axis=0) - outline
    lengths = np.linalg.norm(edges, axis=1)
    if (np.min(lengths) < 1e-6 or not np.allclose(edges[0], -edges[2], atol=1e-6)
            or not np.allclose(edges[1], -edges[3], atol=1e-6)
            or abs(edges[0] @ edges[1]) > 1e-5 * lengths[0] * lengths[1]):
        raise ValueError("Top-down workspace outline must be a metric rectangle")
    pixels = np.asarray([calibration.desk_to_pixel_xy(p) for p in outline])
    # Keep the longer side horizontal and orient it like the camera preview.
    long_edges = np.flatnonzero(np.isclose(lengths, lengths.max(), rtol=1e-6))
    top = min(long_edges, key=lambda i: (pixels[i, 1] + pixels[(i + 1) % 4, 1]) / 2)
    left, right = top, (top + 1) % 4
    if pixels[left, 0] > pixels[right, 0]:
        left, right = right, left
    bottom_left = (left - 1) % 4 if (left + 1) % 4 == right else (left + 1) % 4
    horizontal = outline[right] - outline[left]
    vertical = outline[bottom_left] - outline[left]
    height_px = max(2, int(round((width_px - 1) * np.linalg.norm(vertical) / np.linalg.norm(horizontal))) + 1)
    u, v = np.meshgrid(np.linspace(0, 1, width_px), np.linspace(0, 1, height_px))
    desk_xy = outline[left] + u[..., None] * horizontal + v[..., None] * vertical
    if calibration.camera_to_desk_m is not None:
        desk_xyz = np.column_stack((desk_xy.reshape(-1, 2), np.full(u.size, calibration.table_z_m)))
        transform = np.asarray(calibration.camera_to_desk_m)
        camera_xyz = (desk_xyz - transform[:3, 3]) @ transform[:3, :3]
        distortion = np.asarray(calibration.camera_distortion)
        mapped, _ = cv2.projectPoints(
            camera_xyz, np.zeros(3), np.zeros(3), np.asarray(calibration.camera_matrix_px),
            distortion if distortion.size else None,
        )
    else:
        mapped = cv2.perspectiveTransform(desk_xy.reshape(-1, 1, 2), calibration.desk_to_pixel_h)
    mapped = mapped.reshape(height_px, width_px, 2).astype(np.float32)
    return cv2.remap(image_bgr, mapped[:, :, 0], mapped[:, :, 1], cv2.INTER_LINEAR)


def _make_mapper(points: np.ndarray, width: int, height: int, pad: int = 64):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])

    def map_xy(xy: np.ndarray) -> tuple[int, int]:
        p = (xy - mins) * scale + pad
        return int(round(p[0])), int(round(height - p[1]))

    return map_xy


def render_topdown_replay(
    trajectory: dict[str, Any],
    video_path: str | Path,
    image_path: str | Path | None = None,
    fps: float = 30.0,
) -> None:
    waypoints = trajectory["waypoints"]
    trace_points = np.asarray([row["sim_xyz_m"][:2] for row in trajectory["object_trace"]], dtype=np.float32)
    ee_points = np.asarray([wp["ee_xyz_m"][:2] for wp in waypoints], dtype=np.float32)
    base = np.asarray([[0.0, 0.0]], dtype=np.float32)
    all_points = np.vstack([trace_points, ee_points, base])
    width, height = 960, 720
    mapper = _make_mapper(all_points, width, height)
    output = Path(video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    duration = max(float(waypoints[-1]["t"]), 1.0)
    num_frames = int(np.ceil(duration * fps))
    wp_t = np.asarray([float(wp["t"]) for wp in waypoints], dtype=np.float32)
    wp_ee = np.asarray([wp["ee_xyz_m"] for wp in waypoints], dtype=np.float32)
    wp_obj = np.asarray([wp["object_xyz_m"] for wp in waypoints], dtype=np.float32)

    last_frame = None
    for frame_idx in range(num_frames + 1):
        t = frame_idx / fps
        ee = np.asarray([np.interp(t, wp_t, wp_ee[:, dim]) for dim in range(3)], dtype=np.float32)
        obj = np.asarray([np.interp(t, wp_t, wp_obj[:, dim]) for dim in range(3)], dtype=np.float32)
        canvas = np.full((height, width, 3), (245, 245, 241), dtype=np.uint8)
        for x in range(80, width, 80):
            cv2.line(canvas, (x, 0), (x, height), (226, 226, 220), 1)
        for y in range(80, height, 80):
            cv2.line(canvas, (0, y), (width, y), (226, 226, 220), 1)

        for a, b in zip(trace_points[:-1], trace_points[1:]):
            cv2.line(canvas, mapper(a), mapper(b), (70, 110, 220), 2)
        for a, b in zip(ee_points[:-1], ee_points[1:]):
            cv2.line(canvas, mapper(a), mapper(b), (70, 70, 70), 1)

        base_px = mapper(base[0])
        ee_px = mapper(ee[:2])
        obj_px = mapper(obj[:2])
        cv2.circle(canvas, base_px, 14, (35, 35, 35), -1)
        cv2.line(canvas, base_px, ee_px, (40, 40, 40), 5)
        cv2.circle(canvas, ee_px, 12, (35, 35, 35), -1)
        cv2.rectangle(canvas, (obj_px[0] - 13, obj_px[1] - 13), (obj_px[0] + 13, obj_px[1] + 13), (50, 50, 220), -1)
        cv2.putText(canvas, f"t={t:4.2f}s", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 45, 45), 2)
        cv2.putText(
            canvas,
            f"{trajectory['object_name']} | {waypoints[min(np.searchsorted(wp_t, t), len(waypoints)-1)]['phase']}",
            (28, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (45, 45, 45),
            2,
        )
        writer.write(canvas)
        last_frame = canvas
    writer.release()
    if image_path is not None and last_frame is not None:
        out_img = Path(image_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), last_frame)
