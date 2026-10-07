from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import list_rgb_frames, read_depth_m, read_episode_metadata, read_yaml
from humanoid_retarget.types import Detection, ObjectSpec


def object_specs_from_config(config: dict[str, Any]) -> list[ObjectSpec]:
    specs: list[ObjectSpec] = []
    for raw in config.get("objects", []):
        ranges = []
        for hsv_range in raw.get("hsv_ranges", []):
            lower = tuple(int(v) for v in hsv_range["lower"])
            upper = tuple(int(v) for v in hsv_range["upper"])
            ranges.append((lower, upper))
        if not ranges and "hsv_min" in raw and "hsv_max" in raw:
            ranges = [(tuple(raw["hsv_min"]), tuple(raw["hsv_max"]))]
        if not ranges:
            raise ValueError(f"Object {raw.get('name')} needs hsv_ranges or hsv_min/hsv_max")

        color_bgr = raw.get("color_bgr")
        if color_bgr is None and "color_rgb" in raw:
            rgb = raw["color_rgb"]
            color_bgr = [rgb[2], rgb[1], rgb[0]]
        if color_bgr is None:
            color_bgr = [40, 40, 220]

        specs.append(
            ObjectSpec(
                name=str(raw["name"]),
                shape=str(raw.get("shape", "cuboid")),
                size_m=tuple(float(v) for v in raw.get("size_m", (0.05, 0.05, 0.05))),
                hsv_ranges=ranges,
                color_bgr=tuple(int(v) for v in color_bgr),
                min_area_px=int(raw.get("min_area_px", 160)),
                base_diameter_m=(None if raw.get("base_diameter_m") is None
                                 else float(raw["base_diameter_m"])),
            )
        )
    if not specs:
        raise ValueError("No objects configured")
    return specs


def load_object_specs(path: str | Path) -> list[ObjectSpec]:
    return object_specs_from_config(read_yaml(path))


def hsv_mask(image_bgr: np.ndarray, spec: ObjectSpec) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for lower, upper in spec.hsv_ranges:
        mask |= cv2.inRange(hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_blob_centroid(mask: np.ndarray, min_area_px: int) -> tuple[tuple[float, float], float, np.ndarray] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area_px:
        return None
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return None
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    return (cx, cy), area, contour


def _sphere_center(points: np.ndarray, radius_m: float) -> np.ndarray | None:
    origin = np.median(points, axis=0)
    relative = points - origin
    matrix = np.column_stack((2 * relative, np.ones(len(points))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(relative ** 2, axis=1), rcond=None)
    if rank < 4:
        return None
    radius_squared = float(np.sum(solution[:3] ** 2) + solution[3])
    if radius_squared <= 0:
        return None
    radius = np.sqrt(radius_squared)
    center = origin + solution[:3]
    residual = np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2))
    if not 0.75 * radius_m <= radius <= 1.25 * radius_m or residual > 0.004:
        return None
    return center


def detect_object(
    image_bgr: np.ndarray,
    depth_m: np.ndarray | None,
    spec: ObjectSpec,
    calibration: WorkspaceCalibration,
    frame_idx: int,
    t: float,
) -> Detection | None:
    mask = hsv_mask(image_bgr, spec)
    if calibration.workspace_roi_px is not None:
        x, y, width, height = calibration.workspace_roi_px
        roi = np.zeros_like(mask)
        cv2.rectangle(roi, (int(x), int(y)), (int(x + width), int(y + height)), 255, -1)
        mask &= roi
    if calibration.workspace_outline_m is not None:
        polygon = np.asarray([calibration.desk_to_pixel_xy(p) for p in calibration.workspace_outline_m])
        workspace_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(workspace_mask, np.round(polygon).astype(np.int32), 255)
        mask &= workspace_mask
    for polygon in calibration.excluded_polygons_px:
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 0)
    blob = _largest_blob_centroid(mask, spec.min_area_px)
    if blob is None:
        return None
    (cx, cy), area, contour = blob
    component_mask = np.zeros_like(mask)
    cv2.drawContours(component_mask, [contour], -1, 255, -1)
    component_mask &= mask
    desk_xy = calibration.pixel_to_desk_xy((cx, cy))
    center_z = calibration.table_z_m + spec.size_m[2] * 0.5
    desk_xyz = calibration.pixel_to_desk_at_height((cx, cy), center_z)
    position_source = "nominal_height" if calibration.camera_to_desk_m is not None else "desk_plane"
    depth_quality = 1.0

    if depth_m is not None and calibration.camera_to_desk_m is not None:
        if depth_m.shape != image_bgr.shape[:2]:
            raise ValueError("Depth must be aligned to the RGB image; resizing unaligned depth is not calibration")
        interior = cv2.erode(component_mask, np.ones((3, 3), dtype=np.uint8))
        valid = (interior > 0) & np.isfinite(depth_m) & (depth_m > 0.05)
        depth_quality = float(valid.sum() / max(1, (interior > 0).sum()))
        py, px = np.nonzero(valid)
        if len(px) >= 20:
            stride = max(1, len(px) // 1200)
            px, py = px[::stride], py[::stride]
            points = calibration.pixels_depth_to_desk_xyz(np.column_stack((px, py)), depth_m[py, px])
            points = points[(points[:, 2] >= calibration.table_z_m - 0.015)
                            & (points[:, 2] < calibration.table_z_m + 0.6)]
            if len(points) >= 20:
                center = _sphere_center(points, spec.size_m[2] / 2) if spec.shape == "sphere" else None
                if center is not None:
                    desk_xyz = center
                    desk_xyz[2] = max(center_z, float(desk_xyz[2]))
                    position_source = "aligned_depth_sphere_fit"
                else:
                    desk_xyz = np.median(points, axis=0)
                    desk_xyz[2] = max(center_z, float(np.percentile(points[:, 2], 95)) - spec.size_m[2] / 2)
                    position_source = "aligned_depth_surface"

    elif depth_m is not None and calibration.table_depth_m is not None:
        h, w = image_bgr.shape[:2]
        if depth_m.shape[:2] != (h, w):
            raise ValueError("Depth must be aligned to the RGB image; resizing unaligned depth is not calibration")
        dilated = cv2.dilate(component_mask, np.ones((7, 7), dtype=np.uint8))
        valid_depth = depth_m[(dilated > 0) & np.isfinite(depth_m) & (depth_m > 0.05)]
        if valid_depth.size > 8:
            object_depth_m = float(np.median(valid_depth))
            height_above_table = max(0.0, calibration.table_depth_m - object_depth_m)
            center_z = calibration.table_z_m + max(spec.size_m[2] * 0.5, height_above_table - spec.size_m[2] * 0.5)
            desk_xyz = np.asarray([*desk_xy, center_z])
            position_source = "overhead_depth_approximation"

    confidence = float(np.clip(area / max(spec.min_area_px * 8.0, 1.0), 0.05, 1.0))
    if calibration.camera_to_desk_m is not None:
        confidence *= max(0.2, depth_quality)
        if position_source == "nominal_height":
            confidence = min(confidence, 0.3)
    return Detection(
        frame_idx=frame_idx,
        t=t,
        object_name=spec.name,
        pixel_xy=(cx, cy),
        desk_xyz_m=tuple(float(v) for v in desk_xyz),
        mask_area_px=area,
        confidence=confidence,
        position_source=position_source,
    )


def track_episode(
    episode_dir: str | Path,
    specs: list[ObjectSpec],
    calibration: WorkspaceCalibration,
    fps_override: float | None = None,
) -> tuple[list[Detection], dict[str, Any]]:
    episode_path = Path(episode_dir)
    metadata = read_episode_metadata(episode_path)
    rgb_frames = list_rgb_frames(episode_path)
    fps = float(fps_override or metadata.get("fps", 30.0))
    depth_unit_m = float(metadata.get("depth_unit_m", 0.001))
    if calibration.camera_to_desk_m is not None and metadata.get("depth_aligned_to") != "color":
        raise ValueError("3D tracking requires an episode with depth_aligned_to=color in its metadata")

    detections: list[Detection] = []
    for frame_idx, rgb_path in enumerate(rgb_frames):
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image {rgb_path}")
        depth_m = read_depth_m(episode_path, rgb_path.stem, depth_unit_m)
        t = frame_idx / fps
        for spec in specs:
            detection = detect_object(image, depth_m, spec, calibration, frame_idx, t)
            if detection is not None:
                detections.append(detection)

    metadata = {
        **metadata,
        "fps": fps,
        "num_rgb_frames": len(rgb_frames),
        "episode_dir": str(episode_path),
    }
    return detections, metadata


def detections_by_object(detections: list[Detection]) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.object_name].append(detection)
    for name in grouped:
        grouped[name].sort(key=lambda d: d.frame_idx)
    return dict(grouped)


def filter_detections(detections: list[Detection], fps: float) -> list[Detection]:
    """Causal Kalman tracking of metric and pixel positions, retaining uncertainty."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Tracking FPS must be positive and finite")
    filtered = []
    measurement_std = np.asarray([0.006, 0.006, 0.005, 4.0, 4.0])
    acceleration_std = np.asarray([2.0, 2.0, 2.0, 1400.0, 1400.0])
    for rows in detections_by_object(detections).values():
        kalman = None
        previous_frame = None
        rejected = 0
        for detection in rows:
            measurement = np.asarray([*detection.desk_xyz_m, *detection.pixel_xy], dtype=np.float64)
            dt = (detection.frame_idx - previous_frame) / fps if previous_frame is not None else 1 / fps
            if not np.isfinite(measurement).all() or dt <= 0:
                raise ValueError("Tracking requires finite observations in distinct, increasing frames")
            if kalman is None or dt > 0.5 or rejected >= max(3, int(0.5 * fps)):
                kalman = cv2.KalmanFilter(10, 5, 0, cv2.CV_64F)
                kalman.measurementMatrix = np.column_stack((np.eye(5), np.zeros((5, 5))))
                kalman.statePost = np.concatenate((measurement, np.zeros(5))).reshape(10, 1)
                kalman.errorCovPost = np.diag(np.concatenate((measurement_std ** 2, acceleration_std ** 2)))
                rejected = 0
                filtered.append(detection)
                previous_frame = detection.frame_idx
                continue
            kalman.transitionMatrix = np.block([[np.eye(5), dt * np.eye(5)], [np.zeros((5, 5)), np.eye(5)]])
            acceleration = np.diag(acceleration_std ** 2)
            kalman.processNoiseCov = np.block([
                [dt ** 4 / 4 * acceleration, dt ** 3 / 2 * acceleration],
                [dt ** 3 / 2 * acceleration, dt ** 2 * acceleration],
            ])
            kalman.measurementNoiseCov = np.diag((measurement_std / max(0.25, detection.confidence)) ** 2)
            prediction = kalman.predict().reshape(-1)
            innovation = measurement[:3] - prediction[:3]
            covariance = kalman.errorCovPre[:3, :3] + kalman.measurementNoiseCov[:3, :3]
            accepted = float(innovation @ np.linalg.solve(covariance, innovation)) <= 36.0
            if accepted:
                estimate = kalman.correct(measurement.reshape(5, 1)).reshape(-1)
                rejected = 0
            else:
                # A gated-out measurement is a prediction, not a visible observation.
                estimate = prediction
                kalman.statePost = kalman.statePre.copy()
                kalman.errorCovPost = kalman.errorCovPre.copy()
                rejected += 1
            filtered.append(replace(
                detection,
                desk_xyz_m=tuple(float(v) for v in estimate[:3]),
                pixel_xy=tuple(float(v) for v in estimate[3:5]),
                visible=detection.visible and accepted,
                confidence=detection.confidence if accepted else min(0.2, detection.confidence),
                position_source=detection.position_source if accepted else "kalman_prediction",
            ))
            previous_frame = detection.frame_idx
    return sorted(filtered, key=lambda d: (d.frame_idx, d.object_name))


def select_active_object(
    detections: list[Detection],
    min_displacement_m: float = 0.035,
) -> tuple[str, dict[str, float]]:
    grouped = detections_by_object(detections)
    if not grouped:
        raise ValueError("No object detections found")

    scores: dict[str, float] = {}
    eligible: list[str] = []
    for name, object_detections in grouped.items():
        positions = np.asarray([d.desk_xyz_m for d in object_detections], dtype=np.float32)
        if len(positions) < 3:
            scores[name] = 0.0
            continue
        initial = np.median(positions[: min(10, len(positions))], axis=0)
        final = np.median(positions[-min(10, len(positions)) :], axis=0)
        displacement = float(np.linalg.norm(final[:2] - initial[:2]))
        path_length = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
        motion_extent = float(np.percentile(np.linalg.norm(positions - initial, axis=1), 95))
        if motion_extent >= min_displacement_m or displacement >= min_displacement_m:
            eligible.append(name)
        visibility_bonus = min(0.03, len(object_detections) * 0.0003)
        scores[name] = displacement + 0.2 * path_length + visibility_bonus

    if not eligible:
        raise ValueError(
            f"No object moved more than {min_displacement_m:.3f} m. Scores: "
            + ", ".join(f"{name}={score:.3f}" for name, score in scores.items())
        )
    best_name = max(eligible, key=scores.get)
    return best_name, scores
