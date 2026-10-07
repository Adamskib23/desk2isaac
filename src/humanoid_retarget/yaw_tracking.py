from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import list_rgb_frames, read_depth_m, read_episode_metadata, read_json
from humanoid_retarget.mask_tracking import _inside_workspace, file_sha256


def axis_difference(a, b):
    """Signed difference for an undirected cuboid axis (180-degree symmetry)."""
    return (np.asarray(a) - np.asarray(b) + np.pi / 2) % np.pi - np.pi / 2


def axis_mean(angles) -> float:
    values = np.asarray(angles, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Yaw mean needs finite observations")
    vector = np.mean(np.exp(2j * values))
    if abs(vector) < 0.5:
        raise ValueError("Yaw observations have no consistent axis")
    return float(np.angle(vector) / 2)


def desk_to_sim_yaw(yaw: float, retarget: dict) -> float:
    mapping = retarget.get("mapping", "desk_y_forward_desk_x_left")
    matrix = (np.eye(2) if mapping == "identity" else np.array([[0, 1], [-1, 0]])
              if mapping == "desk_y_forward_desk_x_left" else np.asarray(retarget["xy_matrix"], dtype=float))
    scale = float(retarget.get("scale_xy", 1))
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0
            or not np.allclose(matrix.T @ matrix, np.eye(2) * np.sum(matrix[:, 0] ** 2), atol=1e-6)
            or not np.isfinite(yaw) or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Yaw retargeting requires a non-reflecting uniform XY rotation/scale")
    direction = matrix @ np.array([np.cos(yaw), np.sin(yaw)])
    return float(axis_difference(np.arctan2(direction[1], direction[0]), 0))


def yaw_quaternion(yaw: float, *, downward: bool = False) -> list[float]:
    if not np.isfinite(yaw):
        raise ValueError("Yaw must be finite")
    c, s = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
    return [0.0, c, s, 0.0] if downward else [c, 0.0, 0.0, s]


def quaternion_yaw_tilt(quaternions) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(quaternions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=1) < 1e-6):
        raise ValueError("Object quaternion telemetry must be finite, non-zero wxyz")
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    return yaw, tilt


def fit_upright_yaw(points: np.ndarray, size_m) -> dict:
    """Fit the known rectangular upper face in metric desk XY, not image pixels."""
    points, size = np.asarray(points, dtype=float), np.asarray(size_m, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Yaw fitting needs positive finite object dimensions")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Yaw fitting needs finite XYZ points")
    if max(size[:2]) / min(size[:2]) < 1.15:
        return {"status": "ambiguous_square_footprint"}
    if len(points) < 50:
        return {"status": "insufficient_upper_face"}
    rectangle = cv2.minAreaRect(points[:, :2].astype(np.float32))
    dimensions = np.asarray(rectangle[1], dtype=float)
    if np.min(dimensions) < 1e-6:
        return {"status": "degenerate_upper_face"}
    angle = np.deg2rad(rectangle[2])
    costs = [np.linalg.norm(np.log(d / size[:2])) for d in (dimensions, dimensions[::-1])]
    swap = int(np.argmin(costs))
    fitted = dimensions[::-1] if swap else dimensions
    yaw = float(axis_difference(angle + swap * np.pi / 2, 0))
    hull = cv2.convexHull(points[:, :2].astype(np.float32))
    fill = float(cv2.contourArea(hull) / np.prod(dimensions))
    design = np.column_stack((points[:, :2] - points[:, :2].mean(axis=0), np.ones(len(points))))
    plane, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    tilt = float(np.arctan(np.linalg.norm(plane[:2])))
    residual = float(np.percentile(np.abs(design @ plane - points[:, 2]), 95))
    ratio = fitted / size[:2]
    diagnostic = {"fitted_size_xy_m": fitted.tolist(), "hull_fill_fraction": fill,
                  "top_plane_tilt_deg": float(np.rad2deg(tilt)), "top_plane_residual_p95_m": residual,
                  "upper_face_points": len(points), "axis_assignment_margin": float(abs(costs[0] - costs[1]))}
    if (rank < 3 or np.any(ratio < 0.75) or np.any(ratio > 1.22) or fill < 0.70
            or tilt > np.deg2rad(15) or residual > 0.004 or abs(costs[0] - costs[1]) < 0.035):
        return {**diagnostic, "status": "uncertain_geometry"}
    confidence = float(fill * np.exp(-min(costs)) * np.exp(-residual / 0.004))
    return {**diagnostic, "status": "observed", "desk_yaw_rad": yaw, "confidence": confidence}


def yaw_from_depth_mask(mask, depth, spec, calibration: WorkspaceCalibration) -> dict:
    if spec.shape != "cuboid":
        return {"status": "unobservable_symmetric_shape"}
    if depth is None:
        return {"status": "missing_depth"}
    if mask.shape != calibration.image_size_px[::-1] or depth.shape != mask.shape:
        raise ValueError("Yaw masks/depth must match calibrated colour dimensions")
    interior = cv2.erode((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    py, px = np.nonzero(interior & np.isfinite(depth) & (depth > 0.05))
    if len(px) < 50:
        return {"status": "insufficient_depth"}
    cloud = calibration.pixels_depth_to_desk_xyz(np.column_stack((px, py)), depth[py, px])
    keep = (np.isfinite(cloud).all(axis=1) & (cloud[:, 2] > calibration.table_z_m + 0.004)
            & (cloud[:, 2] < calibration.table_z_m + 0.6) & _inside_workspace(cloud[:, :2], calibration))
    foreground = np.zeros(mask.shape, np.uint8)
    foreground[py[keep], px[keep]] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if count < 2:
        return {"status": "no_workspace_foreground"}
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cloud = cloud[keep & (labels[py, px] == component)]
    if len(cloud) < max(50, spec.min_area_px):
        return {"status": "insufficient_component_depth"}
    top = float(np.percentile(cloud[:, 2], 90))
    # Only the upper face: visible side walls and hands cannot be flattened into a footprint.
    upper = cloud[(cloud[:, 2] >= top - min(0.006, spec.size_m[2] * 0.2))
                  & (cloud[:, 2] <= np.percentile(cloud[:, 2], 99))]
    upper = upper[::max(1, len(upper) // 6000)]
    return {**fit_upright_yaw(upper, spec.size_m), "upper_face_height_m": top}


def boundary_yaw(rows: list[dict], frame_count: int, fps: float, *, final: bool = False) -> dict:
    width = min(frame_count, max(3, int(np.ceil(0.4 * fps))))
    start = frame_count - width if final else 0
    observed = [r for r in rows if start <= r["frame_idx"] < start + width and r["status"] == "observed"]
    result = {"window_frames": width, "observations": len(observed), "passed": False}
    if len(observed) < np.ceil(0.75 * width):
        return {**result, "status": "insufficient_boundary_yaw"}
    try:
        centre = axis_mean([r["desk_yaw_rad"] for r in observed])
    except ValueError:
        return {**result, "status": "inconsistent_boundary_yaw"}
    supported = [r for r in observed if abs(axis_difference(r["desk_yaw_rad"], centre)) <= np.deg2rad(8)]
    edge = frame_count - 3 if final else 3
    if (len(supported) < np.ceil(0.75 * width)
            or not any(r["frame_idx"] >= edge if final else r["frame_idx"] < edge for r in supported)):
        return {**result, "status": "unstable_boundary_yaw"}
    centre = axis_mean([r["desk_yaw_rad"] for r in supported])
    return {**result, "status": "observed", "passed": True, "desk_yaw_rad": centre,
            "source_frame_indices": [r["frame_idx"] for r in supported],
            "axis_jitter_p95_deg": float(np.rad2deg(np.percentile(
                np.abs(axis_difference([r["desk_yaw_rad"] for r in supported], centre)), 95)))}


def attach_yaw(task: dict, observations: list[dict], frame_count: int) -> dict:
    result = deepcopy(task)
    observations = deepcopy(observations)
    initial = boundary_yaw(observations, frame_count, float(task["fps"]))
    final = boundary_yaw(observations, frame_count, float(task["fps"]), final=True)
    supported = task["object"].get("shape") == "cuboid" and initial["passed"] and final["passed"]
    orientation = {"schema_version": "hmnd_upright_yaw_v1", "mode": "observed_yaw" if supported else "position_only",
                   "symmetry_period_rad": float(np.pi), "initial": initial, "final": final,
                   "frames": observations, "control_knots": [], "training_ready": False,
                   "notes": "Upright shape-axis yaw modulo 180 degrees; no measured roll/pitch or semantic front. "
                            "Smoothed/interpolated control yaw is not an observed teacher label."}
    result["orientation"] = orientation
    by_frame = {r["frame_idx"]: r for r in observations}
    for row in result.get("object_trace", []):
        measured = by_frame.get(row["frame_idx"], {})
        row.update(yaw_observed=measured.get("status") == "observed", desk_yaw_rad=measured.get("desk_yaw_rad"))
    if not supported:
        return result
    retarget = task["workspace"]["retarget"]
    start = desk_to_sim_yaw(initial["desk_yaw_rad"], retarget)
    goal = start + float(axis_difference(desk_to_sim_yaw(final["desk_yaw_rad"], retarget), start))
    accepted = {r["frame_idx"] for r in task.get("object_trace", []) if r.get("visible", True)}
    measured = [r for r in observations if r["status"] == "observed" and r["frame_idx"] in accepted
                and task["events"]["pick_frame"] < r["frame_idx"] < task["events"]["place_frame"]]
    knots = [{"frame_idx": 0, "sim_yaw_rad": start}]
    previous, previous_frame = start, 0
    for i, row in enumerate(measured):
        try:
            local = axis_mean([r["desk_yaw_rad"] for r in measured[max(0, i - 2):i + 3]])
        except ValueError:
            row["control_rejected"] = "inconsistent_local_axis"
            continue
        if abs(axis_difference(row["desk_yaw_rad"], local)) > np.deg2rad(12):
            row["control_rejected"] = "temporal_outlier"
            continue
        angle = previous + float(axis_difference(desk_to_sim_yaw(local, retarget), previous))
        elapsed = (row["frame_idx"] - previous_frame) / float(task["fps"])
        if abs(angle - previous) > 2.0 * elapsed + np.deg2rad(4):
            row["control_rejected"] = "implausible_angular_velocity"
            continue
        knots.append({"frame_idx": row["frame_idx"], "sim_yaw_rad": angle})
        previous, previous_frame = angle, row["frame_idx"]
    goal = previous + float(axis_difference(goal, previous))
    knots.append({"frame_idx": frame_count - 1, "sim_yaw_rad": goal})
    orientation.update(start_sim_yaw_rad=start, goal_sim_yaw_rad=goal, control_knots=knots)
    return result


def estimate_episode_yaw(task: dict, spec, calibration: WorkspaceCalibration) -> dict:
    episode = Path(task["source_episode"])
    metadata = read_episode_metadata(episode)
    if metadata.get("depth_aligned_to") != "color":
        raise ValueError("Yaw tracking requires colour-aligned depth")
    intr = metadata.get("color_intrinsics")
    if intr is not None and (calibration.image_size_px != (intr["width"], intr["height"])
                            or not np.allclose(calibration.camera_matrix_px,
                                [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], rtol=1e-5, atol=1e-4)):
        raise ValueError("Yaw capture intrinsics differ from calibration")
    if "color_distortion" in metadata and not np.allclose(metadata["color_distortion"], calibration.camera_distortion):
        raise ValueError("Yaw capture distortion differs from calibration")
    frames = list_rgb_frames(episode)
    mask_path = Path(task["events"]["mask_manifest"])
    manifest = read_json(mask_path)
    if (manifest.get("completed") is not True or manifest.get("schema_version") != "hmnd_sam2_masks_v1"
            or Path(manifest["source_episode"]).resolve() != episode.resolve()
            or manifest["object_name"] != spec.name or spec.name != task["object_name"]
            or not np.allclose(task["object"]["size_m"], spec.size_m)
            or manifest["source_metadata_sha256"] != file_sha256(episode / "metadata.json")
            or file_sha256(mask_path) != task["events"]["mask_manifest_sha256"]
            or len(manifest["frames"]) != len(frames)):
        raise ValueError("Yaw masks/configuration do not match the source task/capture")
    unit = float(metadata.get("depth_unit_m", 0.001))
    times = np.asarray(metadata.get("timestamps_relative_s", np.arange(len(frames)) / task["fps"]), dtype=float)
    if not np.isfinite(unit) or unit <= 0 or times.shape != (len(frames),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Yaw capture depth scale/timestamps must be valid")
    observations = []
    sources = {str(mask_path.resolve()): file_sha256(mask_path),
               str((episode / "metadata.json").resolve()): file_sha256(episode / "metadata.json")}
    for index, (rgb, entry) in enumerate(zip(frames, manifest["frames"])):
        mask_file = mask_path.parent / f"{index:06d}.png"
        if (entry["frame_idx"] != index or entry["rgb_name"] != rgb.name or entry["rgb_sha256"] != file_sha256(rgb)
                or entry["mask_sha256"] != file_sha256(mask_file)):
            raise ValueError(f"Yaw source fingerprint/order mismatch at frame {index}")
        sources[str(mask_file.resolve())] = entry["mask_sha256"]
        depth_file = episode / "depth" / (rgb.stem + ".png")
        if not depth_file.is_file():
            depth_file = depth_file.with_suffix(".npy")
        if depth_file.is_file():
            sources[str(depth_file.resolve())] = file_sha256(depth_file)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
            raise ValueError("Yaw requires binary uint8 SAM masks")
        estimate = yaw_from_depth_mask(mask, read_depth_m(episode, rgb.stem, unit), spec, calibration)
        observations.append({"frame_idx": index, "capture_t_s": float(times[index]), **estimate})
    result = attach_yaw(task, observations, len(frames))
    result["orientation"]["source_file_sha256"] = sources
    return result
