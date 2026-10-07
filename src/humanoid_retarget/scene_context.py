from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from humanoid_retarget.calibration import calibration_from_config
from humanoid_retarget.io import list_rgb_frames, read_episode_metadata, read_json, read_yaml, read_depth_m, write_json
from humanoid_retarget.mask_tracking import detection_from_depth_mask, file_sha256
from humanoid_retarget.planner import _desk_to_sim_xyz
from humanoid_retarget.tracking import hsv_mask, load_object_specs
from humanoid_retarget.yaw_tracking import axis_difference, axis_mean, desk_to_sim_yaw, yaw_from_depth_mask, yaw_quaternion


def marker_layout(workspace_config: dict, retarget: dict) -> list[dict]:
    workspace = workspace_config.get("workspace", workspace_config)
    reference = workspace.get("calibration_reference", {})
    ids = reference.get("marker_ids", [])
    if not ids:
        return []
    dictionary_name = reference["dictionary"]
    if not dictionary_name.startswith("DICT_") or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError("Unknown ArUco dictionary")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    size = float(reference["marker_size_m"]) * float(retarget.get("scale_xy", 1))
    corners = np.asarray(workspace["desk_points_m"], dtype=float)
    if corners.shape != (4 * len(ids), 2) or not np.isfinite(corners).all() or not np.isfinite(size) or size <= 0:
        raise ValueError("Marker layout requires four ordered desk corners per marker and a positive size")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate marker IDs")
    result = []
    table_z = float(retarget.get("sim_table_z_m", retarget.get("sim_origin_m", [0, 0, 0])[2]))
    for index, marker_id in enumerate(ids):
        desk_corners = corners[index * 4:index * 4 + 4]
        sim_corners = np.asarray([_desk_to_sim_xyz(np.r_[xy, workspace.get("table_z_m", 0)], retarget)
                                  for xy in desk_corners], dtype=float)
        right = sim_corners[1, :2] - sim_corners[0, :2]
        length = np.linalg.norm(right)
        if length < 1e-6:
            raise ValueError("Degenerate marker corners")
        right /= length
        up = np.asarray([-right[1], right[0]])
        if np.dot(up, sim_corners[0, :2] - sim_corners[3, :2]) <= 0:
            raise ValueError("Mirrored/degenerate marker mapping would change the ArUco pattern")
        centre = sim_corners[:, :2].mean(axis=0)
        yaw = float(np.arctan2(right[1], right[0]))
        cells = dictionary.markerSize + 2
        bits = cv2.aruco.generateImageMarker(dictionary, int(marker_id), cells)
        square = [centre + size / 2 * (x * right + y * up) for x, y in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
        result.append({"id": int(marker_id), "dictionary": dictionary_name, "black_square_size_m": size,
                       "initial_sim_xyz_m": [*centre.tolist(), table_z],
                       "quat_wxyz": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
                       "sim_corners_m": [p.tolist() for p in square], "cells": (bits == 0).astype(int).tolist(),
                       "collision_enabled": False})
    return result


def marker_visual_parts(marker: dict) -> list[dict]:
    """Render dictionary bits as flat, non-colliding geometry with a white quiet zone."""
    size = marker["black_square_size_m"]
    centre = np.asarray(marker["initial_sim_xyz_m"], dtype=float)
    w, _, _, z = marker["quat_wxyz"]
    right, up = np.asarray([w * w - z * z, 2 * w * z]), np.asarray([-2 * w * z, w * w - z * z])
    parts = [{"name": "Paper", "size_m": [size + 0.02, size + 0.02, 0.0002],
              "pos_m": (centre + [0, 0, 0.00015]).tolist(), "color_rgb": [1.0, 1.0, 1.0],
              "quat_wxyz": marker["quat_wxyz"]}]
    cells = np.asarray(marker["cells"], dtype=int)
    count = len(cells)
    if cells.shape != (count, count) or not np.isin(cells, (0, 1)).all():
        raise ValueError("Marker cells must be a square binary grid")
    step = size / count
    for row in range(count):
        column = 0
        while column < count:
            if not cells[row, column]:
                column += 1
                continue
            start = column
            while column < count and cells[row, column]:
                column += 1
            offset = step * ((start + column) / 2 - count / 2) * right + step * (count / 2 - row - 0.5) * up
            parts.append({"name": f"Bits_{row}_{start}", "size_m": [(column - start) * step, step, 0.0001],
                          "pos_m": [centre[0] + offset[0], centre[1] + offset[1], centre[2] + 0.0003],
                          "color_rgb": [0.0, 0.0, 0.0], "quat_wxyz": marker["quat_wxyz"]})
    return parts


def stationary_reference(spec, initial: list, final: list, retarget: dict, table_z: float) -> tuple[dict | None, dict]:
    diagnostic = {"name": spec.name, "initial_observations": len(initial), "final_observations": len(final)}
    if min(len(initial), len(final)) < 4:
        return None, {**diagnostic, "status": "insufficient_boundary_depth"}
    stable = []
    rejected = []
    for rows in (initial, final):
        points = np.asarray([d.desk_xyz_m for d in rows])
        median = np.median(points, axis=0)
        supported = (np.isfinite(points).all(axis=1)
                     & (np.linalg.norm(points[:, :2] - median[:2], axis=1) <= 0.005)
                     & (np.abs(points[:, 2] - (table_z + spec.size_m[2] / 2)) <= 0.015))
        rejected.extend(d.frame_idx for d, keep in zip(rows, supported) if not keep)
        stable.append(points[supported])
    diagnostic.update(initial_consensus_observations=len(stable[0]), final_consensus_observations=len(stable[1]),
                      rejected_boundary_frames=rejected)
    if min(map(len, stable)) < 4:
        return None, {**diagnostic, "status": "unstable_moved_or_not_resting"}
    start, end = stable
    centre = np.median(start, axis=0)
    shift = float(np.linalg.norm(np.median(end, axis=0)[:2] - centre[:2]))
    jitter = max(float(np.percentile(np.linalg.norm(p[:, :2] - np.median(p, axis=0)[:2], axis=1), 95))
                 for p in (start, end))
    height_error = max(float(np.percentile(np.abs(p[:, 2] - (table_z + spec.size_m[2] / 2)), 95))
                       for p in (start, end))
    diagnostic.update(boundary_shift_m=shift, boundary_jitter_p95_m=jitter, resting_height_error_p95_m=height_error)
    if jitter > 0.005 or shift > 0.012 or height_error > 0.015:
        return None, {**diagnostic, "status": "unstable_moved_or_not_resting"}
    centre[2] = table_z + spec.size_m[2] / 2
    position = _desk_to_sim_xyz(centre, retarget).tolist()
    position[2] = float(retarget.get("sim_table_z_m", position[2] - centre[2])) + spec.size_m[2] / 2
    return {**asdict(spec), "initial_sim_xyz_m": position, "initial_desk_xyz_m": centre.tolist(),
            "initial_pixel_xy": np.median([d.pixel_xy for d in initial], axis=0).tolist(),
            "position_source": "boundary_hsv_aligned_depth_visual_reference", "collision_enabled": False,
            "training_ready": False, "orientation_assumption": "upright; yaw unmeasured"}, {**diagnostic, "status": "visual_reference"}


def prepare_scene_context(task_path: Path, workspace_path: Path, objects_path: Path, output: Path) -> dict:
    task = read_json(task_path)
    episode = Path(task["source_episode"])
    if output.resolve().is_relative_to(episode.resolve()) or output.exists():
        raise ValueError("Choose a fresh scene-context output outside the source recording")
    config = read_yaml(workspace_path)
    calibration = calibration_from_config(config)
    specs = load_object_specs(objects_path)
    for key, path in (("workspace_config_sha256", workspace_path), ("objects_config_sha256", objects_path)):
        if task.get("events", {}).get(key) and file_sha256(path) != task["events"][key]:
            raise ValueError(f"Scene context {key} differs from the trajectory")
    metadata = read_episode_metadata(episode)
    if metadata.get("depth_aligned_to") != "color":
        raise ValueError("Background scene references require colour-aligned depth")
    intrinsics = metadata.get("color_intrinsics")
    if intrinsics is not None:
        matrix = [[intrinsics["fx"], 0, intrinsics["cx"]], [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]]
        if ((intrinsics["width"], intrinsics["height"]) != calibration.image_size_px
                or not np.allclose(matrix, calibration.camera_matrix_px, rtol=1e-5, atol=1e-4)):
            raise ValueError("Scene reference intrinsics differ from calibration")
    if "color_distortion" in metadata and not np.allclose(metadata["color_distortion"], calibration.camera_distortion):
        raise ValueError("Scene reference distortion differs from calibration")
    frames = list_rgb_frames(episode)
    width = min(len(frames) // 2, max(6, round(float(task["fps"]) * 0.4)))
    if width < 6:
        raise ValueError("Scene reference requires at least twelve source frames")
    windows = [np.unique(np.linspace(start, start + width - 1, 6).astype(int))
               for start in (0, len(frames) - width)]
    background_specs = [s for s in specs if s.name != task["object_name"]]
    observations = {s.name: [[], []] for s in background_specs}
    yaw_observations = {s.name: [[], []] for s in background_specs}
    sources = {str(path.resolve()): file_sha256(path) for path in (task_path, workspace_path, objects_path, episode / "metadata.json")}
    for boundary, indices in enumerate(windows):
        for index in indices:
            rgb_path = frames[index]
            image = cv2.imread(str(rgb_path))
            if image is None or image.shape[:2] != calibration.image_size_px[::-1]:
                raise ValueError("Scene reference image differs from calibrated dimensions")
            depth = read_depth_m(episode, rgb_path.stem, float(metadata.get("depth_unit_m", 0.001)))
            sources[str(rgb_path.resolve())] = file_sha256(rgb_path)
            depth_path = episode / "depth" / (rgb_path.stem + ".png")
            if not depth_path.is_file():
                depth_path = depth_path.with_suffix(".npy")
            if depth_path.is_file():
                sources[str(depth_path.resolve())] = file_sha256(depth_path)
            for spec in background_specs:
                # Colour proposes a region; depth removes the desk/shadow after expanding faint object edges.
                mask = cv2.dilate(hsv_mask(image, spec), np.ones((17, 17), np.uint8))
                detection, _ = detection_from_depth_mask(mask, depth, spec, calibration, int(index), 0)
                if detection is not None:
                    observations[spec.name][boundary].append(detection)
                    yaw = yaw_from_depth_mask(mask, depth, spec, calibration)
                    if yaw["status"] == "observed":
                        yaw_observations[spec.name][boundary].append({"frame_idx": int(index), **yaw})
    retarget = task["workspace"]["retarget"]
    backgrounds, diagnostics = [], []
    for spec in background_specs:
        obj, diagnostic = stationary_reference(spec, *observations[spec.name], retarget, calibration.table_z_m)
        if obj is not None:
            obj["yaw_source"] = "unmeasured_position_only"
            initial_yaw, final_yaw = yaw_observations[spec.name]
            if min(len(initial_yaw), len(final_yaw)) >= 4:
                try:
                    a, b = (axis_mean([r["desk_yaw_rad"] for r in rows]) for rows in (initial_yaw, final_yaw))
                    stable_yaw = (abs(axis_difference(a, b)) <= np.deg2rad(10)
                                  and all(np.max(np.abs(axis_difference([r["desk_yaw_rad"] for r in rows], mean))) <= np.deg2rad(8)
                                          for rows, mean in ((initial_yaw, a), (final_yaw, b))))
                    if stable_yaw:
                        sim_yaw = desk_to_sim_yaw(a, retarget)
                        obj.update(initial_desk_yaw_rad=a, initial_sim_yaw_rad=sim_yaw,
                                   initial_quat_wxyz=yaw_quaternion(sim_yaw), yaw_source="boundary_depth_upper_face_shape_axis",
                                   yaw_source_frame_indices=[r["frame_idx"] for r in initial_yaw],
                                   orientation_assumption="upright; measured shape-axis yaw modulo 180 degrees")
                except ValueError:
                    pass
            backgrounds.append(obj)
        diagnostics.append(diagnostic)
    payload = {"schema_version": "hmnd_scene_context_v1", "source_episode": str(episode.resolve()),
               "source_task_sha256": file_sha256(task_path), "selected_object": task["object_name"],
               "markers": marker_layout(config, retarget), "background_objects": backgrounds,
               "background_diagnostics": diagnostics, "source_file_sha256": sources,
               "collision_enabled": False, "training_ready": False,
               "boundary_mask_method": "HSV proposals expanded 8 px; calibrated depth foreground and known-size checks",
               "notes": "Static visual references only. Boundary colour/depth estimates are not approved labels or obstacle poses; objects are not tracked through the carry. Marker centres/orientations use the task calibration; white margin is illustrative. Cuboid yaw is applied only with consistent boundary geometry; other orientations remain unmeasured."}
    write_json(output, payload)
    preview = cv2.imread(str(frames[0]))
    for index, marker in enumerate(payload["markers"]):
        cv2.putText(preview, f"marker {marker['id']}", tuple(np.round(calibration.desk_to_pixel_xy(
            np.asarray(config["workspace"]["desk_points_m"])[index * 4:index * 4 + 4].mean(axis=0))).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    for obj in backgrounds:
        cx, cy = np.round(obj["initial_pixel_xy"]).astype(int)
        cv2.drawMarker(preview, (int(cx), int(cy)), (30, 230, 80), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(preview, obj["name"] + " (static)", (int(cx) + 14, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 230, 80), 2)
    for index, diagnostic in enumerate(diagnostics):
        if diagnostic["status"] != "visual_reference":
            cv2.putText(preview, diagnostic["name"] + ": omitted (" + diagnostic["status"] + ")",
                        (25, 40 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 180, 255), 2)
    preview = cv2.resize(preview, (960, round(preview.shape[0] * 960 / preview.shape[1])))
    if not cv2.imwrite(str(output.with_name("scene_context_preview.jpg")), preview):
        raise OSError("Could not write scene-context preview")
    return payload


def load_scene_context(path: Path, task_path: Path) -> dict:
    context = read_json(path)
    task = read_json(task_path)
    if (context.get("schema_version") != "hmnd_scene_context_v1"
            or context["selected_object"] != task["object_name"]
            or Path(context["source_episode"]).resolve() != Path(task["source_episode"]).resolve()
            or context["source_task_sha256"] != file_sha256(task_path)):
        raise ValueError("Scene context does not match the source task")
    if context.get("collision_enabled") is not False:
        raise ValueError("Scene context must remain visual-only")
    for source, expected in context["source_file_sha256"].items():
        source_path = Path(source)
        if context.get("source_files_relative_to") == "context":
            source_path = (path.parent / source).resolve()
            if Path(source).is_absolute() or not source_path.is_relative_to(path.parent.resolve()):
                raise ValueError("Scene reference must remain inside the packaged scene directory")
        if file_sha256(source_path) != expected:
            raise ValueError(f"Scene reference source changed: {source}")
    names = [obj["name"] for obj in context["background_objects"]]
    if len(names) != len(set(names)) or task["object_name"] in names:
        raise ValueError("Duplicate/active object in background scene")
    return context
