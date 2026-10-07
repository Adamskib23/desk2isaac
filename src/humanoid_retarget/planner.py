from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.tracking import detections_by_object, select_active_object
from humanoid_retarget.types import Detection, ObjectSpec, Trajectory, Waypoint


DOWNWARD_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def _smooth_positions(positions: np.ndarray, window: int = 5) -> np.ndarray:
    if len(positions) < 3 or window <= 1:
        return positions
    window = min(window, len(positions))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return positions
    pad = window // 2
    padded = np.pad(positions, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(positions, dtype=np.float32)
    for dim in range(positions.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def _desk_to_sim_xyz(desk_xyz_m: np.ndarray, retarget_cfg: dict[str, Any]) -> np.ndarray:
    origin = np.asarray(retarget_cfg.get("sim_origin_m", [0.55, 0.0, 0.0]), dtype=np.float32)
    scale = float(retarget_cfg.get("scale_xy", 1.0))
    mapping = retarget_cfg.get("mapping", "desk_y_forward_desk_x_left")
    x, y, z = [float(v) for v in desk_xyz_m]
    desk_center = retarget_cfg.get("desk_center_m", [0.0, 0.0])
    x -= float(desk_center[0])
    y -= float(desk_center[1])
    if mapping == "identity":
        sim_xy = np.asarray([x, y], dtype=np.float32)
    elif mapping == "desk_y_forward_desk_x_left":
        sim_xy = np.asarray([y, -x], dtype=np.float32)
    else:
        matrix = np.asarray(retarget_cfg["xy_matrix"], dtype=np.float32).reshape(2, 2)
        sim_xy = matrix @ np.asarray([x, y], dtype=np.float32)
    return np.asarray([origin[0] + scale * sim_xy[0], origin[1] + scale * sim_xy[1], origin[2] + z], dtype=np.float32)


def _trace_for_object(detections: list[Detection], fps: float) -> list[dict[str, Any]]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.frame_idx)
    frames = np.asarray([d.frame_idx for d in detections], dtype=np.float32)
    positions = np.asarray([d.desk_xyz_m for d in detections], dtype=np.float32)
    confidences = np.asarray([d.confidence for d in detections], dtype=np.float32)
    all_frames = np.arange(int(frames.min()), int(frames.max()) + 1, dtype=np.float32)
    interp_pos = np.vstack([np.interp(all_frames, frames, positions[:, dim]) for dim in range(3)]).T
    interp_pos = _smooth_positions(interp_pos)
    interp_conf = np.interp(all_frames, frames, confidences)
    visible_frames = {d.frame_idx: d.visible for d in detections}
    return [
        {
            "frame_idx": int(frame),
            "t": float(frame / fps),
            "desk_xyz_m": [float(v) for v in xyz],
            "confidence": float(conf),
            "visible": visible_frames.get(int(frame), False),
        }
        for frame, xyz, conf in zip(all_frames, interp_pos, interp_conf)
    ]


def _movement_window(trace: list[dict[str, Any]], fps: float, threshold_m: float) -> tuple[int, int, dict[str, Any]]:
    positions = np.asarray([row["desk_xyz_m"] for row in trace], dtype=np.float32)
    frames = np.asarray([row["frame_idx"] for row in trace], dtype=np.int32)
    hold_frames = min(max(3, int(round(0.4 * fps))), len(positions))
    initial = np.median(positions[:hold_frames], axis=0)
    final = np.median(positions[-hold_frames:], axis=0)
    displacement = np.linalg.norm(positions[:, :2] - initial[:2], axis=1)
    moving = np.linalg.norm(positions - initial, axis=1) > threshold_m
    if np.any(moving):
        motion_index = int(np.argmax(moving))
        pick_index = max(0, motion_index - max(1, int(round(0.15 * fps))))
    else:
        xy_speed = np.zeros(len(positions), dtype=np.float32)
        if len(positions) > 1:
            xy_speed[1:] = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1) * fps
        motion_index = int(np.argmax(xy_speed))
        pick_index = motion_index
    pick_frame = int(frames[pick_index])
    settling_threshold = min(0.010, threshold_m * 0.4)
    settled = np.linalg.norm(positions - final, axis=1) <= settling_threshold
    place_frame = int(frames[-1])
    placement_settled = False
    for index in range(motion_index + 1, len(positions) - hold_frames + 1):
        if np.all(settled[index:index + hold_frames]):
            place_frame = int(frames[index])
            placement_settled = True
            break
    stats = {
        "initial_desk_xyz_m": [float(v) for v in initial],
        "final_desk_xyz_m": [float(v) for v in final],
        "max_displacement_m": float(np.max(displacement)),
        "path_length_m": float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()),
        "max_lift_m": float(max(0, positions[:, 2].max() - initial[2])),
        "placement_settled": placement_settled,
    }
    return pick_frame, place_frame, stats


def build_trajectory(
    detections: list[Detection],
    specs: list[ObjectSpec],
    calibration: WorkspaceCalibration,
    metadata: dict[str, Any],
    retarget_cfg: dict[str, Any],
    *,
    object_name: str | None = None,
) -> Trajectory:
    fps = float(metadata.get("fps", 30.0))
    spec_by_name = {spec.name: spec for spec in specs}
    declared_from_argument = object_name is not None
    requested_object = object_name if object_name is not None else metadata.get("action_object")
    if requested_object is not None and not isinstance(requested_object, str):
        raise ValueError("Episode action_object must be an object-name string")
    if requested_object is not None and requested_object not in spec_by_name:
        raise ValueError(f"Requested object {requested_object!r} is not configured")
    automatic_object, object_scores = select_active_object(
        detections,
        min_displacement_m=float(retarget_cfg.get("min_displacement_m", 0.035)),
    )
    if requested_object is None:
        object_name = automatic_object
    else:
        # A declared capture label must still have a detected, moving object.
        object_name, _ = select_active_object(
            [d for d in detections if d.object_name == requested_object],
            min_displacement_m=float(retarget_cfg.get("min_displacement_m", 0.035)),
        )
    spec = spec_by_name[object_name]
    object_detections = detections_by_object(detections)[object_name]
    trace = _trace_for_object(object_detections, fps)
    pick_frame, place_frame, movement_stats = _movement_window(
        trace,
        fps,
        threshold_m=float(retarget_cfg.get("movement_threshold_m", 0.025)),
    )

    for row in trace:
        sim_xyz = _desk_to_sim_xyz(np.asarray(row["desk_xyz_m"], dtype=np.float32), retarget_cfg)
        row["sim_xyz_m"] = [float(v) for v in sim_xyz]

    pick_row = min(trace, key=lambda row: abs(row["frame_idx"] - pick_frame))
    pick_xyz = _desk_to_sim_xyz(np.asarray(movement_stats["initial_desk_xyz_m"]), retarget_cfg)
    place_xyz = _desk_to_sim_xyz(np.asarray(movement_stats["final_desk_xyz_m"]), retarget_cfg)
    hover_height = float(retarget_cfg.get("hover_height_m", 0.24))
    hand_offset = float(retarget_cfg.get("hand_to_grasp_offset_m", 0.1034))
    table_z = float(retarget_cfg.get("sim_table_z_m", 0.0))
    hover_z = max(table_z + hover_height, float(np.max([row["sim_xyz_m"][2] for row in trace])) + 0.10)
    grasp_z = float(pick_xyz[2]) + hand_offset

    waypoints: list[Waypoint] = []

    def add(phase: str, t: float, ee_xyz: np.ndarray, gripper: str, object_xyz: np.ndarray) -> None:
        waypoints.append(
            Waypoint(
                t=round(float(t), 4),
                phase=phase,
                ee_xyz_m=tuple(float(v) for v in ee_xyz),
                ee_quat_wxyz=DOWNWARD_QUAT_WXYZ,
                gripper=gripper,
                object_xyz_m=tuple(float(v) for v in object_xyz),
            )
        )

    t = 0.0
    add("ready", t, np.asarray([pick_xyz[0], pick_xyz[1], hover_z]), "open", pick_xyz)
    t += 1.0
    add("descend_to_grasp", t, np.asarray([pick_xyz[0], pick_xyz[1], grasp_z]), "open", pick_xyz)
    t += 0.35
    add("close_gripper", t, np.asarray([pick_xyz[0], pick_xyz[1], grasp_z]), "closed", pick_xyz)
    carry_rows = [row for row in trace if pick_frame <= row["frame_idx"] <= place_frame]
    max_carry = max(2, int(retarget_cfg.get("max_carry_waypoints", 80) or 80))
    indices = np.unique(np.linspace(0, len(carry_rows) - 1, min(max_carry, len(carry_rows))).astype(int))
    carry_start = t + 0.15
    for index in indices:
        row = carry_rows[index]
        sim_xyz = np.asarray(row["sim_xyz_m"], dtype=np.float32)
        t = carry_start + float(row["t"] - pick_row["t"])
        phase = "lift" if index == 0 else "carry_observed_object_path"
        add(phase, t, sim_xyz + [0, 0, hand_offset], "closed", sim_xyz)

    t += 0.4
    add("descend_to_place", t, place_xyz + [0, 0, hand_offset], "closed", place_xyz)
    t += 0.25
    add("open_gripper", t, place_xyz + [0, 0, hand_offset], "open", place_xyz)
    t += 0.35
    add("retreat", t, np.asarray([place_xyz[0], place_xyz[1], hover_z]), "open", place_xyz)

    workspace = {
        "desk_limits_m": calibration.desk_limits_m,
        "table_z_m": calibration.table_z_m,
        "retarget": retarget_cfg,
        "ee_frame": "panda_hand",
        "hand_to_grasp_offset_m": hand_offset,
    }
    events = {
        "pick_frame": pick_frame,
        "place_frame": place_frame,
        "object_scores": object_scores,
        "selection_method": "declared_object" if requested_object is not None else "automatic_motion",
        "automatic_object_name": automatic_object,
        **movement_stats,
    }
    if requested_object is not None:
        events["declaration_source"] = "explicit_argument" if declared_from_argument else "episode_metadata"
    return Trajectory(
        schema_version="hmnd_rgbd_retarget_v1",
        source_episode=str(metadata.get("episode_dir", "")),
        object_name=object_name,
        fps=fps,
        events=events,
        object=asdict(spec),
        workspace=workspace,
        object_trace=trace,
        waypoints=[asdict(wp) for wp in waypoints],
        scene_objects=[
            {
                **asdict(spec_by_name[name]),
                "initial_sim_xyz_m": _desk_to_sim_xyz(
                    np.median([d.desk_xyz_m for d in rows[:12]], axis=0), retarget_cfg,
                ).tolist(),
            }
            for name, rows in detections_by_object(detections).items() if name in spec_by_name
        ],
    )
