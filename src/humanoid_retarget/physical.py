from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from humanoid_retarget.planner import DOWNWARD_QUAT_WXYZ
from humanoid_retarget.yaw_tracking import axis_difference, quaternion_yaw_tilt, yaw_quaternion


def apply_visual_yaw(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Augment the legacy visual replay only; physical plans have their own timing."""
    result = deepcopy(trajectory)
    orientation = result.get("orientation", {})
    if orientation.get("mode") != "observed_yaw":
        return result
    knots = orientation["control_knots"]
    frames = [k["frame_idx"] for k in knots]
    angles = [k["sim_yaw_rad"] for k in knots]
    lift = next(w["t"] for w in result["waypoints"] if w["phase"] == "lift")
    for waypoint in result["waypoints"]:
        phase = waypoint["phase"]
        if phase in ("lift", "carry_observed_object_path"):
            frame = result["events"]["pick_frame"] + (waypoint["t"] - lift) * result["fps"]
            angle = float(np.interp(frame, frames, angles))
        elif phase in ("descend_to_place", "open_gripper", "retreat"):
            angle = orientation["goal_sim_yaw_rad"]
        else:
            angle = orientation["start_sim_yaw_rad"]
        waypoint.update(ee_yaw_rad=angle, object_yaw_rad=angle, ee_quat_wxyz=yaw_quaternion(angle, downward=True))
    return result


def build_physical_plan(
    trajectory: dict[str, Any], *, clearance_m: float = 0.06, release_at_lift: bool = False,
) -> dict[str, Any]:
    """Convert measured start/goal/path into robot-only approach, grasp and place commands."""
    if not np.isfinite(clearance_m) or clearance_m < 0.03:
        raise ValueError("Carry clearance must be finite and at least 30 mm")
    source = trajectory["waypoints"]
    table_z = float(trajectory["workspace"]["retarget"].get("sim_table_z_m", 1.05))
    offset = float(trajectory["workspace"].get("hand_to_grasp_offset_m", 0.1034))
    size = np.asarray(trajectory["object"]["size_m"], dtype=float)
    observed_start = np.asarray(source[0]["object_xyz_m"], dtype=float)
    observed_goal = np.asarray(source[-1]["object_xyz_m"], dtype=float)
    if (size.shape != (3,) or observed_start.shape != (3,) or observed_goal.shape != (3,)
            or not all(np.all(np.isfinite(v)) for v in (size, observed_start, observed_goal, [table_z, offset]))
            or np.any(size <= 0) or offset <= 0):
        raise ValueError("Task start/goal, table height, grasp offset, and object dimensions must be finite and valid")
    start, goal = observed_start.copy(), observed_goal.copy()
    # The demonstration starts and ends on the desk; do not spawn inside it due to depth noise.
    start[2] = goal[2] = table_z + size[2] / 2
    events = trajectory.get("events", {})
    guidance = [row for row in trajectory.get("object_trace", [])
                if events.get("pick_frame", -1) <= row["frame_idx"] <= events.get("place_frame", float("inf"))]
    if guidance:
        guidance = [guidance[i] for i in np.unique(np.linspace(0, len(guidance) - 1, min(80, len(guidance))).astype(int))]
        path = np.asarray([row["sim_xyz_m"] for row in guidance], dtype=float)
        path_times = np.asarray([row["t"] for row in guidance], dtype=float)
    else:
        guidance = [wp for wp in source if wp["phase"] in ("lift", "carry_observed_object_path")]
        path = np.asarray([wp["object_xyz_m"] for wp in guidance], dtype=float)
        path_times = np.asarray([wp["t"] for wp in guidance], dtype=float)
    if len(path) < 2 or not np.all(np.isfinite(path)) or not np.all(np.diff(path_times) > 0):
        raise ValueError("A physical task needs at least two finite, time-ordered guidance positions")
    path_times -= path_times[0]
    orientation = trajectory.get("orientation", {})
    yaw_enabled = orientation.get("mode") == "observed_yaw"
    start_yaw = goal_yaw = 0.0
    path_yaw = np.zeros(len(path))
    if yaw_enabled:
        if (trajectory["object"].get("shape") != "cuboid"
                or not orientation.get("initial", {}).get("passed") or not orientation.get("final", {}).get("passed")
                or not np.isclose(orientation.get("symmetry_period_rad", 0), np.pi)):
            raise ValueError("Observed yaw requires supported cuboid boundaries and 180-degree symmetry")
        start_yaw, goal_yaw = float(orientation["start_sim_yaw_rad"]), float(orientation["goal_sim_yaw_rad"])
        knots = orientation["control_knots"]
        knot_frames = np.asarray([r["frame_idx"] for r in knots], dtype=float)
        knot_angles = np.asarray([r["sim_yaw_rad"] for r in knots], dtype=float)
        if (len(knots) < 2 or not np.isfinite([start_yaw, goal_yaw]).all()
                or not np.isfinite(knot_frames).all() or not np.isfinite(knot_angles).all()
                or not np.all(np.diff(knot_frames) > 0) or any(r.get("frame_idx") is None for r in guidance)):
            raise ValueError("Yaw guidance needs finite, frame-indexed, increasing control knots")
        path_yaw = np.interp([r["frame_idx"] for r in guidance], knot_frames, knot_angles)
    safe_path = path.copy()
    safe_path[:, 2] = np.maximum(path[:, 2] + start[2] - observed_start[2], start[2] + clearance_m)
    hover_z = max(float(source[0]["ee_xyz_m"][2]), float(safe_path[:, 2].max()) + offset + 0.06)
    waypoints = []

    def add(t: float, phase: str, centre: np.ndarray, gripper: str, *, hover: bool = False, yaw: float = start_yaw) -> None:
        hand = centre + [0, 0, offset]
        if hover:
            hand[2] = hover_z
        if yaw_enabled and waypoints:
            # Slow the robot command when yaw needs more time; this never relabels the human capture.
            t = max(t, waypoints[-1]["t"] + abs(yaw - waypoints[-1]["ee_yaw_rad"]) / 0.6)
        waypoint = {"t": float(t), "phase": phase, "ee_xyz_m": hand.tolist(),
                    "ee_quat_wxyz": yaw_quaternion(yaw, downward=True) if yaw_enabled else list(DOWNWARD_QUAT_WXYZ),
                    "gripper": gripper, "object_xyz_m": centre.tolist()}
        if yaw_enabled:
            waypoint.update(ee_yaw_rad=float(yaw), object_yaw_rad=float(yaw))
        waypoints.append(waypoint)

    add(0, "approach", start, "open", hover=True)
    add(1.2, "descend_to_grasp", start, "open")
    add(1.21, "close_gripper", start, "closed")
    add(2.0, "grasp_hold", start, "closed")
    add(2.8, "lift", start + [0, 0, clearance_m], "closed")
    peak = int(np.argmax(safe_path[:, 2]))
    delay = 0.0
    for i, (position, observed_t) in enumerate(zip(safe_path, path_times)):
        t = 3.1 + float(observed_t) + delay
        add(t, "carry_guidance", position, "closed", yaw=float(path_yaw[i]))
        delay += waypoints[-1]["t"] - t
        t = waypoints[-1]["t"]
        if i == peak:
            add(t + 0.7, "airborne_hold", position, "closed", yaw=float(path_yaw[i]))
            delay += 0.7
            if release_at_lift:
                add(t + 0.71, "release_probe", position, "open", yaw=float(path_yaw[i]))
                add(t + 2.1, "observe_free_fall", position, "open", yaw=float(path_yaw[i]))
                add(t + 2.7, "retreat", position, "open", hover=True, yaw=float(path_yaw[i]))
                break
    if not release_at_lift:
        t = waypoints[-1]["t"]
        add(t + 0.5, "place_hover", goal + [0, 0, clearance_m], "closed", yaw=goal_yaw)
        t = waypoints[-1]["t"] - 0.5
        add(t + 1.3, "descend_to_place", goal, "closed", yaw=goal_yaw)
        add(t + 1.6, "place_hold", goal, "closed", yaw=goal_yaw)
        add(t + 1.61, "open_gripper", goal, "open", yaw=goal_yaw)
        add(t + 2.1, "release_hold", goal, "open", yaw=goal_yaw)
        add(t + 2.8, "retreat", goal, "open", hover=True, yaw=goal_yaw)
    plan = deepcopy(trajectory)
    plan["waypoints"] = waypoints
    plan["task_input"] = {
        "observed_start_xyz_m": observed_start.tolist(), "observed_goal_xyz_m": observed_goal.tolist(),
        "start_xyz_m": start.tolist(), "goal_xyz_m": goal.tolist(),
        "guidance_trajectory": [{"t": float(t), "xyz_m": p.tolist(), "source_frame_idx": row.get("frame_idx")}
                                for t, p, row in zip(path_times, path, guidance)],
        "carry_clearance_m": clearance_m,
        "orientation_source": "sam2_rgbd_upright_shape_axis_yaw" if yaw_enabled else "fixed_downward_not_observed",
        "start_goal_height_source": "resting_on_measured_table", "release_at_lift": release_at_lift,
    }
    if yaw_enabled:
        plan["task_input"].update(start_yaw_rad=start_yaw, goal_yaw_rad=goal_yaw, yaw_symmetry_period_rad=float(np.pi),
                                  yaw_command_max_speed_rad_s=0.6,
                                  yaw_interpolation_policy="Robot command only; unsupported frames are not teacher labels")
        for row, yaw in zip(plan["task_input"]["guidance_trajectory"], path_yaw):
            row["control_yaw_rad"] = float(yaw)
    return plan


def _longest_duration(times: np.ndarray, mask: np.ndarray) -> float:
    best, start = 0.0, None
    for t, active in zip(times, mask):
        if active:
            if start is None:
                start = t
            best = max(best, t - start)
        else:
            start = None
    return float(best)


def evaluate_physical_grasp(
    samples: list[dict[str, Any]], goal_xyz_m: list[float], *, completed: bool, release_at_lift: bool = False,
    goal_yaw_rad: float | None = None,
) -> dict[str, Any]:
    """Evaluate measured physics states and object-filtered finger contact, never commanded box poses."""
    if not samples:
        raise ValueError("No physical samples to evaluate")
    times = np.asarray([row["t"] for row in samples])
    xyz = np.asarray([row["object_xyz_m"] for row in samples])
    clearance = np.asarray([row["object_bottom_clearance_m"] for row in samples])
    forces = np.asarray([row["finger_box_contact_force_n"] for row in samples])
    speed = np.asarray([row["object_speed_m_s"] for row in samples])
    angular_speed = np.asarray([row["object_angular_speed_rad_s"] for row in samples])
    relative = np.asarray([row["object_in_hand_xyz_m"] for row in samples])
    if not all(np.all(np.isfinite(a)) for a in (times, xyz, clearance, forces, speed, angular_speed, relative)):
        raise ValueError("Non-finite physical telemetry")
    if not np.all(np.diff(times) > 0):
        raise ValueError("Physical sample times must be strictly increasing")
    airborne = clearance >= 0.03
    bilateral = np.all(forces >= 0.5, axis=1)
    held = airborne & bilateral
    held_duration = _longest_duration(times, held)
    lifted = held_duration >= 0.4
    tail = times >= times[-1] - 0.5
    final_error = float(np.linalg.norm(np.median(xyz[tail, :2], axis=0) - np.asarray(goal_xyz_m)[:2]))
    on_table = bool(np.all(np.abs(clearance[tail]) < 0.005))
    settled = bool(np.all(speed[tail] < 0.03) and np.all(angular_speed[tail] < 0.3))
    released = bool(np.all(forces[tail] < 0.5)
                    and all(row["gripper_command"] == "open" for row, active in zip(samples, tail) if active))
    slip = None
    if np.any(held):
        first = int(np.argmax(held))
        reference_mask = held & (times <= times[first] + 0.2)
        reference = np.median(relative[reference_mask], axis=0)
        slip = float(np.linalg.norm(relative[held] - reference, axis=1).max())
    result = {
        "max_bottom_clearance_m": float(clearance.max()), "bilateral_airborne_hold_s": held_duration,
        "finger_box_contact_peak_n": forces.max(axis=0).tolist(), "grasp_slip_max_m": slip,
        "lift_passed": bool(lifted), "final_goal_xy_error_m": final_error,
        "on_table": on_table, "settled": settled, "released": released,
        "physical_grasp_validated": bool(completed and lifted),
        "pick_place_passed": bool(completed and lifted and final_error <= 0.025 and on_table and settled and released
                                  and slip is not None and slip <= 0.02),
    }
    if release_at_lift:
        probe = np.asarray([row["phase"] == "release_probe" for row in samples])
        release_index = int(np.argmax(probe)) if np.any(probe) else None
        drop = float(clearance[release_index] - np.median(clearance[tail])) if release_index is not None else 0.0
        result["release_drop_m"] = drop
        result["release_test_passed"] = bool(completed and lifted and drop >= 0.04 and released and on_table and settled)
        result["pick_place_passed"] = False
    if goal_yaw_rad is not None:
        if not np.isfinite(goal_yaw_rad):
            raise ValueError("Goal yaw must be finite")
        yaw, tilt = quaternion_yaw_tilt([row["object_quat_wxyz"] for row in samples])
        error = np.abs(axis_difference(yaw[tail], goal_yaw_rad))
        yaw_error = float(np.percentile(error, 95))
        tilt_error = float(np.percentile(tilt[tail], 95))
        passed = bool(completed and yaw_error <= np.deg2rad(10) and tilt_error <= np.deg2rad(15))
        result.update(final_goal_yaw_error_deg=float(np.rad2deg(yaw_error)),
                      final_upright_tilt_deg=float(np.rad2deg(tilt_error)),
                      yaw_tolerance_deg=10.0, upright_tolerance_deg=15.0,
                      yaw_goal_passed=passed, pick_place_with_yaw_passed=bool(result["pick_place_passed"] and passed))
        carry = np.asarray([row["phase"] in ("carry_guidance", "airborne_hold")
                            and row.get("target_object_yaw_rad") is not None for row in samples])
        if np.any(carry):
            targets = np.asarray([row.get("target_object_yaw_rad", 0) for row in samples], dtype=float)
            tracking_error = np.abs(axis_difference(yaw[carry], targets[carry]))
            result["carry_yaw_error_p95_deg"] = float(np.rad2deg(np.percentile(tracking_error, 95)))
    return result
