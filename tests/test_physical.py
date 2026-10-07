from copy import deepcopy

import numpy as np
import pytest

from humanoid_retarget.physical import apply_visual_yaw, build_physical_plan, evaluate_physical_grasp
from humanoid_retarget.yaw_tracking import yaw_quaternion


@pytest.fixture
def trajectory():
    return {
        "object": {"size_m": [0.05555, 0.045, 0.031]},
        "workspace": {"retarget": {"sim_table_z_m": 1.05}, "hand_to_grasp_offset_m": 0.1034},
        "events": {"pick_frame": 10, "place_frame": 30},
        "waypoints": [
            {"t": 0, "phase": "ready", "object_xyz_m": [0.5, 0.1, 1.063], "ee_xyz_m": [0.5, 0.1, 1.31]},
            {"t": 5, "phase": "retreat", "object_xyz_m": [0.6, -0.1, 1.068]},
        ],
        "object_trace": [{"frame_idx": i, "t": i / 30, "sim_xyz_m": [0.5 + (i - 10) * 0.005, 0.1 - (i - 10) * 0.01,
                                                                 1.063 + (0.08 if i == 20 else 0)]}
                         for i in range(10, 31)],
    }


def test_plan_uses_measured_start_goal_and_guidance_without_mutating_input(trajectory):
    original = deepcopy(trajectory)
    plan = build_physical_plan(trajectory)
    assert trajectory == original
    task = plan["task_input"]
    assert task["start_xyz_m"] == pytest.approx([0.5, 0.1, 1.0655])
    assert task["goal_xyz_m"] == pytest.approx([0.6, -0.1, 1.0655])
    assert task["observed_start_xyz_m"] == original["waypoints"][0]["object_xyz_m"]
    assert len(task["guidance_trajectory"]) == 21
    carry = [wp for wp in plan["waypoints"] if wp["phase"] == "carry_guidance"]
    assert np.allclose(np.asarray([wp["object_xyz_m"][:2] for wp in carry]),
                       np.asarray([row["sim_xyz_m"][:2] for row in original["object_trace"]]))
    assert min(wp["object_xyz_m"][2] for wp in carry) >= 1.1255
    assert np.all(np.diff([wp["t"] for wp in plan["waypoints"]]) > 0)
    closed = next(wp for wp in plan["waypoints"] if wp["phase"] == "close_gripper")
    held = next(wp for wp in plan["waypoints"] if wp["phase"] == "grasp_hold")
    assert held["t"] - closed["t"] >= 0.7


def test_release_plan_holds_before_opening_and_has_no_fake_placement(trajectory):
    plan = build_physical_plan(trajectory, release_at_lift=True)
    phases = [wp["phase"] for wp in plan["waypoints"]]
    assert "descend_to_place" not in phases
    assert "release_probe" in phases
    probe = next(wp for wp in plan["waypoints"] if wp["phase"] == "release_probe")
    assert probe["gripper"] == "open"
    assert all(wp["gripper"] == "open" for wp in plan["waypoints"] if wp["t"] >= probe["t"])


@pytest.mark.parametrize("clearance", [0.01, float("nan")])
def test_plan_rejects_unsafe_clearance(trajectory, clearance):
    with pytest.raises(ValueError, match="clearance"):
        build_physical_plan(trajectory, clearance_m=clearance)


@pytest.mark.parametrize("failure", ["goal_nan", "bad_size", "bad_time"])
def test_plan_rejects_invalid_task_inputs(trajectory, failure):
    if failure == "goal_nan":
        trajectory["waypoints"][-1]["object_xyz_m"][0] = float("nan")
    elif failure == "bad_size":
        trajectory["object"]["size_m"][2] = -0.01
    else:
        trajectory["object_trace"][2]["t"] = trajectory["object_trace"][1]["t"]
    with pytest.raises(ValueError):
        build_physical_plan(trajectory)


@pytest.fixture
def physics_samples():
    rows = []
    for t in np.arange(0, 3.01, 0.05):
        held = 0.5 <= t < 2
        rows.append({"t": float(t), "phase": "carry_guidance" if held else "retreat",
                     "gripper_command": "closed" if held else "open",
                     "object_xyz_m": [0.6 if t >= 2 else 0.5, -0.1, 1.1255 if held else 1.0655],
                     "object_bottom_clearance_m": 0.06 if held else 0,
                     "finger_box_contact_force_n": [10, 10] if held else [0, 0],
                     "object_speed_m_s": 0.001, "object_angular_speed_rad_s": 0.001,
                     "object_in_hand_xyz_m": [0, 0, 0.1034]})
    return rows


def test_metrics_require_contact_lift_release_settling_and_measured_goal(physics_samples):
    result = evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True)
    assert result["physical_grasp_validated"]
    assert result["pick_place_passed"]
    assert result["bilateral_airborne_hold_s"] > 1
    assert result["final_goal_xy_error_m"] < 1e-8


@pytest.mark.parametrize("failure", ["no_lift", "no_contact", "one_finger", "wrong_goal", "moving", "slip", "not_released"])
def test_physical_failures_cannot_pass_based_on_arm_tracking(physics_samples, failure):
    goal = [0.6, -0.1, 1.0655]
    if failure == "wrong_goal":
        goal[0] += 0.1
    for row in physics_samples:
        if failure == "no_lift":
            row["object_bottom_clearance_m"] = 0
        elif failure == "no_contact":
            row["finger_box_contact_force_n"] = [0, 0]
        elif failure == "one_finger":
            row["finger_box_contact_force_n"][0] = 0
        elif failure == "moving":
            row["object_speed_m_s"] = 0.1
        elif failure == "slip" and row["t"] > 1:
            row["object_in_hand_xyz_m"][0] = 0.03
        elif failure == "not_released":
            row["gripper_command"] = "closed"
    result = evaluate_physical_grasp(physics_samples, goal, completed=True)
    assert not result["pick_place_passed"]


def test_incomplete_run_cannot_validate_grasp(physics_samples):
    result = evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=False)
    assert not result["physical_grasp_validated"]
    assert not result["pick_place_passed"]


def test_no_samples_and_non_finite_telemetry_are_rejected(physics_samples):
    with pytest.raises(ValueError, match="No physical samples"):
        evaluate_physical_grasp([], [0.6, -0.1, 1.0655], completed=True)
    physics_samples[10]["object_bottom_clearance_m"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True)


def test_out_of_order_samples_are_rejected(physics_samples):
    physics_samples[10]["t"] = physics_samples[9]["t"]
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True)


def test_release_probe_requires_actual_fall_and_does_not_claim_pick_place(physics_samples):
    for row in physics_samples:
        if 1.9 <= row["t"] < 2:
            row["phase"] = "release_probe"
            row["gripper_command"] = "open"
        elif row["t"] >= 2:
            row["phase"] = "observe_free_fall"
    result = evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True, release_at_lift=True)
    assert result["release_test_passed"]
    assert not result["pick_place_passed"]
    assert result["release_drop_m"] == pytest.approx(0.06)
    for row in physics_samples:
        if row["t"] >= 1.9:
            row["object_bottom_clearance_m"] = 0.06
    result = evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True, release_at_lift=True)
    assert not result["release_test_passed"]


def add_yaw(trajectory):
    trajectory["object"]["shape"] = "cuboid"
    trajectory["orientation"] = {"mode": "observed_yaw", "initial": {"passed": True}, "final": {"passed": True},
                                 "symmetry_period_rad": float(np.pi), "start_sim_yaw_rad": 0.2, "goal_sim_yaw_rad": 0.7,
                                 "control_knots": [{"frame_idx": 10, "sim_yaw_rad": 0.2},
                                                   {"frame_idx": 30, "sim_yaw_rad": 0.7}]}


def test_physical_plan_rotates_wrist_and_preserves_legacy_without_yaw(trajectory):
    legacy = build_physical_plan(trajectory)
    assert all(w["ee_quat_wxyz"] == [0, 1, 0, 0] for w in legacy["waypoints"])
    add_yaw(trajectory)
    original = deepcopy(trajectory)
    plan = build_physical_plan(trajectory)
    assert trajectory == original
    assert plan["waypoints"][0]["ee_quat_wxyz"] == pytest.approx(yaw_quaternion(0.2, downward=True))
    assert plan["waypoints"][-1]["ee_quat_wxyz"] == pytest.approx(yaw_quaternion(0.7, downward=True))
    waypoints = plan["waypoints"]
    rates = np.diff([w["ee_yaw_rad"] for w in waypoints]) / np.diff([w["t"] for w in waypoints])
    assert np.max(np.abs(rates)) <= 0.600001


def test_bad_yaw_or_unsupported_boundary_cannot_drive_robot(trajectory):
    add_yaw(trajectory)
    trajectory["orientation"]["initial"]["passed"] = False
    with pytest.raises(ValueError, match="supported cuboid"):
        build_physical_plan(trajectory)
    trajectory["orientation"]["initial"]["passed"] = True
    trajectory["orientation"]["goal_sim_yaw_rad"] = float("nan")
    with pytest.raises(ValueError, match="Yaw guidance"):
        build_physical_plan(trajectory)


@pytest.mark.parametrize("actual,passes", [(0.4, True), (0.4 + np.pi, True), (0.8, False)])
def test_yaw_metrics_use_actual_object_orientation_not_arm_tracking(physics_samples, actual, passes):
    for row in physics_samples:
        row["object_quat_wxyz"] = yaw_quaternion(actual)
    result = evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True, goal_yaw_rad=0.4)
    assert result["pick_place_passed"]
    assert result["pick_place_with_yaw_passed"] == passes


def test_upside_down_object_cannot_pass_upright_yaw_check(physics_samples):
    for row in physics_samples:
        row["object_quat_wxyz"] = yaw_quaternion(0.4, downward=True)
    result = evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True, goal_yaw_rad=0.4)
    assert not result["pick_place_with_yaw_passed"]
    assert result["final_upright_tilt_deg"] == pytest.approx(180)


def test_legacy_visual_mode_can_show_yaw_without_changing_xyz(trajectory):
    add_yaw(trajectory)
    trajectory["fps"] = 30
    trajectory["waypoints"].insert(1, {"t": 1.5, "phase": "lift", "object_xyz_m": [0.5, 0.1, 1.15]})
    trajectory["waypoints"].insert(2, {"t": 1.5 + 20 / 30, "phase": "carry_observed_object_path", "object_xyz_m": [0.6, -0.1, 1.15]})
    original = deepcopy(trajectory)
    result = apply_visual_yaw(trajectory)
    assert trajectory == original
    assert result["waypoints"][0]["ee_yaw_rad"] == pytest.approx(0.2)
    assert result["waypoints"][2]["ee_yaw_rad"] == pytest.approx(0.7)
    assert result["waypoints"][-1]["ee_yaw_rad"] == pytest.approx(0.7)
    assert [w["object_xyz_m"] for w in result["waypoints"]] == [w["object_xyz_m"] for w in original["waypoints"]]


def test_carry_yaw_error_is_based_on_actual_object_and_command(physics_samples):
    for row in physics_samples:
        row["object_quat_wxyz"] = yaw_quaternion(0.4)
        row["target_object_yaw_rad"] = 0.6
    result = evaluate_physical_grasp(physics_samples, [0.6, -0.1, 1.0655], completed=True, goal_yaw_rad=0.4)
    assert result["carry_yaw_error_p95_deg"] == pytest.approx(np.rad2deg(0.2))
