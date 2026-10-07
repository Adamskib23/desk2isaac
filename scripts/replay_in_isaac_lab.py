#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from humanoid_retarget.io import read_json, write_json
from humanoid_retarget.dataset_review import comparison_frame, reference_frame, reference_knots
from humanoid_retarget.physical import apply_visual_yaw, build_physical_plan, evaluate_physical_grasp
from humanoid_retarget.scene_context import load_scene_context, marker_visual_parts
from humanoid_retarget.yaw_tracking import yaw_quaternion
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay recorded object motion with a Franka in Isaac Lab.")
parser.add_argument("--trajectory", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", choices=("kinematic", "dynamic"), default="kinematic")
parser.add_argument("--object-mass-kg", type=float, default=0.1, help="Assumed box mass, not measured by the camera.")
parser.add_argument("--friction", type=float, default=0.8, help="Assumed contact friction coefficient.")
parser.add_argument("--grip-force-n", type=float, default=40.0, help="Per-finger actuator effort limit.")
parser.add_argument("--carry-clearance-m", type=float, default=0.06)
parser.add_argument("--release-at-lift", action="store_true", help="Dynamic diagnostic: open fingers during an airborne hold.")
parser.add_argument("--replay-speed", type=float, default=1.0)
parser.add_argument("--hold-seconds", type=float, default=1.0)
parser.add_argument("--settle-seconds", type=float, default=2.0)
parser.add_argument("--video", type=Path, help="Save RGB replay video and start/lift/end images.")
parser.add_argument("--reference-episode", type=Path, help="Show/save recorded RGB alongside the simulated camera")
parser.add_argument("--scene-context", type=Path, help="Fingerprint-checked calibrated markers and visual-only background objects")
parser.add_argument("--review-title", default="", help="Episode index/name for the comparison panel")
parser.add_argument("--report", type=Path, help="Save controller and, in dynamic mode, physical-grasp metrics.")
parser.add_argument("--real-time", action="store_true", help="Pace headless playback; GUI playback is paced automatically.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.replay_speed <= 0 or min(args_cli.hold_seconds, args_cli.settle_seconds) < 0:
    parser.error("Environment count and speed must be positive; hold/settle durations non-negative")
if not all(math.isfinite(v) and v > 0 for v in (args_cli.object_mass_kg, args_cli.friction, args_cli.grip_force_n, args_cli.replay_speed)):
    parser.error("Mass, friction, and grip force must be positive")
if args_cli.release_at_lift and args_cli.mode != "dynamic":
    parser.error("--release-at-lift requires --mode dynamic")
if args_cli.video or args_cli.reference_episode:
    args_cli.enable_cameras = True
TRAJECTORY = read_json(args_cli.trajectory)
SCENE_CONTEXT = load_scene_context(args_cli.scene_context, Path(args_cli.trajectory)) if args_cli.scene_context else None
if not TRAJECTORY.get("waypoints"):
    parser.error("Trajectory contains no waypoints")
if args_cli.mode == "dynamic":
    TRAJECTORY = build_physical_plan(TRAJECTORY, clearance_m=args_cli.carry_clearance_m,
                                     release_at_lift=args_cli.release_at_lift)
else:
    TRAJECTORY = apply_visual_yaw(TRAJECTORY)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

TABLE_Z = float(TRAJECTORY["workspace"]["retarget"].get("sim_table_z_m", 1.05))
OBJ_SIZE = tuple(float(v) for v in TRAJECTORY["object"]["size_m"])
FIRST_OBJ = tuple(float(v) for v in TRAJECTORY["waypoints"][0]["object_xyz_m"])
INITIAL_YAW = float(TRAJECTORY.get("task_input", {}).get("start_yaw_rad",
                    TRAJECTORY.get("orientation", {}).get("start_sim_yaw_rad", 0)))
CAMERA_EYE = [1.28, 1.10, TABLE_Z + 0.90]
CAMERA_TARGET = [0.32, 0.0, TABLE_Z + 0.20]


def object_spawn_cfg(spec, *, dynamic=False, visual_only=False):
    size = tuple(float(v) for v in spec["size_m"])
    bgr = spec.get("color_bgr", [40, 40, 220])
    properties = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=not dynamic, disable_gravity=not dynamic,
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.object_mass_kg if dynamic else 0.25),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args_cli.friction, dynamic_friction=args_cli.friction,
            restitution=0.0, friction_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple((c / 255) ** 2.2 for c in bgr[::-1]), roughness=0.7),
    )
    shape = spec.get("shape", "cuboid")
    if visual_only:
        for key in ("rigid_props", "mass_props", "collision_props", "physics_material"):
            properties.pop(key)
    if shape == "sphere":
        return sim_utils.SphereCfg(radius=size[0] / 2, **properties)
    if shape == "cylinder":
        return sim_utils.CylinderCfg(radius=size[0] / 2, height=size[2], **properties)
    if shape != "cuboid":
        raise ValueError(f"Unsupported simulation object shape: {shape}")
    return sim_utils.CuboidCfg(size=size, **properties)


@configclass
class RetargetReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2800, color=(0.9, 0.9, 0.9)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.25, 0.9, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.21, 0.23), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.48, 0.0, TABLE_Z - 0.025)),
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (-0.02, 0.0, TABLE_Z)
    if args_cli.mode == "dynamic":
        robot.spawn.activate_contact_sensors = True
        robot.spawn.articulation_props.solver_position_iteration_count = 16
        robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        robot.actuators["panda_hand"].effort_limit_sim = args_cli.grip_force_n
        left_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger", update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/ObservedObject"],
        )
        right_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/ObservedObject"],
        )
    observed_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObservedObject", spawn=object_spawn_cfg(TRAJECTORY["object"], dynamic=args_cli.mode == "dynamic"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FIRST_OBJ, rot=tuple(yaw_quaternion(INITIAL_YAW))),
    )
    if args_cli.video or args_cli.reference_episode:
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ReplayCamera", width=960, height=720, data_types=["rgb"],
            update_period=1 / 30,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=28.0, clipping_range=(0.05, 20)),
        )


def interp_vector(t, times, values):
    return np.asarray([np.interp(t, times, values[:, dim]) for dim in range(values.shape[1])], dtype=np.float32)


def run_simulator(sim, scene):
    sim_dt = sim.get_physics_dt()
    robot, observed_object = scene["robot"], scene["observed_object"]
    ee_body_idx = robot.body_names.index("panda_hand")
    ee_jacobian_idx = ee_body_idx - 1 if robot.is_fixed_base else ee_body_idx
    arm_ids = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    finger_ids = [i for i, name in enumerate(robot.joint_names) if name.startswith("panda_finger_joint")]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.reset()
    previous_joint_target = robot.data.default_joint_pos[:, arm_ids].clone()
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=scene.num_envs, device=sim.device,
    )
    waypoints = TRAJECTORY["waypoints"]
    times = np.asarray([wp["t"] for wp in waypoints], dtype=float)
    if not np.all(np.diff(times) > 0):
        raise ValueError("Waypoint times must be strictly increasing")
    ee_values = np.asarray([wp["ee_xyz_m"] for wp in waypoints])
    object_values = np.asarray([wp["object_xyz_m"] for wp in waypoints])
    yaw_values = np.asarray([wp["ee_yaw_rad"] for wp in waypoints]) if all("ee_yaw_rad" in wp for wp in waypoints) else None
    duration = float(times[-1]) / args_cli.replay_speed + args_cli.hold_seconds
    camera = scene["camera"] if args_cli.video or args_cli.reference_episode else None
    reference_paths, capture_times, knots = [], [], None
    image_provider, review_window = None, None
    if args_cli.reference_episode:
        from humanoid_retarget.io import list_rgb_frames, read_episode_metadata
        reference_paths = list_rgb_frames(args_cli.reference_episode)
        capture_times = read_episode_metadata(args_cli.reference_episode)["timestamps_relative_s"]
        if len(reference_paths) != len(capture_times):
            raise ValueError("Reference RGB/timestamps mismatch")
        if Path(TRAJECTORY["source_episode"]).resolve() != args_cli.reference_episode.resolve():
            raise ValueError("Reference episode does not match the trajectory")
        knots = reference_knots(TRAJECTORY, capture_times)
        if not args_cli.headless:
            import omni.ui as ui
            image_provider = ui.ByteImageProvider()
            review_window = ui.Window("Dataset Review " + args_cli.review_title, width=1280, height=520)
            with review_window.frame:
                ui.ImageWithProvider(image_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
    writer = None
    temporary_video = None
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([CAMERA_EYE], device=sim.device).repeat(scene.num_envs, 1) + scene.env_origins,
            torch.tensor([CAMERA_TARGET], device=sim.device).repeat(scene.num_envs, 1) + scene.env_origins,
        )
    if args_cli.video:
        args_cli.video.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args_cli.video.parent, suffix=".mp4", delete=False) as stream:
            temporary_video = Path(stream.name)
        writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                 (1920 if reference_paths else 960, 720))
        if not writer.isOpened():
            raise OSError(f"Could not open video writer: {args_cli.video}")
    steps = []
    snapshots = {}
    last_image = None
    lift_time = float(times[np.argmax(object_values[:, 2])]) / args_cli.replay_speed
    sim_time = -args_cli.settle_seconds
    count = 0
    wall_start = time.perf_counter()
    object_pose_commands = 0
    closed_width = 0.0 if args_cli.mode == "dynamic" else min(0.04, min(OBJ_SIZE[:2]) / 2)
    print(f"[INFO] Replaying {TRAJECTORY['object_name']}: {len(waypoints)} waypoints, {duration:.2f}s", flush=True)
    print(f"[INFO] Hand body {ee_body_idx}, Jacobian {ee_jacobian_idx}, fixed base: {robot.is_fixed_base}", flush=True)
    print(f"[INFO] Object mode: {args_cli.mode}; dynamic objects receive no replay pose commands", flush=True)
    try:
        while simulation_app.is_running() and sim_time <= duration:
            command_time = float(np.clip(max(0, sim_time) * args_cli.replay_speed, times[0], times[-1]))
            index = int(np.clip(np.searchsorted(times, command_time, side="right") - 1, 0, len(times) - 1))
            target = torch.tensor(interp_vector(command_time, times, ee_values), device=sim.device).repeat(scene.num_envs, 1)
            target += scene.env_origins
            command_yaw = float(np.interp(command_time, times, yaw_values)) if yaw_values is not None else None
            quaternion = yaw_quaternion(command_yaw, downward=True) if command_yaw is not None else waypoints[index]["ee_quat_wxyz"]
            target_quat = torch.tensor(quaternion, device=sim.device).repeat(scene.num_envs, 1)
            root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
            ee_pos, ee_quat = robot.data.body_pos_w[:, ee_body_idx], robot.data.body_quat_w[:, ee_body_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pos, ee_quat)
            target_b, quat_b = subtract_frame_transforms(root_pos, root_quat, target, target_quat)
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobian_idx, :, arm_ids].clone()
            world_to_base = matrix_from_quat(root_quat).transpose(1, 2)
            jacobian[:, :3] = world_to_base @ jacobian[:, :3]
            jacobian[:, 3:] = world_to_base @ jacobian[:, 3:]
            joint_pos = robot.data.joint_pos[:, arm_ids]
            controller.set_command(torch.cat((target_b, quat_b), dim=1))
            desired = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            limits = robot.data.soft_joint_pos_limits[:, arm_ids]
            # Bound command velocity, not its offset from the current joint state:
            # the latter would also cap servo torque and make the arm sluggish.
            desired = previous_joint_target + torch.clamp(desired - previous_joint_target, -2.0 * sim_dt, 2.0 * sim_dt)
            desired = torch.clamp(desired, limits[..., 0], limits[..., 1])
            if not torch.isfinite(desired).all():
                raise RuntimeError("IK generated non-finite joint commands")
            robot.set_joint_position_target(desired, joint_ids=arm_ids)
            previous_joint_target = desired.clone()
            if finger_ids:
                width = 0.04 if waypoints[index]["gripper"] == "open" else closed_width
                robot.set_joint_position_target(torch.full((scene.num_envs, len(finger_ids)), width, device=sim.device), joint_ids=finger_ids)
            if args_cli.mode == "kinematic":
                object_pose = observed_object.data.default_root_state[:, :7].clone()
                object_pose[:, :3] = torch.tensor(interp_vector(command_time, times, object_values), device=sim.device).repeat(scene.num_envs, 1)
                object_pose[:, :3] += scene.env_origins
                if command_yaw is not None:
                    object_pose[:, 3:7] = torch.tensor(yaw_quaternion(command_yaw), device=sim.device)
                observed_object.write_root_pose_to_sim(object_pose)
                object_pose_commands += 1
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            if sim_time >= 0:
                actual = robot.data.body_pos_w[:, ee_body_idx]
                error = torch.linalg.vector_norm(actual - target, dim=1)
                angle = 2 * torch.acos(torch.clamp(torch.abs((robot.data.body_quat_w[:, ee_body_idx] * target_quat).sum(-1)), max=1))
                sample = {
                    "t": round(sim_time, 5), "phase": waypoints[index]["phase"],
                    "target_ee_xyz_m": target[0].cpu().tolist(), "actual_ee_xyz_m": actual[0].cpu().tolist(),
                    "position_error_m": float(error.max()), "orientation_error_rad": float(angle.max()),
                }
                if args_cli.mode == "dynamic":
                    object_pos = observed_object.data.root_pos_w
                    object_rotation = matrix_from_quat(observed_object.data.root_quat_w)
                    if TRAJECTORY["object"]["shape"] == "sphere":
                        extent = torch.full((scene.num_envs,), OBJ_SIZE[0] / 2, device=sim.device)
                    elif TRAJECTORY["object"]["shape"] == "cylinder":
                        extent = OBJ_SIZE[0] / 2 * torch.linalg.vector_norm(object_rotation[:, 2, :2], dim=1) + OBJ_SIZE[2] / 2 * object_rotation[:, 2, 2].abs()
                    else:
                        extent = object_rotation[:, 2].abs() @ torch.tensor(OBJ_SIZE, device=sim.device) / 2
                    clearance = object_pos[:, 2] - scene.env_origins[:, 2] - extent - TABLE_Z
                    hand_pos, hand_quat = robot.data.body_pos_w[:, ee_body_idx], robot.data.body_quat_w[:, ee_body_idx]
                    relative_pos, _ = subtract_frame_transforms(hand_pos, hand_quat, object_pos)
                    contact = torch.stack([
                        torch.linalg.vector_norm(scene[name].data.force_matrix_w.reshape(scene.num_envs, -1, 3).sum(1), dim=1)
                        for name in ("left_finger_contact", "right_finger_contact")
                    ], dim=1)
                    sample["environments"] = [{
                        "t": sample["t"], "phase": sample["phase"], "gripper_command": waypoints[index]["gripper"],
                        "object_xyz_m": (object_pos[env] - scene.env_origins[env]).cpu().tolist(),
                        "object_quat_wxyz": observed_object.data.root_quat_w[env].cpu().tolist(),
                        "object_bottom_clearance_m": float(clearance[env]),
                        "object_speed_m_s": float(torch.linalg.vector_norm(observed_object.data.root_lin_vel_w[env])),
                        "object_angular_speed_rad_s": float(torch.linalg.vector_norm(observed_object.data.root_ang_vel_w[env])),
                        "object_in_hand_xyz_m": relative_pos[env].cpu().tolist(),
                        "finger_box_contact_force_n": contact[env].cpu().tolist(),
                        "finger_positions_m": robot.data.joint_pos[env, finger_ids].cpu().tolist(),
                    } for env in range(scene.num_envs)]
                    if command_yaw is not None:
                        for state in sample["environments"]:
                            state["target_object_yaw_rad"] = command_yaw
                steps.append(sample)
                if camera is not None and count % 4 == 0:
                    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                    last_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(last_image, (0, 0), (959, 75 if args_cli.mode == "dynamic" else 47), (25, 25, 25), -1)
                    mode_label = "DYNAMIC CONTACT" if args_cli.mode == "dynamic" else "KINEMATIC REPLAY"
                    label = f"{mode_label} | {waypoints[index]['phase'].replace('_', ' ')} | IK {error[0].item() * 1000:.1f} mm"
                    cv2.putText(last_image, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    if args_cli.mode == "dynamic":
                        state = sample["environments"][0]
                        left, right = state["finger_box_contact_force_n"]
                        detail = f"Object clearance {state['object_bottom_clearance_m'] * 1000:.0f} mm | finger contact L/R {left:.1f}/{right:.1f} N"
                        cv2.putText(last_image, detail, (20, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 230, 190), 1, cv2.LINE_AA)
                    if reference_paths:
                        source_index = reference_frame(command_time, knots, capture_times)
                        source_rgb = cv2.imread(str(reference_paths[source_index]))
                        if source_rgb is None:
                            raise ValueError(f"Cannot decode reference RGB frame {source_index}")
                        last_image = comparison_frame(source_rgb, last_image, title=args_cli.review_title,
                                                      capture_t=capture_times[source_index], sim_t=sim_time)
                        if image_provider is not None:
                            rgba = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGBA)
                            image_provider.set_data_array(rgba.reshape(-1), [rgba.shape[1], rgba.shape[0]])
                    if writer is not None:
                        writer.write(last_image)
                    if "start" not in snapshots:
                        snapshots["start"] = last_image.copy()
                    if "lift" not in snapshots and sim_time >= lift_time:
                        snapshots["lift"] = last_image.copy()
            count += 1
            sim_time += sim_dt
            if args_cli.real_time or not args_cli.headless:
                time.sleep(max(0, wall_start + count * sim_dt - time.perf_counter()))
    finally:
        if writer is not None:
            writer.release()
            if sim_time > duration and last_image is not None:
                temporary_video.replace(args_cli.video)
                snapshots["end"] = last_image
                for name, frame in snapshots.items():
                    cv2.imwrite(str(args_cli.video.with_name(args_cli.video.stem + "_" + name + ".png")), frame)
            else:
                temporary_video.unlink(missing_ok=True)
    if not steps:
        raise RuntimeError("Replay ended before collecting any tracking metrics")
    errors = np.asarray([row["position_error_m"] for row in steps])
    angles = np.asarray([row["orientation_error_rad"] for row in steps])
    completed = sim_time > duration
    report = {
        "mode": "dynamic_contact_grasp_with_guidance_ik" if args_cli.mode == "dynamic" else "kinematic_object_pose_replay_with_pose_ik",
        "physical_grasp_validated": False,
        "object_pose_commands_during_run": object_pose_commands,
        "source_trajectory": args_cli.trajectory, "device": sim.device, "completed": completed,
        "position_error_mean_m": float(errors.mean()), "position_error_p95_m": float(np.percentile(errors, 95)),
        "position_error_max_m": float(errors.max()), "final_position_error_m": float(errors[-1]),
        "orientation_error_p95_rad": float(np.percentile(angles, 95)),
        "tracking_passed": bool(completed and np.percentile(errors, 95) < 0.02 and errors.max() < 0.04
                                and errors[-1] < 0.01 and np.percentile(angles, 95) < 0.1),
        "samples": steps,
    }
    if reference_paths:
        report["reference_playback"] = {"episode": str(args_cli.reference_episode), "title": args_cli.review_title,
                                        "alignment": "Piecewise capture-frame guidance alignment; robot holds pause the reference. Not equal wall-clock playback."}
    if SCENE_CONTEXT is not None:
        report["scene_context"] = {"source": str(args_cli.scene_context),
                                   "marker_ids": [m["id"] for m in SCENE_CONTEXT["markers"]],
                                   "background_objects": [o["name"] for o in SCENE_CONTEXT["background_objects"]],
                                   "reference_prims_checked": SCENE_CONTEXT["reference_prims_checked"],
                                   "collision_enabled": False, "training_ready": False}
    if args_cli.mode == "dynamic":
        report["task_input"] = TRAJECTORY["task_input"]
        report["physics_assumptions"] = {"mass_kg": args_cli.object_mass_kg, "friction": args_cli.friction,
                                         "per_finger_effort_limit_n": args_cli.grip_force_n,
                                         "gravity_enabled": True, "object_pose_commands_during_run": object_pose_commands}
        report["physical_results"] = [evaluate_physical_grasp(
            [row["environments"][env] for row in steps], TRAJECTORY["task_input"]["goal_xyz_m"],
            completed=completed, release_at_lift=args_cli.release_at_lift,
            goal_yaw_rad=TRAJECTORY["task_input"].get("goal_yaw_rad"),
        ) for env in range(scene.num_envs)]
        report["physical_grasp_validated"] = all(row["physical_grasp_validated"] for row in report["physical_results"])
        check = "release_test_passed" if args_cli.release_at_lift else "pick_place_passed"
        report[check] = all(row[check] for row in report["physical_results"])
        if "goal_yaw_rad" in TRAJECTORY["task_input"]:
            report["pick_place_with_yaw_passed"] = all(row["pick_place_with_yaw_passed"] for row in report["physical_results"])
        print(f"[INFO] Physical results: {report['physical_results']}", flush=True)
    filename = "isaac_physical_metrics.json" if args_cli.mode == "dynamic" else "isaac_metrics.json"
    report_path = args_cli.report or Path(args_cli.trajectory).with_name(filename)
    write_json(report_path, report)
    if args_cli.mode == "dynamic":
        write_json(report_path.with_name(report_path.stem + "_plan.json"), TRAJECTORY)
    print(f"[INFO] Replay complete: {completed}; tracking passed: {report['tracking_passed']}", flush=True)
    print(f"[INFO] Position error mean/p95/max: {errors.mean()*1000:.1f}/{np.percentile(errors,95)*1000:.1f}/{errors.max()*1000:.1f} mm", flush=True)
    print(f"[INFO] Report: {report_path}; video: {args_cli.video}", flush=True)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        device=args_cli.device, dt=1 / 120, render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=args_cli.friction, dynamic_friction=args_cli.friction),
    ))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)
    scene_cfg = RetargetReplaySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    for index, (x, y) in enumerate([(-0.04, -0.35), (-0.04, 0.35), (0.98, -0.35), (0.98, 0.35)]):
        setattr(scene_cfg, f"table_leg_{index}", AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TableLeg" + str(index),
            spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, TABLE_Z - 0.05),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, (TABLE_Z - 0.05) / 2)),
        ))
    for obj in ([] if SCENE_CONTEXT is not None else TRAJECTORY.get("scene_objects", [])):
        if obj["name"] != TRAJECTORY["object_name"]:
            setattr(scene_cfg, obj["name"], RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + obj["name"], spawn=object_spawn_cfg(obj),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(obj["initial_sim_xyz_m"])),
            ))
    if SCENE_CONTEXT is not None:
        for index, obj in enumerate(SCENE_CONTEXT["background_objects"]):
            setattr(scene_cfg, f"reference_object_{index}", AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/ReferenceObject_" + obj["name"],
                spawn=object_spawn_cfg(obj, visual_only=True),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(obj["initial_sim_xyz_m"]),
                                                       rot=tuple(obj.get("initial_quat_wxyz", [1, 0, 0, 0]))),
            ))
        for marker in SCENE_CONTEXT["markers"]:
            for part in marker_visual_parts(marker):
                setattr(scene_cfg, f"marker_{marker['id']}_{part['name']}", AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/ReferenceMarker_" + str(marker["id"]) + "_" + part["name"],
                    spawn=sim_utils.CuboidCfg(size=tuple(part["size_m"]),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(part["color_rgb"]), roughness=1.0)),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(part["pos_m"]), rot=tuple(part["quat_wxyz"])),
                ))
    print("[INFO] Building scene with a primitive table and the measured objects...", flush=True)
    scene = InteractiveScene(scene_cfg)
    if SCENE_CONTEXT is not None:
        from pxr import UsdPhysics
        references = [prim for prim in scene.stage.Traverse()
                      if "/ReferenceObject_" in str(prim.GetPath()) or "/ReferenceMarker_" in str(prim.GetPath())]
        if any(prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in references):
            raise RuntimeError("Reference markers/backgrounds must not modify the physics scene")
        SCENE_CONTEXT["reference_prims_checked"] = len(references)
        print(f"[INFO] Checked {len(references)} visual reference prims: no collision/rigid-body APIs", flush=True)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Immediate Kit shutdown can exit before Python emits its pending exception.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        # Camera stop/drain can hang after artifacts are flushed, including GUI
        # comparison runs. Standalone replay exits here rather than retaining Kit.
        simulation_app.close(wait_for_replicator=False,
                             skip_cleanup=args_cli.headless or bool(args_cli.video or args_cli.reference_episode))
