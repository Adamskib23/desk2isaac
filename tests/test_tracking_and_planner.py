from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.planner import build_trajectory
from humanoid_retarget.tracking import ObjectSpec, detect_object, filter_detections, load_object_specs, select_active_object, track_episode
from humanoid_retarget.types import Detection


def _write_test_episode(root: Path) -> WorkspaceCalibration:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    calibration = WorkspaceCalibration(
        image_size_px=(160, 120),
        image_points_px=((20, 20), (140, 20), (140, 100), (20, 100)),
        desk_points_m=((-0.3, 0.2), (0.3, 0.2), (0.3, -0.2), (-0.3, -0.2)),
        table_z_m=0.0,
        table_depth_m=1.0,
    )
    for idx in range(24):
        frame = np.full((120, 160, 3), 180, dtype=np.uint8)
        alpha = min(1.0, max(0.0, (idx - 6) / 12.0))
        xy = np.asarray([-0.16 + 0.26 * alpha, 0.02 + 0.08 * alpha], dtype=np.float32)
        px = calibration.desk_to_pixel_xy(xy)
        cv2.rectangle(frame, (int(px[0]) - 8, int(px[1]) - 8), (int(px[0]) + 8, int(px[1]) + 8), (0, 0, 255), -1)
        cv2.imwrite(str(root / "rgb" / f"{idx:06d}.jpg"), frame)
        np.save(root / "depth" / f"{idx:06d}.npy", np.full((120, 160), 1.0, dtype=np.float32))
    with (root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"fps": 12.0, "depth_unit_m": 1.0}, f)
    return calibration


def test_tracking_selects_moving_cube_and_builds_waypoints(tmp_path: Path) -> None:
    calibration = _write_test_episode(tmp_path)
    specs = [
        ObjectSpec(
            name="red_cube",
            shape="cuboid",
            size_m=(0.05, 0.05, 0.05),
            hsv_ranges=[((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))],
            min_area_px=40,
        )
    ]
    detections, metadata = track_episode(tmp_path, specs, calibration)
    trajectory = build_trajectory(
        detections,
        specs,
        calibration,
        metadata,
        {
            "sim_origin_m": [0.55, 0.0, 1.05],
            "sim_table_z_m": 1.05,
            "mapping": "desk_y_forward_desk_x_left",
            "scale_xy": 1.0,
            "min_displacement_m": 0.03,
        },
    )
    assert trajectory.object_name == "red_cube"
    assert trajectory.events["max_displacement_m"] > 0.2
    assert len(trajectory.waypoints) >= 8
    assert trajectory.waypoints[0]["gripper"] == "open"
    assert any(wp["gripper"] == "closed" for wp in trajectory.waypoints)


def test_black_markers_and_background_are_excluded_from_tracking() -> None:
    calibration = WorkspaceCalibration(
        image_size_px=(300, 200),
        image_points_px=((0, 0), (299, 0), (299, 199), (0, 199)),
        desk_points_m=((0, 0.2), (0.3, 0.2), (0.3, 0), (0, 0)),
        workspace_roi_px=(80, 10, 200, 180),
        excluded_polygons_px=(((100, 20), (180, 20), (180, 100), (100, 100)),),
    )
    frame = np.full((200, 300, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (70, 199), (0, 0, 0), -1)
    cv2.rectangle(frame, (100, 20), (180, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (220, 120), (250, 145), (0, 0, 0), -1)
    spec = ObjectSpec("black_box", "cuboid", (0.05555, 0.045, 0.031),
                      [((0, 0, 0), (179, 255, 80))], min_area_px=100)
    detection = detect_object(frame, None, spec, calibration, 0, 0)
    assert detection is not None
    assert np.allclose(detection.pixel_xy, (235, 132.5), atol=1)


@pytest.mark.parametrize("hue", [10, 40, 140])
def test_real_black_box_accepts_low_saturation_without_relying_on_hue(hue: int) -> None:
    specs = load_object_specs(Path(__file__).resolve().parents[1] / "configs/objects.real.yaml")
    spec = next(s for s in specs if s.name == "black_box")
    hsv = np.full((200, 300, 3), (20, 65, 175), dtype=np.uint8)
    hsv[90:140, 120:180] = (hue, 25, 90)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    calibration = WorkspaceCalibration(
        (300, 200), ((0, 0), (299, 0), (299, 199), (0, 199)),
        ((0, 0.2), (0.3, 0.2), (0.3, 0), (0, 0)),
    )
    detection = detect_object(frame, None, spec, calibration, 0, 0)
    assert detection is not None
    assert np.allclose(detection.pixel_xy, (149.5, 114.5), atol=1)


def test_depth_comes_from_selected_component_only() -> None:
    calibration = WorkspaceCalibration(
        image_size_px=(180, 140),
        image_points_px=((0, 0), (179, 0), (179, 139), (0, 139)),
        desk_points_m=((0, 0.14), (0.18, 0.14), (0.18, 0), (0, 0)),
        table_depth_m=1.0,
    )
    frame = np.full((140, 180, 3), 220, dtype=np.uint8)
    depth = np.ones((140, 180), dtype=np.float32)
    for (left, top, right, bottom), distance in [((15, 25, 60, 70), 0.9),
                                               ((95, 15, 130, 50), 0.6),
                                               ((95, 80, 130, 115), 0.6)]:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), -1)
        depth[top:bottom + 1, left:right + 1] = distance
    spec = ObjectSpec("red_cube", "cuboid", (0.05, 0.05, 0.05),
                      [((0, 80, 50), (10, 255, 255))], min_area_px=100)
    detection = detect_object(frame, depth, spec, calibration, 0, 0)
    assert detection is not None
    assert np.isclose(detection.desk_xyz_m[2], 0.075, atol=1e-5)


def test_lift_height_uses_camera_transform() -> None:
    angle = 0.25
    rotation = cv2.Rodrigues(np.asarray([angle, 0, 0]))[0] @ np.diag([1.0, -1.0, -1.0])
    translation = np.asarray([0, 0, 0.8])
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = -rotation.T @ translation
    matrix = np.asarray([[600.0, 0, 160], [0, 600.0, 120], [0, 0, 1]])
    calibration = WorkspaceCalibration(
        (320, 240), ((0, 0), (319, 0), (319, 239), (0, 239)),
        ((-0.2, 0.2), (0.2, 0.2), (0.2, -0.2), (-0.2, -0.2)),
        camera_matrix_px=tuple(tuple(row) for row in matrix),
        camera_to_desk_m=tuple(tuple(row) for row in transform),
    )
    center = np.asarray([0.05, 0.02, 0.15])
    surface = center + [0, 0, 0.025]
    pixel = calibration.desk_to_pixel_xy(center[:2])
    actual_pixel = cv2.projectPoints(surface.reshape(1, 3), cv2.Rodrigues(rotation)[0], translation,
                                     matrix, None)[0].reshape(2)
    assert np.linalg.norm(actual_pixel - pixel) > 5
    px, py = np.round(actual_pixel).astype(int)
    frame = np.full((240, 320, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (px - 10, py - 10), (px + 10, py + 10), (0, 0, 255), -1)
    depth = np.full((240, 320), (rotation @ surface + translation)[2], dtype=np.float32)
    spec = ObjectSpec("red_cube", "cuboid", (0.05, 0.05, 0.05),
                      [((0, 80, 50), (10, 255, 255))], min_area_px=100)
    detection = detect_object(frame, depth, spec, calibration, 0, 0)
    assert detection.position_source == "aligned_depth_surface"
    assert np.allclose(detection.desk_xyz_m, center, atol=0.005)


def test_stationary_noise_does_not_trigger_manipulation() -> None:
    random = np.random.default_rng(7)
    detections = [Detection(i, i / 30, "black_box", (100, 100), tuple(xyz), 1000, 0.9)
                  for i, xyz in enumerate(random.normal(0, 0.001, (300, 3)) + [0.1, 0.2, 0.0155])]
    with pytest.raises(ValueError, match="No object moved"):
        select_active_object(detections)


def test_workspace_polygon_excludes_background_clothing() -> None:
    calibration = WorkspaceCalibration(
        (300, 200), ((0, 0), (299, 0), (299, 199), (0, 199)),
        ((0, 0.2), (0.3, 0.2), (0.3, 0), (0, 0)),
        workspace_outline_m=((0.1, 0.19), (0.29, 0.10), (0.29, 0.02), (0.1, 0.02)),
    )
    frame = np.full((200, 300, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (220, 10), (270, 60), (0, 0, 0), -1)
    cv2.rectangle(frame, (210, 110), (240, 135), (0, 0, 0), -1)
    spec = ObjectSpec("black_box", "cuboid", (0.05555, 0.045, 0.031),
                      [((0, 0, 0), (179, 255, 80))], min_area_px=100)
    detection = detect_object(frame, None, spec, calibration, 0, 0)
    assert detection is not None
    assert np.allclose(detection.pixel_xy, (225, 122.5), atol=1)


def test_lift_and_return_can_select_active_object() -> None:
    detections = [Detection(i, i / 30, "black_box", (100, 100),
                            (0.1, 0.2, 0.0155 + (0.1 if 20 <= i < 60 else 0)), 1000, 0.9)
                  for i in range(80)]
    assert select_active_object(detections)[0] == "black_box"


def test_kalman_tracking_reduces_stationary_pixel_and_metric_noise() -> None:
    rng = np.random.default_rng(12)
    rows = [Detection(i, i / 30, "box", tuple(rng.normal(0, 4, 2) + [200, 100]),
                      tuple(rng.normal(0, 0.006, 3) + [0.1, 0.2, 0.05]), 1000, 1.0)
            for i in range(300)]
    filtered = filter_detections(rows, 30)
    raw_xyz = np.asarray([d.desk_xyz_m for d in rows[20:]])
    smooth_xyz = np.asarray([d.desk_xyz_m for d in filtered[20:]])
    raw_pixels = np.asarray([d.pixel_xy for d in rows[20:]])
    smooth_pixels = np.asarray([d.pixel_xy for d in filtered[20:]])
    assert np.all(smooth_xyz.std(0) < 0.85 * raw_xyz.std(0))
    assert np.all(smooth_pixels.std(0) < 0.85 * raw_pixels.std(0))


def test_kalman_tracking_gates_a_spike_without_claiming_visibility() -> None:
    rows = [Detection(i, i / 30, "box", (100, 100),
                      (0.5, 0.5, 0.5) if i == 50 else (0.1, 0.2, 0.0155), 1000, 1.0)
            for i in range(100)]
    filtered = filter_detections(rows, 30)
    assert not filtered[50].visible
    assert filtered[50].position_source == "kalman_prediction"
    assert filtered[50].confidence <= 0.2
    assert np.allclose(filtered[50].desk_xyz_m, (0.1, 0.2, 0.0155), atol=0.001)
    assert filtered[51].visible
    assert rows[50].visible


def test_kalman_tracking_preserves_a_constant_velocity_move_and_resets_after_gap() -> None:
    rng = np.random.default_rng(9)
    truth = np.asarray([[i / 30 * 0.15, 0.1, 0.02] for i in range(120)])
    rows = [Detection(i, i / 30, "box", (100 + i, 100), tuple(p + rng.normal(0, 0.003, 3)), 1000, 1.0)
            for i, p in enumerate(truth)]
    filtered = filter_detections(rows, 30)
    positions = np.asarray([d.desk_xyz_m for d in filtered])
    assert np.sqrt(np.mean((positions[10:] - truth[10:]) ** 2)) < 0.006
    assert select_active_object(filtered)[0] == "box"
    gap = Detection(200, 200 / 30, "box", (200, 200), (0.3, 0.2, 0.02), 1000, 1.0)
    assert filter_detections([*rows, gap], 30)[-1].desk_xyz_m == gap.desk_xyz_m
    with pytest.raises(ValueError, match="positive"):
        filter_detections(rows, 0)


def _planned_pick_place(return_to_start: bool = False):
    calibration = WorkspaceCalibration(
        (300, 200), ((0, 0), (299, 0), (299, 199), (0, 199)),
        ((0, 0.2), (0.3, 0.2), (0.3, 0), (0, 0)),
    )
    rows = []
    for i in range(120):
        alpha = np.clip((i - 25) / 30, 0, 1)
        x = 0.1 if return_to_start else 0.1 + 0.15 * alpha
        z = 0.0155 + (0.08 * np.sin(np.pi * alpha) if 25 < i < 55 else 0)
        rows.append(Detection(i, i / 30, "box", (100, 100), (x, 0.1, z), 1000, 1.0))
    spec = ObjectSpec("box", "cuboid", (0.05555, 0.045, 0.031), [((0, 0, 0), (179, 255, 100))])
    return build_trajectory(rows, [spec], calibration, {"fps": 30}, {
        "sim_origin_m": [0.55, 0, 1.05], "sim_table_z_m": 1.05,
        "mapping": "identity", "hand_to_grasp_offset_m": 0.1034,
    })


@pytest.mark.parametrize("return_to_start", [False, True])
def test_placement_uses_final_stable_hold_not_the_recording_end(return_to_start: bool) -> None:
    trajectory = _planned_pick_place(return_to_start)
    assert 15 < trajectory.events["pick_frame"] < 40
    assert 45 < trajectory.events["place_frame"] < 70
    assert trajectory.events["placement_settled"]
    assert trajectory.events["max_lift_m"] > 0.07


def test_palm_targets_follow_observed_3d_motion_with_grasp_offset() -> None:
    trajectory = _planned_pick_place()
    initial = trajectory.waypoints[0]["object_xyz_m"]
    assert np.allclose(initial, (0.65, 0.1, 1.0655), atol=0.001)
    carried = [w for w in trajectory.waypoints if w["phase"] in ("lift", "carry_observed_object_path")]
    assert all(np.allclose(np.asarray(w["ee_xyz_m"]) - w["object_xyz_m"], [0, 0, 0.1034], atol=1e-6) for w in carried)
    assert np.ptp([w["ee_xyz_m"][2] for w in carried]) > 0.05
    assert np.all(np.diff([w["t"] for w in trajectory.waypoints]) > 0)
    assert len(trajectory.scene_objects) == 1
    assert trajectory.scene_objects[0]["size_m"] == (0.05555, 0.045, 0.031)


def _explicit_selection_inputs():
    calibration = WorkspaceCalibration(
        (300, 200), ((0, 0), (299, 0), (299, 199), (0, 199)),
        ((0, 0.2), (0.3, 0.2), (0.3, 0), (0, 0)),
    )
    specs = [ObjectSpec(name, "cuboid", (0.05, 0.05, 0.05), [((0, 0, 0), (179, 255, 100))])
             for name in ("box", "shadow", "stationary")]
    rows = []
    for i in range(120):
        alpha = np.clip((i - 25) / 30, 0, 1)
        rows.extend([
            Detection(i, i / 30, "box", (100, 100), (0.1 + 0.15 * alpha, 0.1, 0.025), 1000, 1.0),
            Detection(i, i / 30, "shadow", (100, 100), (0.2 + 0.15 * np.sin(i), 0.1, 0.025), 1000, 1.0),
            Detection(i, i / 30, "stationary", (100, 100), (0.1, 0.1, 0.025), 1000, 1.0),
        ])
    return rows, specs, calibration


def test_explicit_object_label_overrides_false_motion_but_preserves_scene_tracks():
    rows, specs, calibration = _explicit_selection_inputs()
    automatic = build_trajectory(rows, specs, calibration, {"fps": 30}, {"sim_origin_m": [0.55, 0, 1.05]})
    assert automatic.object_name == "shadow"
    declared = build_trajectory(rows, specs, calibration, {"fps": 30}, {"sim_origin_m": [0.55, 0, 1.05]}, object_name="box")
    assert declared.object_name == "box"
    assert declared.events["selection_method"] == "declared_object"
    assert declared.events["automatic_object_name"] == "shadow"
    assert len(declared.scene_objects) == 3
    assert len(declared.object_trace) == 120


@pytest.mark.parametrize("name, message", [("unknown", "not configured"), ("stationary", "No object moved")])
def test_declared_label_cannot_invent_an_unknown_or_stationary_task(name, message):
    rows, specs, calibration = _explicit_selection_inputs()
    with pytest.raises(ValueError, match=message):
        build_trajectory(rows, specs, calibration, {"fps": 30}, {}, object_name=name)


def test_batch_metadata_label_selects_the_declared_object():
    rows, specs, calibration = _explicit_selection_inputs()
    trajectory = build_trajectory(rows, specs, calibration, {
        "fps": 30, "action_object": "box",
    }, {"sim_origin_m": [0.55, 0, 1.05]})
    assert trajectory.object_name == "box"
    assert trajectory.events["selection_method"] == "declared_object"
    assert trajectory.events["declaration_source"] == "episode_metadata"
    assert trajectory.events["automatic_object_name"] == "shadow"
    assert len(trajectory.scene_objects) == 3


def test_explicit_argument_overrides_the_batch_metadata_label():
    rows, specs, calibration = _explicit_selection_inputs()
    trajectory = build_trajectory(rows, specs, calibration, {
        "fps": 30, "action_object": "unknown",
    }, {}, object_name="box")
    assert trajectory.object_name == "box"
    assert trajectory.events["declaration_source"] == "explicit_argument"


@pytest.mark.parametrize("label,message", [
    ("unknown", "not configured"), ("stationary", "No object moved"),
    ({"name": "box"}, "object-name string"),
])
def test_batch_metadata_label_must_be_valid_and_still_have_real_motion(label, message):
    rows, specs, calibration = _explicit_selection_inputs()
    with pytest.raises(ValueError, match=message):
        build_trajectory(rows, specs, calibration, {"fps": 30, "action_object": label}, {})
