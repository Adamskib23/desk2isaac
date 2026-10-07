from copy import deepcopy

import numpy as np
import pytest

from humanoid_retarget.yaw_tracking import (
    attach_yaw, axis_difference, axis_mean, boundary_yaw, desk_to_sim_yaw,
    estimate_episode_yaw, fit_upright_yaw, quaternion_yaw_tilt, yaw_from_depth_mask, yaw_quaternion,
)


def upper_face(angle, size=(0.05555, 0.045, 0.031), tilt=0):
    x, y = np.meshgrid(np.linspace(-size[0] / 2, size[0] / 2, 40),
                       np.linspace(-size[1] / 2, size[1] / 2, 30))
    points = np.column_stack((x.ravel(), y.ravel()))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = points @ rotation.T
    return np.column_stack((xy + [0.2, 0.3], size[2] + xy[:, 0] * np.tan(tilt)))


@pytest.mark.parametrize("angle", [-89, -65, -20, 0, 20, 65, 89, 179])
@pytest.mark.parametrize("size", [(0.05555, 0.045, 0.031), (0.051, 0.035, 0.085), (0.035, 0.051, 0.085)])
def test_metric_rectangle_recovers_local_x_axis_modulo_180(angle, size):
    result = fit_upright_yaw(upper_face(np.deg2rad(angle), size), size)
    assert result["status"] == "observed"
    assert abs(axis_difference(result["desk_yaw_rad"], np.deg2rad(angle))) < np.deg2rad(0.1)


def test_square_partial_and_tilted_geometry_are_not_labels():
    assert fit_upright_yaw(upper_face(0), [0.05, 0.05, 0.03])["status"] == "ambiguous_square_footprint"
    assert fit_upright_yaw(upper_face(0)[:15], [0.05555, 0.045, 0.031])["status"] == "insufficient_upper_face"
    points = upper_face(0)
    points[:, 0] = 0.2 + (points[:, 0] - 0.2) * 0.5
    assert fit_upright_yaw(points, [0.05555, 0.045, 0.031])["status"] == "uncertain_geometry"
    assert fit_upright_yaw(upper_face(0, tilt=np.deg2rad(25)), [0.05555, 0.045, 0.031])["status"] == "uncertain_geometry"


def test_axis_wrap_and_default_desk_mapping():
    mean = axis_mean(np.deg2rad([89, -89]))
    assert abs(axis_difference(mean, np.pi / 2)) < 1e-8
    assert desk_to_sim_yaw(0.4, {}) == pytest.approx(0.4 - np.pi / 2)
    assert desk_to_sim_yaw(0.4, {"mapping": "identity"}) == pytest.approx(0.4)
    assert desk_to_sim_yaw(0.4, {"mapping": "custom", "xy_matrix": [[0, -1], [1, 0]]}) == pytest.approx(0.4 - np.pi / 2)
    with pytest.raises(ValueError):
        axis_mean([0, np.pi / 2])


@pytest.mark.parametrize("matrix", [[[1, 0], [0, -1]], [[1, 0.1], [0, 1]], [[0, 0], [0, 0]]])
def test_yaw_cannot_follow_reflected_sheared_or_degenerate_maps(matrix):
    with pytest.raises(ValueError, match="uniform XY"):
        desk_to_sim_yaw(0, {"mapping": "custom", "xy_matrix": matrix})


def test_quaternions_rotate_downward_hand_and_upright_object():
    yaw, tilt = quaternion_yaw_tilt([yaw_quaternion(0.7), yaw_quaternion(0.7 + np.pi)])
    assert np.max(np.abs(axis_difference(yaw, 0.7))) < 1e-8
    assert np.allclose(tilt, 0)
    hand = yaw_quaternion(0.7, downward=True)
    assert hand == pytest.approx([0, np.cos(0.35), np.sin(0.35), 0])
    _, tilt = quaternion_yaw_tilt([hand])
    assert tilt[0] == pytest.approx(np.pi)
    with pytest.raises(ValueError):
        quaternion_yaw_tilt([[0, 0, 0, 0]])


def observations():
    return [{"frame_idx": i, "status": "observed", "desk_yaw_rad": np.deg2rad(80 + i / 2)} for i in range(60)]


def test_boundary_consensus_does_not_accept_missing_or_unstable_endpoints():
    rows = observations()
    assert boundary_yaw(rows, 60, 30)["passed"]
    for row in rows[:5]:
        row["status"] = "occluded"
    assert not boundary_yaw(rows, 60, 30)["passed"]
    rows = observations()
    for row in rows[-12:]:
        row["desk_yaw_rad"] = np.deg2rad(30 if row["frame_idx"] % 2 else 80)
    assert not boundary_yaw(rows, 60, 30, final=True)["passed"]


def test_attach_preserves_xyz_and_marks_only_observed_yaw():
    task = {"fps": 30, "object": {"shape": "cuboid"}, "workspace": {"retarget": {}},
            "events": {"pick_frame": 12, "place_frame": 45},
            "object_trace": [{"frame_idx": i, "visible": True, "sim_xyz_m": [0.5, 0, 1.1]} for i in range(60)]}
    rows = observations()
    rows[30] = {"frame_idx": 30, "status": "uncertain_geometry"}
    original, raw = deepcopy(task), deepcopy(rows)
    out = attach_yaw(task, rows, 60)
    assert task == original and rows == raw
    assert out["orientation"]["mode"] == "observed_yaw"
    assert not out["object_trace"][30]["yaw_observed"]
    assert out["object_trace"][30]["desk_yaw_rad"] is None
    assert all(a["sim_xyz_m"] == b["sim_xyz_m"] for a, b in zip(task["object_trace"], out["object_trace"]))
    assert np.max(np.abs(np.diff([r["sim_yaw_rad"] for r in out["orientation"]["control_knots"]]))) < np.deg2rad(20)


def test_sphere_yaw_is_unobservable_not_identity_ground_truth():
    from types import SimpleNamespace
    assert yaw_from_depth_mask(None, None, SimpleNamespace(shape="sphere"), None)["status"] == "unobservable_symmetric_shape"
    task = {"fps": 30, "object": {"shape": "sphere"}, "object_trace": []}
    out = attach_yaw(task, [{"frame_idx": i, "status": "unobservable_symmetric_shape"} for i in range(60)], 60)
    assert out["orientation"]["mode"] == "position_only"
    assert "goal_sim_yaw_rad" not in out["orientation"]


@pytest.fixture
def rgbd_yaw_capture(tmp_path):
    import cv2
    from humanoid_retarget.calibration import WorkspaceCalibration
    from humanoid_retarget.io import write_json
    from humanoid_retarget.mask_tracking import file_sha256
    from humanoid_retarget.types import ObjectSpec
    calibration = WorkspaceCalibration(
        (160, 120), ((0, 0), (159, 0), (159, 119), (0, 119)),
        ((-0.4, 0.3), (0.4, 0.3), (0.4, -0.3), (-0.4, -0.3)),
        workspace_outline_m=((-0.15, 0.15), (0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)),
        camera_matrix_px=((200, 0, 80), (0, 200, 60), (0, 0, 1)),
        camera_to_desk_m=((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0.8), (0, 0, 0, 1)),
    )
    spec = ObjectSpec("box", "cuboid", (0.05555, 0.045, 0.031), [], min_area_px=20)
    episode, masks = tmp_path / "episode", tmp_path / "masks"
    (episode / "rgb").mkdir(parents=True)
    (episode / "depth").mkdir()
    masks.mkdir()
    write_json(episode / "metadata.json", {"depth_aligned_to": "color", "depth_unit_m": 0.001,
                                           "timestamps_relative_s": [0, 0.04, 0.08]})
    mask = np.zeros((120, 160), np.uint8)
    mask[53:67, 72:88] = 255
    entries = []
    for i in range(3):
        rgb, mask_file = episode / "rgb" / f"{i:06d}.jpg", masks / f"{i:06d}.png"
        cv2.imwrite(str(rgb), np.full((120, 160, 3), 100, np.uint8))
        cv2.imwrite(str(mask_file), mask)
        cv2.imwrite(str(episode / "depth" / f"{i:06d}.png"), np.full((120, 160), 769, np.uint16))
        entries.append({"frame_idx": i, "rgb_name": rgb.name, "rgb_sha256": file_sha256(rgb),
                        "mask_sha256": file_sha256(mask_file)})
    path = masks / "manifest.json"
    write_json(path, {"schema_version": "hmnd_sam2_masks_v1", "completed": True,
                      "source_episode": str(episode), "object_name": "box", "frames": entries,
                      "source_metadata_sha256": file_sha256(episode / "metadata.json")})
    task = {"source_episode": str(episode), "object_name": "box", "object": {"shape": "cuboid", "size_m": list(spec.size_m)},
            "events": {"mask_manifest": str(path), "mask_manifest_sha256": file_sha256(path), "pick_frame": 0, "place_frame": 2},
            "object_trace": [{"frame_idx": i, "visible": True} for i in range(3)], "fps": 30,
            "workspace": {"retarget": {"mapping": "identity"}}}
    return task, spec, calibration, mask


def test_rgbd_yaw_uses_depth_and_binds_source_files(rgbd_yaw_capture):
    task, spec, calibration, mask = rgbd_yaw_capture
    result = estimate_episode_yaw(task, spec, calibration)
    assert result["orientation"]["mode"] == "observed_yaw"
    assert result["orientation"]["frames"][1]["capture_t_s"] == 0.04
    assert abs(result["orientation"]["start_sim_yaw_rad"]) < 1e-6
    assert len(result["orientation"]["source_file_sha256"]) == 8
    assert yaw_from_depth_mask(mask, np.full(mask.shape, 0.8), spec, calibration)["status"] == "no_workspace_foreground"
    assert yaw_from_depth_mask(mask, None, spec, calibration)["status"] == "missing_depth"
    with pytest.raises(ValueError, match="dimensions"):
        yaw_from_depth_mask(mask, np.ones((10, 10)), spec, calibration)


@pytest.mark.parametrize("changed", ["rgb", "mask", "metadata", "object"])
def test_yaw_rejects_changed_capture_and_object_identity(rgbd_yaw_capture, changed):
    from pathlib import Path
    from humanoid_retarget.io import read_json, write_json
    task, spec, calibration, mask = rgbd_yaw_capture
    if changed == "object":
        task["object_name"] = "another_object"
    else:
        path = (Path(task["source_episode"]) / "metadata.json" if changed == "metadata" else
                Path(task["source_episode"]) / "rgb/000001.jpg" if changed == "rgb" else
                Path(task["events"]["mask_manifest"]).parent / "000001.png")
        if changed == "metadata":
            metadata = read_json(path)
            metadata["changed"] = True
            write_json(path, metadata)
        else:
            with path.open("ab") as stream:
                stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint|do not match"):
        estimate_episode_yaw(task, spec, calibration)
