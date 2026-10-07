from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np
import pytest

from humanoid_retarget.io import write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.scene_context import load_scene_context, marker_layout, marker_visual_parts, prepare_scene_context, stationary_reference
from humanoid_retarget.types import Detection, ObjectSpec


def marker_config():
    return {"workspace": {"desk_points_m": [[-0.0532, 0.0532], [0.0532, 0.0532],
                                            [0.0532, -0.0532], [-0.0532, -0.0532]],
                          "calibration_reference": {"marker_ids": [3], "dictionary": "DICT_4X4_1000",
                                                    "marker_size_m": 0.1064}}}


def test_marker_pattern_is_the_actual_dictionary_id_and_not_mirrored():
    retarget = {"mapping": "desk_y_forward_desk_x_left", "sim_origin_m": [0.55, 0, 1.05], "sim_table_z_m": 1.05}
    marker = marker_layout(marker_config(), retarget)[0]
    assert marker["id"] == 3
    assert np.allclose(marker["initial_sim_xyz_m"], [0.55, 0, 1.05])
    assert np.allclose(marker["quat_wxyz"], [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)])
    assert np.isclose(np.linalg.norm(np.subtract(*marker["sim_corners_m"][:2])), 0.1064)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    image = cv2.resize((1 - np.asarray(marker["cells"], np.uint8)) * 255, (240, 240), interpolation=cv2.INTER_NEAREST)
    image = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    assert ids.reshape(-1).tolist() == [3]
    parts = marker_visual_parts(marker)
    black_area = sum(p["size_m"][0] * p["size_m"][1] for p in parts[1:])
    assert np.isclose(black_area, np.sum(marker["cells"]) * (0.1064 / 6) ** 2)
    assert all(p["pos_m"][2] > 1.05 for p in parts)
    assert marker["collision_enabled"] is False


def test_marker_uses_the_task_centre_scale_and_each_decoded_rotation():
    cfg = marker_config()
    corners = np.asarray(cfg["workspace"]["desk_points_m"])
    rotation = np.asarray([[0, -1], [1, 0]])
    cfg["workspace"]["desk_points_m"] = (corners @ rotation.T + [0.4, -0.2]).tolist()
    marker = marker_layout(cfg, {"mapping": "identity", "sim_origin_m": [1, 2, 3],
                                 "desk_center_m": [0.1, -0.1], "scale_xy": 2})[0]
    assert np.allclose(marker["initial_sim_xyz_m"], [1.6, 1.8, 3])
    assert np.allclose(marker["quat_wxyz"], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    assert marker["black_square_size_m"] == 0.2128


@pytest.mark.parametrize("invalid", ["corners", "size", "dictionary", "reflection", "degenerate"])
def test_marker_invalid_geometry_is_rejected(invalid):
    cfg = marker_config()
    retarget = {"mapping": "identity"}
    if invalid == "corners":
        cfg["workspace"]["desk_points_m"].pop()
    elif invalid == "size":
        cfg["workspace"]["calibration_reference"]["marker_size_m"] = float("nan")
    elif invalid == "dictionary":
        cfg["workspace"]["calibration_reference"]["dictionary"] = "__dict__"
    elif invalid == "reflection":
        retarget = {"mapping": "custom", "xy_matrix": [[-1, 0], [0, 1]]}
    else:
        cfg["workspace"]["desk_points_m"][1] = cfg["workspace"]["desk_points_m"][0]
    with pytest.raises(ValueError):
        marker_layout(cfg, retarget)


def boundary_observations():
    return [Detection(i, i / 30, "box", (100, 80), (0.2 + i * 0.0001, -0.1, 0.02), 500, 0.1)
            for i in range(6)]


def test_stationary_reference_uses_observed_start_and_nominal_table_height():
    spec = ObjectSpec("box", "cuboid", (0.05, 0.04, 0.04), [])
    observations = boundary_observations()
    original = deepcopy(observations)
    obj, diagnostic = stationary_reference(spec, observations, observations,
                                           {"sim_origin_m": [0.55, 0, 1.05], "sim_table_z_m": 1.05}, 0)
    assert np.allclose(obj["initial_sim_xyz_m"], [0.45, -0.20025, 1.07])
    assert obj["collision_enabled"] is False and obj["training_ready"] is False
    assert diagnostic["status"] == "visual_reference"
    assert observations == original


@pytest.mark.parametrize("invalid", ["missing", "moved", "jitter", "lifted"])
def test_uncertain_backgrounds_are_not_invented(invalid):
    spec = ObjectSpec("box", "cuboid", (0.05, 0.04, 0.04), [])
    initial, final = boundary_observations(), boundary_observations()
    if invalid == "missing":
        final = final[:3]
    elif invalid == "moved":
        final = [replace(d, desk_xyz_m=(0.3, -0.1, 0.02)) for d in final]
    elif invalid == "jitter":
        initial[:3] = [replace(d, desk_xyz_m=(0.25, -0.1, 0.02)) for d in initial[:3]]
    else:
        final = [replace(d, desk_xyz_m=(0.2, -0.1, 0.08)) for d in final]
    obj, diagnostic = stationary_reference(spec, initial, final, {}, 0)
    assert obj is None and diagnostic["status"] != "visual_reference"


def test_background_consensus_discards_isolated_depth_outliers():
    spec = ObjectSpec("box", "cuboid", (0.05, 0.04, 0.04), [])
    initial, final = boundary_observations(), boundary_observations()
    initial[0] = replace(initial[0], desk_xyz_m=(0.25, -0.1, 0.02))
    obj, diagnostic = stationary_reference(spec, initial, final, {}, 0)
    assert obj is not None
    assert diagnostic["initial_consensus_observations"] == 5
    assert diagnostic["rejected_boundary_frames"] == [0]
    assert abs(obj["initial_desk_xyz_m"][0] - 0.2) < 0.001


@pytest.mark.parametrize("invalid", [None, "task", "source", "active", "duplicate", "collisions"])
def test_scene_context_is_bound_to_unchanged_sources_and_the_same_task(tmp_path, invalid):
    task = tmp_path / "task.json"
    write_json(task, {"object_name": "box", "source_episode": str(tmp_path / "episode")})
    source = tmp_path / "source.json"
    write_json(source, {})
    context = {"schema_version": "hmnd_scene_context_v1", "selected_object": "box",
               "source_episode": str(tmp_path / "episode"), "source_task_sha256": file_sha256(task),
               "source_file_sha256": {str(source): file_sha256(source)}, "collision_enabled": False,
               "background_objects": [{"name": "sphere"}]}
    if invalid == "task":
        context["source_task_sha256"] = "changed"
    elif invalid == "source":
        write_json(source, {"changed": True})
    elif invalid == "active":
        context["background_objects"][0]["name"] = "box"
    elif invalid == "duplicate":
        context["background_objects"] *= 2
    elif invalid == "collisions":
        context["collision_enabled"] = True
    path = tmp_path / "context.json"
    write_json(path, context)
    if invalid:
        with pytest.raises(ValueError):
            load_scene_context(path, task)
    else:
        assert load_scene_context(path, task)["selected_object"] == "box"


@pytest.fixture
def scene_inputs(tmp_path):
    episode = tmp_path / "episode"
    (episode / "rgb").mkdir(parents=True)
    (episode / "depth").mkdir()
    config = {"workspace": {"image_size_px": [160, 120],
                           "image_points_px": [[0, 0], [159, 0], [159, 119], [0, 119]],
                           "desk_points_m": [[-0.4, 0.3], [0.4, 0.3], [0.4, -0.3], [-0.4, -0.3]],
                           "workspace_outline_m": [[-0.15, 0.15], [0.15, 0.15], [0.15, -0.15], [-0.15, -0.15]],
                           "camera_matrix_px": [[200, 0, 80], [0, 200, 60], [0, 0, 1]],
                           "camera_to_desk_m": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.8], [0, 0, 0, 1]]}}
    workspace, objects = tmp_path / "workspace.yaml", tmp_path / "objects.yaml"
    # JSON is a valid YAML subset; use the project writer for fixtures.
    write_json(workspace, config)
    write_json(objects, {"objects": [{"name": "background", "shape": "cuboid", "size_m": [0.05, 0.04, 0.04],
                                     "min_area_px": 20, "hsv_ranges": [{"lower": [100, 100, 100], "upper": [140, 255, 255]}]}]})
    for i in range(12):
        image = np.full((120, 160, 3), 180, np.uint8)
        image[50:70, 70:90] = (255, 0, 0)
        depth = np.full((120, 160), 800, np.uint16)
        depth[50:70, 70:90] = 760
        cv2.imwrite(str(episode / "rgb" / f"{i:06d}.png"), image)
        cv2.imwrite(str(episode / "depth" / f"{i:06d}.png"), depth)
    write_json(episode / "metadata.json", {"depth_aligned_to": "color", "depth_unit_m": 0.001})
    task = tmp_path / "task.json"
    write_json(task, {"object_name": "active", "source_episode": str(episode), "fps": 30,
                      "workspace": {"retarget": {"sim_origin_m": [0.55, 0, 1.05], "sim_table_z_m": 1.05}},
                      "events": {"workspace_config_sha256": file_sha256(workspace),
                                 "objects_config_sha256": file_sha256(objects)}})
    return task, workspace, objects, episode


@pytest.mark.parametrize("reference", ["task.json", "../task.json", "/tmp/task.json"])
def test_packaged_scene_references_stay_relative_and_contained(tmp_path, reference):
    task = tmp_path / "task.json"
    write_json(task, {"object_name": "box", "source_episode": "recorded/box"})
    scene = tmp_path / "scene.json"
    write_json(scene, {"schema_version": "hmnd_scene_context_v1", "selected_object": "box",
                      "source_episode": "recorded/box", "source_task_sha256": file_sha256(task),
                      "source_files_relative_to": "context",
                      "source_file_sha256": {reference: file_sha256(task)},
                      "collision_enabled": False, "background_objects": []})
    if reference == "task.json":
        assert load_scene_context(scene, task)["selected_object"] == "box"
    else:
        with pytest.raises(ValueError, match="inside the packaged scene directory"):
            load_scene_context(scene, task)


def test_context_preparation_creates_preview_without_changing_sources(scene_inputs, tmp_path):
    task, workspace, objects, episode = scene_inputs
    before = {p: file_sha256(p) for p in episode.rglob("*") if p.is_file()}
    path = tmp_path / "output/scene_context.json"
    context = prepare_scene_context(task, workspace, objects, path)
    assert context["markers"] == []
    assert [o["name"] for o in context["background_objects"]] == ["background"]
    assert context["background_diagnostics"][0]["status"] == "visual_reference"
    assert cv2.imread(str(path.with_name("scene_context_preview.jpg"))).shape == (720, 960, 3)
    assert load_scene_context(path, task)["collision_enabled"] is False
    assert before == {p: file_sha256(p) for p in before}


@pytest.mark.parametrize("invalid", ["source_output", "existing_output", "config", "intrinsics", "unaligned"])
def test_context_preparation_rejects_unsafe_or_mismatched_inputs(scene_inputs, tmp_path, invalid):
    task, workspace, objects, episode = scene_inputs
    path = tmp_path / "output/scene_context.json"
    if invalid == "source_output":
        path = episode / "scene_context.json"
    elif invalid == "existing_output":
        write_json(path, {})
    elif invalid == "config":
        write_json(objects, {})
    elif invalid == "intrinsics":
        write_json(episode / "metadata.json", {"depth_aligned_to": "color",
            "color_intrinsics": {"fx": 201, "fy": 200, "cx": 80, "cy": 60, "width": 160, "height": 120}})
    else:
        write_json(episode / "metadata.json", {"depth_aligned_to": "depth"})
    with pytest.raises(ValueError):
        prepare_scene_context(task, workspace, objects, path)
