from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import read_json, write_json
from humanoid_retarget.mask_tracking import (
    detection_from_depth_mask, endpoint_observation_check, file_sha256, track_mask_episode,
)
from humanoid_retarget.types import Detection, ObjectSpec


@pytest.fixture
def calibration():
    return WorkspaceCalibration(
        (160, 120), ((0, 0), (159, 0), (159, 119), (0, 119)),
        ((-0.4, 0.3), (0.4, 0.3), (0.4, -0.3), (-0.4, -0.3)),
        workspace_outline_m=((-0.15, 0.15), (0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)),
        camera_matrix_px=((200, 0, 80), (0, 200, 60), (0, 0, 1)),
        camera_to_desk_m=((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0.8), (0, 0, 0, 1)),
    )


@pytest.fixture
def spec():
    return ObjectSpec("box", "cuboid", (0.05555, 0.045, 0.031), [], min_area_px=20)


@pytest.fixture
def mask():
    result = np.zeros((120, 160), np.uint8)
    result[50:70, 70:90] = 255
    return result


def test_mask_uses_measured_height_not_shadow_or_nominal_fallback(calibration, spec, mask):
    d, info = detection_from_depth_mask(mask, np.full(mask.shape, 0.70), spec, calibration, 3, 0.12)
    assert d is not None and d.frame_idx == 3 and d.t == 0.12
    assert d.position_source == "sam2_aligned_depth_height_ray"
    assert np.isclose(d.desk_xyz_m[2], 0.1 - 0.031 / 2)
    assert d.confidence > 0.9 and info["status"] == "observed"
    d, info = detection_from_depth_mask(mask, np.full(mask.shape, 0.8), spec, calibration, 0, 0)
    assert d is None and info["status"] == "no_workspace_foreground"


def test_elevated_object_survives_marker_image_exclusion(calibration, spec, mask):
    calibration = replace(calibration, excluded_polygons_px=(((60, 40), (100, 40), (100, 80), (60, 80)),),
                          workspace_roi_px=(0, 0, 1, 1))
    d, _ = detection_from_depth_mask(mask, np.full(mask.shape, 0.7), spec, calibration, 0, 0)
    assert d is not None


def test_elevated_object_survives_desk_plane_image_boundary(calibration, spec):
    calibration = replace(calibration, workspace_outline_m=((-0.1, 0.1), (0.1, 0.1), (0.1, -0.1), (-0.1, -0.1)))
    mask = np.zeros((120, 160), np.uint8)
    mask[50:70, 107:121] = 255
    assert calibration.desk_to_pixel_xy((0.1, 0))[0] == 105
    d, _ = detection_from_depth_mask(mask, np.full(mask.shape, 0.45), spec, calibration, 0, 0)
    assert d is not None and 0 < d.desk_xyz_m[0] < 0.1


def test_object_outside_metric_workspace_is_rejected(calibration, spec):
    mask = np.zeros((120, 160), np.uint8)
    mask[50:70, 135:155] = 255
    d, info = detection_from_depth_mask(mask, np.full(mask.shape, 0.7), spec, calibration, 0, 0)
    assert d is None and info["status"] == "no_workspace_foreground"


@pytest.mark.parametrize("depth", [None, 0.0, float("nan"), float("inf")])
def test_missing_invalid_depth_never_creates_a_nominal_label(calibration, spec, mask, depth):
    depth = None if depth is None else np.full(mask.shape, depth)
    d, _ = detection_from_depth_mask(mask, depth, spec, calibration, 0, 0)
    assert d is None


def test_large_hand_like_region_is_not_accepted_as_box(calibration, spec):
    mask = np.zeros((120, 160), np.uint8)
    mask[20:100, 40:120] = 255
    d, info = detection_from_depth_mask(mask, np.full(mask.shape, 0.7), spec, calibration, 0, 0)
    assert d is None and info["status"] == "implausible_object_extent"


def test_depth_and_mask_shape_must_match(calibration, spec, mask):
    with pytest.raises(ValueError, match="aligned"):
        detection_from_depth_mask(mask, np.ones((10, 10)), spec, calibration, 0, 0)
    with pytest.raises(ValueError, match="dimensions"):
        detection_from_depth_mask(np.ones((10, 10)), None, spec, calibration, 0, 0)


def test_no_3d_calibration_is_an_error(calibration, spec, mask):
    calibration = replace(calibration, camera_matrix_px=None, camera_to_desk_m=None)
    with pytest.raises(ValueError, match="calibrated"):
        detection_from_depth_mask(mask, np.ones(mask.shape), spec, calibration, 0, 0)


@pytest.fixture
def episode_and_masks(tmp_path, mask):
    episode, masks = tmp_path / "episode", tmp_path / "masks"
    (episode / "rgb").mkdir(parents=True)
    (episode / "depth").mkdir()
    masks.mkdir()
    write_json(episode / "metadata.json", {"fps": 30, "depth_aligned_to": "color", "depth_unit_m": 0.001,
                                           "timestamps_relative_s": [0.0, 0.041, 0.072], "action_object": "box"})
    entries = []
    for index in range(3):
        source = episode / "rgb" / f"{index:06d}.jpg"
        output = masks / f"{index:06d}.png"
        cv2.imwrite(str(source), np.full((120, 160, 3), 100, np.uint8))
        cv2.imwrite(str(episode / "depth" / f"{index:06d}.png"), np.full(mask.shape, 700, np.uint16))
        cv2.imwrite(str(output), mask)
        entries.append({"frame_idx": index, "rgb_name": source.name, "rgb_sha256": file_sha256(source),
                        "mask_sha256": file_sha256(output)})
    write_json(masks / "manifest.json", {"schema_version": "hmnd_sam2_masks_v1", "completed": True,
                                         "source_episode": str(episode), "object_name": "box", "frames": entries,
                                         "source_metadata_sha256": file_sha256(episode / "metadata.json")})
    return episode, masks


def test_cached_masks_need_no_torch_and_preserve_real_timestamps(episode_and_masks, calibration, spec):
    episode, masks = episode_and_masks
    rows, meta = track_mask_episode(episode, [spec], calibration, masks)
    assert len(rows) == 3 and [d.t for d in rows] == [0.0, 0.041, 0.072]
    assert meta["tracking_backend"] == "sam2_rgbd" and not meta["training_ready"]
    assert meta["scene_object_coverage"] == "selected_object_only"
    assert all(len(row["depth_sha256"]) == 64 for row in meta["frame_diagnostics"])


@pytest.mark.parametrize("change, message", [("rgb", "RGB source"), ("mask", "Mask fingerprint"),
                                              ("metadata", "metadata changed"), ("order", "exactly once"),
                                              ("incomplete", "incomplete")])
def test_provenance_and_complete_frame_order_are_checked(episode_and_masks, calibration, spec, change, message):
    episode, masks = episode_and_masks
    if change == "rgb":
        cv2.imwrite(str(episode / "rgb/000001.jpg"), np.zeros((120, 160, 3), np.uint8))
    elif change == "mask":
        cv2.imwrite(str(masks / "000001.png"), np.zeros((120, 160), np.uint8))
    elif change == "metadata":
        write_json(episode / "metadata.json", {"depth_aligned_to": "color", "depth_unit_m": 0.01})
    else:
        manifest = read_json(masks / "manifest.json")
        if change == "order":
            manifest["frames"].reverse()
        else:
            manifest["completed"] = False
        write_json(masks / "manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        track_mask_episode(episode, [spec], calibration, masks)


def test_saved_prompt_cannot_be_used_on_another_episode(episode_and_masks, tmp_path):
    episode, _ = episode_and_masks
    path = Path(__file__).resolve().parents[1] / "scripts/segment_sam2_episode.py"
    module_spec = importlib.util.spec_from_file_location("sam2_episode_script", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(path.parent))
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    source = episode / "rgb/000000.jpg"
    prompt = tmp_path / "prompt.json"
    write_json(prompt, {"frame_idx": 0, "source_image": str(tmp_path / "another/rgb/000000.jpg"),
                        "source_sha256": file_sha256(source), "box_xyxy_px": [1, 1, 10, 10]})
    with pytest.raises(ValueError, match="another frame"):
        module.saved_prompt(prompt, [source])


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_mask_fps_override_is_validated(episode_and_masks, calibration, spec, fps):
    episode, masks = episode_and_masks
    with pytest.raises(ValueError, match="FPS"):
        track_mask_episode(episode, [spec], calibration, masks, fps_override=fps)


@pytest.mark.parametrize("field, value, message", [
    ("color_intrinsics", {"fx": 201, "fy": 200, "cx": 80, "cy": 60, "width": 160, "height": 120}, "intrinsics"),
    ("color_distortion", [0, 0, 0, 0], "distortion"),
    ("depth_unit_m", 0, "Depth unit"),
])
def test_capture_geometry_must_match(episode_and_masks, calibration, spec, field, value, message):
    episode, masks = episode_and_masks
    metadata = read_json(episode / "metadata.json")
    metadata[field] = value
    write_json(episode / "metadata.json", metadata)
    with pytest.raises(ValueError, match=message):
        track_mask_episode(episode, [spec], calibration, masks)


def _endpoint_rows():
    return [Detection(i, i / 30, "box", (80, 60), (0, 0, 0.1), 1000, 1.0) for i in range(100)]


def test_endpoint_checks_require_visible_supported_boundary_observations():
    rows = _endpoint_rows()
    assert endpoint_observation_check(rows, 100, 30)["passed"]
    assert not endpoint_observation_check(rows[12:], 100, 30)["initial"]["passed"]
    assert not endpoint_observation_check(rows[:-12], 100, 30)["final"]["passed"]
    predicted = [replace(d, visible=False) if d.frame_idx >= 88 else d for d in rows]
    assert not endpoint_observation_check(predicted, 100, 30)["passed"]
    weak = [replace(d, confidence=0.5) if d.frame_idx >= 88 else d for d in rows]
    assert not endpoint_observation_check(weak, 100, 30)["passed"]


def test_endpoint_checks_allow_small_depth_dropouts_but_not_missing_boundary():
    rows = _endpoint_rows()
    assert endpoint_observation_check(rows[:-2], 100, 30)["passed"]
    assert not endpoint_observation_check(rows[:-3], 100, 30)["passed"]


@pytest.fixture
def build_tool():
    path = Path(__file__).resolve().parents[1] / "scripts/build_trajectory.py"
    module_spec = importlib.util.spec_from_file_location("mask_build_tool", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("location", ["episode", "masks", "existing", "reserved"])
def test_sam_output_never_overwrites_sources_or_prior_results(build_tool, tmp_path, location):
    episode, masks = tmp_path / "episode", tmp_path / "masks"
    if location in ("episode", "masks"):
        out = (episode if location == "episode" else masks) / "nested/trajectory.json"
    elif location == "existing":
        out = tmp_path / "derived/trajectory.json"
        write_json(out.with_name("raw_detections.json"), {"keep": True})
    else:
        out = tmp_path / "derived/detections.json"
    with pytest.raises(ValueError):
        build_tool.validate_mask_output(out, episode, masks, True)


def test_sam_cli_requires_explicit_output(build_tool, episode_and_masks, monkeypatch):
    episode, masks = episode_and_masks
    monkeypatch.setattr(sys, "argv", ["build_trajectory.py", "--episode", str(episode), "--mask-dir", str(masks)])
    with pytest.raises(SystemExit) as error:
        build_tool.main()
    assert error.value.code == 2


def test_sam_cli_keeps_diagnostics_but_no_task_for_stationary_clip(
    build_tool, episode_and_masks, calibration, spec, tmp_path, monkeypatch,
):
    episode, masks = episode_and_masks
    out = tmp_path / "derived/trajectory.json"
    monkeypatch.setattr(sys, "argv", ["build_trajectory.py", "--episode", str(episode), "--mask-dir", str(masks),
                                      "--out", str(out), "--no-video"])
    monkeypatch.setattr(build_tool, "calibration_from_config", lambda _: calibration)
    monkeypatch.setattr(build_tool, "load_object_specs", lambda _: [spec])
    with pytest.raises(SystemExit) as error:
        build_tool.main()
    assert error.value.code == 2 and not out.exists()
    quality = read_json(out.with_name("tracking_quality.json"))
    assert quality["endpoint_observations"]["passed"] and not quality["task_generated"]
    assert "No object moved" in quality["task_error"]
    assert read_json(out.with_name("raw_detections.json"))["metadata"]["training_ready"] is False


def test_sam_cli_does_not_export_unobserved_final_goal(
    build_tool, episode_and_masks, calibration, spec, tmp_path, monkeypatch,
):
    episode, masks = episode_and_masks
    cv2.imwrite(str(episode / "depth/000002.png"), np.zeros((120, 160), np.uint16))
    out = tmp_path / "derived/trajectory.json"
    monkeypatch.setattr(sys, "argv", ["build_trajectory.py", "--episode", str(episode), "--mask-dir", str(masks),
                                      "--out", str(out), "--no-video"])
    monkeypatch.setattr(build_tool, "calibration_from_config", lambda _: calibration)
    monkeypatch.setattr(build_tool, "load_object_specs", lambda _: [spec])
    with pytest.raises(SystemExit) as error:
        build_tool.main()
    assert error.value.code == 2 and not out.exists()
    quality = read_json(out.with_name("tracking_quality.json"))
    assert quality["missing_frames"] == [2] and not quality["task_generated"]
    assert "Start/end" in quality["task_error"]
