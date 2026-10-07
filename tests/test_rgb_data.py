from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.rgb_data import (GEOMETRY_NAMES, decode_pose, load_inputs, load_training,
                                       pose_metrics, recording_group, rgb_observation,
                                       teacher_target, validate_split_groups)


def calibration():
    return WorkspaceCalibration(
        image_size_px=(200, 160), image_points_px=((0, 0), (199, 0), (199, 159), (0, 159)),
        desk_points_m=((-0.2, 0.16), (0.2, 0.16), (0.2, -0.16), (-0.2, -0.16)),
        camera_matrix_px=((400., 0., 100.), (0., 400., 80.), (0., 0., 1.)),
        camera_to_desk_m=((1., 0., 0., 0.), (0., -1., 0., 0.), (0., 0., -1., 0.8), (0., 0., 0., 1.)),
    )


def observation(name="black_box"):
    image = np.full((160, 200, 3), 127, dtype=np.uint8)
    mask = np.zeros((160, 200), dtype=np.uint8)
    mask[50:70, 70:100] = 255
    return rgb_observation(image, mask, name, [0.05555, 0.045, 0.031], calibration())


def test_nested_recording_batch_controls_split():
    assert recording_group({"recording_batch": {"batch_id": "abc"}}, "unused") == "batch:abc"
    clips = [{"recording_group": "batch:abc", "split": "train"},
             {"recording_group": "batch:abc", "split": "val"}]
    with pytest.raises(ValueError, match="cannot cross"):
        validate_split_groups(clips)
    with pytest.raises(ValueError, match="separate"):
        validate_split_groups(clips[:1])


def test_geometry_is_rgb_only_and_keeps_global_pixel_location():
    crop, row = observation()
    assert crop.shape == (224, 224, 3)
    assert len(row["geometry"]) == len(GEOMETRY_NAMES)
    assert row["pixel_xy"] == [84.5, 59.5]
    assert not {"depth", "target", "phase", "clip_index", "desk_xyz_m"}.intersection(row)


@pytest.mark.parametrize("yaw", [-1.5, 0.0, 1.4])
def test_pose_target_decode_roundtrip_without_depth(yaw, monkeypatch):
    _, row = observation()
    xyz = [0.02, -0.03, 0.1]
    monkeypatch.setattr(WorkspaceCalibration, "pixels_depth_to_desk_xyz",
                        lambda *args: pytest.fail("RGB inference must not use depth"))
    recovered = decode_pose(teacher_target(xyz, yaw, row, calibration()), row, calibration())
    assert np.allclose(recovered["desk_xyz_m"], xyz, atol=1e-7)
    assert recovered["desk_yaw_rad"] == pytest.approx(yaw)
    assert not recovered["height_clamped"]


def test_sphere_yaw_and_physical_height_bounds():
    _, row = observation("blue_sphere")
    pose = decode_pose([-1, 0, 0, 0, 0], row, calibration())
    assert pose["desk_yaw_rad"] is None
    assert pose["desk_xyz_m"][2] == pytest.approx(0.0155)
    assert pose["height_clamped"]
    with pytest.raises(ValueError, match="finite"):
        decode_pose([np.nan, 0, 0, 1, 0], row, calibration())


def bundle(tmp_path):
    crop, row = observation()
    crop_path = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(crop_path), crop)
    samples = [{"sample_id": str(i), "clip_index": i, "split": split,
                "rgb_crop": "crop.jpg", "rgb_crop_sha256": file_sha256(crop_path), **row}
               for i, split in enumerate(("train", "val"))]
    payload = {"schema_version": "hmnd_rgb_pose_inputs_v1", "samples": samples,
               "geometry_feature_names": list(GEOMETRY_NAMES),
               "clips": [{"index": i, "recording_group": str(i), "split": split}
                         for i, split in enumerate(("train", "val"))]}
    path = tmp_path / "inputs.json"
    write_json(path, payload)
    return path, payload


def test_prediction_inputs_open_only_rgb_and_input_json(tmp_path, monkeypatch):
    path, _ = bundle(tmp_path)
    original = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self.name in ("inputs.json", "crop.jpg")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert len(load_inputs(path)["samples"]) == 2


@pytest.mark.parametrize("change,error", [("hash", "fingerprint"), ("escape", "escaped"),
                                          ("duplicate", "Duplicate"), ("split", "differs"),
                                          ("features", "order")])
def test_bundle_tampering_rejected(tmp_path, change, error):
    path, payload = bundle(tmp_path)
    if change == "hash":
        payload["samples"][0]["rgb_crop_sha256"] = "wrong"
    elif change == "escape":
        payload["samples"][0]["rgb_crop"] = "../crop.jpg"
    elif change == "duplicate":
        payload["samples"][1]["sample_id"] = "0"
    elif change == "split":
        payload["samples"][0]["split"] = "val"
    else:
        payload["geometry_feature_names"].reverse()
    write_json(path, payload)
    with pytest.raises(ValueError, match=error):
        load_inputs(path)


def test_teacher_and_prediction_metrics_handle_axis_symmetry(tmp_path):
    _, inputs = bundle(tmp_path)
    labels = {str(i): {"desk_xyz_m": [0, 0, 0.1], "desk_yaw_rad": 0.2,
                       "yaw_observed": True, "phase": "carry"} for i in range(2)}
    predicted = [{"sample_id": str(i), "desk_xyz_m": [0.003, 0.004, 0.1], "desk_yaw_rad": 0.2 + np.pi}
                 for i in range(2)]
    metric = pose_metrics(predicted, inputs, labels)["overall"]
    assert metric["xyz_mm"]["median"] == pytest.approx(5)
    assert metric["yaw_deg"]["median"] == pytest.approx(0, abs=1e-12)


def test_teacher_export_cannot_change_after_freezing(tmp_path):
    path, _ = bundle(tmp_path)
    teacher = tmp_path / "teacher.json"
    write_json(teacher, {"schema_version": "hmnd_depth_teacher_candidates_v1", "samples": []})
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"schema_version": "hmnd_rgb_student_training_v1", "inputs": path.name,
                          "inputs_sha256": file_sha256(path), "teacher_labels": teacher.name,
                          "teacher_labels_sha256": "changed"})
    with pytest.raises(ValueError, match="changed"):
        load_training(manifest)


def test_missing_rgb_fails_cleanly():
    with pytest.raises(ValueError, match="Missing"):
        rgb_observation(None, None, "black_box", [1, 1, 1], calibration())
