from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import read_json
from humanoid_retarget.mask_tracking import file_sha256


OBJECT_NAMES = ("black_box", "blue_sphere", "yellow_mustard_tin")
GEOMETRY_NAMES = ("ray_u", "ray_v", "bbox_w_fx", "bbox_h_fy", "mask_fill",
                  "size_x", "size_y", "size_z", "is_box", "is_sphere", "is_tin",
                  "rectangle_w_fx", "rectangle_h_fy", "cos_2_image_axis", "sin_2_image_axis")


def recording_group(metadata: dict, episode: str) -> str:
    batch = metadata.get("recording_batch", {}).get("batch_id")
    return "batch:" + batch if batch else "single:" + str(Path(episode).resolve())


def validate_split_groups(clips: list[dict]) -> None:
    groups = {}
    for clip in clips:
        group, split = clip["recording_group"], clip["split"]
        if group in groups and groups[group] != split:
            raise ValueError("A recording group cannot cross training/validation/test splits")
        groups[group] = split
    if not {"train", "val"}.issubset({c["split"] for c in clips}):
        raise ValueError("Training requires separate train and validation clips")


def rgb_observation(image: np.ndarray, mask: np.ndarray, name: str, size_m,
                    calibration: WorkspaceCalibration, *, crop_size: int = 224) -> tuple[np.ndarray, dict]:
    """RGB/SAM-mask inputs only; never use depth, pose labels, time or clip ID as features."""
    if image is None or mask is None:
        raise ValueError("Missing RGB image or SAM mask")
    if image.shape[:2] != mask.shape or mask.shape != tuple(calibration.image_size_px[::-1]):
        raise ValueError("RGB/mask dimensions must match the saved calibration")
    if name not in OBJECT_NAMES or crop_size < 32:
        raise ValueError("Unsupported object or crop size")
    binary = (mask > 0).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 30:
        raise ValueError("Insufficient RGB mask support")
    x, y, width, height = cv2.boundingRect(points)
    moment = cv2.moments(binary)
    centre = [moment["m10"] / moment["m00"], moment["m01"] / moment["m00"]]
    matrix = np.asarray(calibration.camera_matrix_px)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    rectangle = cv2.minAreaRect(points)
    angle = np.deg2rad(rectangle[2])
    geometry = [(centre[0] - cx) / fx, (centre[1] - cy) / fy, width / fx, height / fy,
                moment["m00"] / (width * height), *(np.asarray(size_m) / 0.1),
                *(float(name == item) for item in OBJECT_NAMES),
                rectangle[1][0] / fx, rectangle[1][1] / fy, np.cos(2 * angle), np.sin(2 * angle)]
    side = max(width, height) * 1.6
    scale = crop_size / side
    affine = np.asarray([[scale, 0, crop_size / 2 - scale * (x + width / 2)],
                         [0, scale, crop_size / 2 - scale * (y + height / 2)]], dtype=float)
    crop = cv2.warpAffine(image, affine, (crop_size, crop_size), borderValue=(127, 127, 127))
    return crop, {"geometry": [float(v) for v in geometry], "pixel_xy": centre,
                  "bbox_wh_px": [width, height], "size_m": list(size_m), "object_name": name}


def projected_centre(xyz, calibration: WorkspaceCalibration) -> np.ndarray:
    transform = np.asarray(calibration.camera_to_desk_m)
    camera = transform[:3, :3].T @ (np.asarray(xyz) - transform[:3, 3])
    distortion = np.asarray(calibration.camera_distortion)
    pixels, _ = cv2.projectPoints(camera.reshape(1, 3), np.zeros(3), np.zeros(3),
                                  np.asarray(calibration.camera_matrix_px), distortion if distortion.size else None)
    return pixels.reshape(2)


def teacher_target(xyz, desk_yaw: float | None, observation: dict,
                   calibration: WorkspaceCalibration) -> list[float]:
    pixel = projected_centre(xyz, calibration)
    offset = (pixel - observation["pixel_xy"]) / observation["bbox_wh_px"]
    yaw = 0.0 if desk_yaw is None else float(desk_yaw)
    return [float((xyz[2] - calibration.table_z_m) / 0.1), *offset.tolist(),
            float(np.cos(2 * yaw)), float(np.sin(2 * yaw))]


def decode_pose(output, observation: dict, calibration: WorkspaceCalibration) -> dict:
    values = np.asarray(output, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Student output must contain five finite pose parameters")
    minimum = observation["size_m"][2] / 2
    height = float(np.clip(values[0] * 0.1, minimum, 0.3))
    pixel = np.asarray(observation["pixel_xy"]) + values[1:3] * observation["bbox_wh_px"]
    xyz = calibration.pixel_to_desk_at_height(pixel, calibration.table_z_m + height)
    yaw = float(np.arctan2(values[4], values[3]) / 2) if observation["object_name"] != "blue_sphere" else None
    return {"desk_xyz_m": xyz.tolist(), "desk_yaw_rad": yaw,
            "height_clamped": not np.isclose(height, values[0] * 0.1),
            "yaw_vector_norm": float(np.linalg.norm(values[3:])) if yaw is not None else None}


def load_inputs(path: Path, *, verify: bool = True) -> dict:
    payload = read_json(path)
    if payload.get("schema_version") != "hmnd_rgb_pose_inputs_v1":
        raise ValueError("Unsupported RGB input bundle")
    if payload.get("geometry_feature_names") != list(GEOMETRY_NAMES):
        raise ValueError("RGB geometry feature order changed")
    root = path.parent.resolve()
    for row in payload["samples"]:
        target = (root / row["rgb_crop"]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("Missing or escaped RGB crop")
        if verify and file_sha256(target) != row["rgb_crop_sha256"]:
            raise ValueError("RGB crop fingerprint changed")
        if len(row["geometry"]) != len(GEOMETRY_NAMES) or not np.isfinite(row["geometry"]).all():
            raise ValueError("Invalid RGB-derived input geometry")
    if len({r["sample_id"] for r in payload["samples"]}) != len(payload["samples"]):
        raise ValueError("Duplicate RGB input sample")
    validate_split_groups(payload["clips"])
    by_clip = {c["index"]: c for c in payload["clips"]}
    for row in payload["samples"]:
        if row["split"] != by_clip[row["clip_index"]]["split"]:
            raise ValueError("Sample split differs from its whole recording")
    return payload


def load_training(manifest_path: Path) -> tuple[dict, dict, dict]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "hmnd_rgb_student_training_v1":
        raise ValueError("Unsupported student training manifest")
    root = manifest_path.parent.resolve()
    paths = {}
    for key in ("inputs", "teacher_labels"):
        path = (root / manifest[key]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != manifest[key + "_sha256"]:
            raise ValueError("Training inputs/teacher labels missing or changed")
        paths[key] = path
    inputs, teacher = load_inputs(paths["inputs"]), read_json(paths["teacher_labels"])
    if teacher.get("schema_version") != "hmnd_depth_teacher_candidates_v1":
        raise ValueError("Unsupported teacher label schema")
    labels = {r["sample_id"]: r for r in teacher["samples"]}
    if len(labels) != len(teacher["samples"]) or set(labels) != {r["sample_id"] for r in inputs["samples"]}:
        raise ValueError("Teacher labels must match RGB sample IDs exactly")
    for row in labels.values():
        if len(row["target"]) != 5 or not np.isfinite(row["target"]).all():
            raise ValueError("Teacher target must contain five finite numbers")
        if len(row["desk_xyz_m"]) != 3 or not np.isfinite(row["desk_xyz_m"]).all():
            raise ValueError("Teacher XYZ must contain three finite numbers")
    return manifest, inputs, labels


def pose_metrics(predictions: list[dict], inputs: dict, labels: dict) -> dict:
    """Agreement with approximate teachers, not independent physical accuracy."""
    by_id = {r["sample_id"]: r for r in inputs["samples"]}
    if len({p["sample_id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction sample")
    results = []
    for prediction in predictions:
        identifier = prediction["sample_id"]
        sample, teacher = by_id[identifier], labels[identifier]
        error = np.asarray(prediction["desk_xyz_m"]) - teacher["desk_xyz_m"]
        yaw_error = None
        if teacher["yaw_observed"] and prediction["desk_yaw_rad"] is not None:
            delta = prediction["desk_yaw_rad"] - teacher["desk_yaw_rad"]
            yaw_error = float(np.rad2deg(abs(np.arctan2(np.sin(2 * delta), np.cos(2 * delta))) / 2))
        results.append({"object": sample["object_name"], "phase": teacher["phase"],
                        "xyz_mm": float(np.linalg.norm(error) * 1000),
                        "xy_mm": float(np.linalg.norm(error[:2]) * 1000),
                        "z_mm": float(abs(error[2]) * 1000), "yaw_deg": yaw_error,
                        "height_clamped": prediction.get("height_clamped", False)})

    def summarise(rows):
        output = {"samples": len(rows)}
        for key in ("xyz_mm", "xy_mm", "z_mm", "yaw_deg"):
            values = [r[key] for r in rows if r[key] is not None]
            output[key] = ({"count": len(values), "mean": float(np.mean(values)),
                            "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
                           if values else None)
        output["height_clamped_fraction"] = float(np.mean([r["height_clamped"] for r in rows])) if rows else None
        return output

    return {"overall": summarise(results),
            "per_object": {n: summarise([r for r in results if r["object"] == n]) for n in OBJECT_NAMES},
            "per_phase": {n: summarise([r for r in results if r["phase"] == n]) for n in ("initial", "carry", "final")}}


def calibration_snapshot(calibration: WorkspaceCalibration) -> dict:
    return asdict(calibration)
