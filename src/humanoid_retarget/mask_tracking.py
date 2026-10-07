from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import list_rgb_frames, read_depth_m, read_episode_metadata, read_json
from humanoid_retarget.tracking import _largest_blob_centroid, _sphere_center
from humanoid_retarget.types import Detection, ObjectSpec


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_workspace(xy: np.ndarray, calibration: WorkspaceCalibration) -> np.ndarray:
    if calibration.workspace_outline_m is None:
        raise ValueError("Mask tracking requires a metric workspace outline")
    polygon = np.asarray(calibration.workspace_outline_m, dtype=np.float64)
    if not cv2.isContourConvex(polygon.astype(np.float32)):
        raise ValueError("Mask tracking requires a convex metric workspace")
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = np.asarray(xy).reshape(-1, 1, 2) - polygon
    cross = edges[:, 0] * relative[:, :, 1] - edges[:, 1] * relative[:, :, 0]
    tolerance = 0.003 * np.linalg.norm(edges, axis=1)
    return np.all(cross >= -tolerance, axis=1) | np.all(cross <= tolerance, axis=1)


def detection_from_depth_mask(
    mask: np.ndarray,
    depth_m: np.ndarray | None,
    spec: ObjectSpec,
    calibration: WorkspaceCalibration,
    frame_idx: int,
    t: float,
) -> tuple[Detection | None, dict[str, Any]]:
    """Estimate a centre from an instance mask, requiring measured 3D foreground."""
    if calibration.camera_to_desk_m is None:
        raise ValueError("Mask tracking requires calibrated camera intrinsics and camera-to-desk transform")
    if mask.shape != calibration.image_size_px[::-1]:
        raise ValueError("Mask dimensions must match the calibrated RGB image")
    if not np.isfinite(mask).all():
        raise ValueError("Mask must be finite")
    diagnostic: dict[str, Any] = {"frame_idx": frame_idx, "t": t, "mask_pixels": int(np.count_nonzero(mask))}
    if depth_m is None:
        return None, {**diagnostic, "status": "missing_depth"}
    if depth_m.shape != mask.shape:
        raise ValueError("Depth must be aligned to the RGB mask; resizing is not calibration")
    interior = cv2.erode((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    diagnostic["interior_pixels"] = int(interior.sum())
    valid = interior & np.isfinite(depth_m) & (depth_m > 0.05)
    py, px = np.nonzero(valid)
    diagnostic["valid_depth_pixels"] = len(px)
    if len(px) < 20:
        return None, {**diagnostic, "status": "insufficient_depth"}
    points = calibration.pixels_depth_to_desk_xyz(np.column_stack((px, py)), depth_m[py, px])
    # Table/shadow/marker pixels are removed in 3D, not by their image footprint.
    keep = (np.isfinite(points).all(axis=1)
            & (points[:, 2] > calibration.table_z_m + 0.004)
            & (points[:, 2] < calibration.table_z_m + 0.6)
            & _inside_workspace(points[:, :2], calibration))
    diagnostic["foreground_pixels"] = int(keep.sum())
    if keep.sum() < 20:
        return None, {**diagnostic, "status": "no_workspace_foreground"}
    foreground = np.zeros(mask.shape, np.uint8)
    foreground[py[keep], px[keep]] = 255
    blob = _largest_blob_centroid(foreground, spec.min_area_px)
    if blob is None:
        return None, {**diagnostic, "status": "foreground_too_small"}
    (cx, cy), area, contour = blob
    component = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(component, [contour], -1, 255, -1)
    selected = keep & (component[py, px] > 0)
    points = points[selected]
    if len(points) < 20:
        return None, {**diagnostic, "status": "insufficient_component_depth"}
    points = points[::max(1, len(points) // 1200)]
    span = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
    diagnostic["cloud_span_m"] = span.tolist()
    if np.linalg.norm(span) > 1.75 * np.linalg.norm(spec.size_m):
        return None, {**diagnostic, "status": "implausible_object_extent"}
    centre = _sphere_center(points, spec.size_m[2] / 2) if spec.shape == "sphere" else None
    nominal_z = calibration.table_z_m + spec.size_m[2] / 2
    if centre is not None:
        centre[2] = max(nominal_z, float(centre[2]))
        source = "sam2_aligned_depth_sphere_fit"
    else:
        centre_z = max(nominal_z, float(np.percentile(points[:, 2], 95)) - spec.size_m[2] / 2)
        # The visible surface median is not generally the object's XY centre.
        centre = calibration.pixel_to_desk_at_height((cx, cy), centre_z)
        source = "sam2_aligned_depth_height_ray"
    if (not np.isfinite(centre).all() or not _inside_workspace(centre[:2], calibration)[0]
            or centre[2] >= calibration.table_z_m + 0.6):
        return None, {**diagnostic, "status": "invalid_object_centre"}
    support = float(selected.sum() / max(1, interior.sum()))
    confidence = float(np.clip(area / max(1, spec.min_area_px * 8), 0, 1) * support)
    detection = Detection(frame_idx, t, spec.name, (cx, cy), tuple(float(v) for v in centre),
                          area, confidence, position_source=source)
    return detection, {**diagnostic, "status": "observed", "depth_support_fraction": support,
                       "detection": asdict(detection)}


def track_mask_episode(
    episode_dir: str | Path,
    specs: list[ObjectSpec],
    calibration: WorkspaceCalibration,
    mask_dir: str | Path,
    fps_override: float | None = None,
) -> tuple[list[Detection], dict[str, Any]]:
    episode, directory = Path(episode_dir), Path(mask_dir)
    metadata = read_episode_metadata(episode)
    if metadata.get("depth_aligned_to") != "color":
        raise ValueError("Mask tracking requires explicitly colour-aligned RGB-D")
    if calibration.camera_to_desk_m is None or calibration.workspace_outline_m is None:
        raise ValueError("Mask tracking requires full RGB-D calibration and a metric workspace")
    intrinsics = metadata.get("color_intrinsics")
    if intrinsics is not None:
        matrix = np.asarray([[intrinsics["fx"], 0, intrinsics["cx"]],
                             [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]])
        if (not np.allclose(matrix, calibration.camera_matrix_px, rtol=1e-5, atol=1e-4)
                or (intrinsics["width"], intrinsics["height"]) != calibration.image_size_px):
            raise ValueError("Capture intrinsics differ from the supplied RGB-D calibration")
    if "color_distortion" in metadata:
        distortion = np.asarray(metadata["color_distortion"])
        calibrated = np.asarray(calibration.camera_distortion)
        if distortion.shape != calibrated.shape or not np.allclose(distortion, calibrated):
            raise ValueError("Capture distortion differs from the supplied RGB-D calibration")
    depth_unit = float(metadata.get("depth_unit_m", 0.001))
    if not np.isfinite(depth_unit) or depth_unit <= 0:
        raise ValueError("Depth unit must be positive and finite")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != "hmnd_sam2_masks_v1" or manifest.get("completed") is not True:
        raise ValueError("Missing or incomplete SAM mask manifest")
    if Path(manifest["source_episode"]).resolve() != episode.resolve():
        raise ValueError("SAM masks belong to a different episode")
    if manifest["source_metadata_sha256"] != file_sha256(episode / "metadata.json"):
        raise ValueError("Capture metadata changed after SAM segmentation")
    spec = next((s for s in specs if s.name == manifest["object_name"]), None)
    if spec is None:
        raise ValueError("SAM mask object is not present in the supplied object configuration")
    frames = list_rgb_frames(episode)
    entries = manifest["frames"]
    if len(entries) != len(frames) or [e["frame_idx"] for e in entries] != list(range(len(frames))):
        raise ValueError("SAM masks must cover each original frame exactly once, in order")
    fps = float(metadata.get("fps", 30) if fps_override is None else fps_override)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Tracking FPS must be positive and finite")
    times = metadata.get("timestamps_relative_s")
    if times is not None:
        times = np.asarray(times, dtype=float)
        if (times.shape != (len(frames),) or not np.isfinite(times).all()
                or np.any(times < 0) or np.any(np.diff(times) <= 0)):
            raise ValueError("Capture timestamps must be finite, increasing and match the frame count")
    detections, diagnostics = [], []
    for index, (frame, entry) in enumerate(zip(frames, entries)):
        path = directory / f"{index:06d}.png"
        if frame.name != entry["rgb_name"] or file_sha256(frame) != entry["rgb_sha256"]:
            raise ValueError(f"RGB source fingerprint mismatch at frame {index}")
        if file_sha256(path) != entry["mask_sha256"]:
            raise ValueError(f"Mask fingerprint mismatch at frame {index}")
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
            raise ValueError(f"Frame {index} needs a full-resolution binary uint8 mask")
        depth = read_depth_m(episode, frame.stem, depth_unit)
        t = float(times[index]) if times is not None else index / fps
        detection, diagnostic = detection_from_depth_mask(mask, depth, spec, calibration, index, t)
        depth_path = episode / "depth" / f"{frame.stem}.png"
        if not depth_path.is_file():
            depth_path = depth_path.with_suffix(".npy")
        diagnostic["depth_sha256"] = file_sha256(depth_path) if depth_path.is_file() else None
        diagnostics.append(diagnostic)
        if detection is not None:
            detections.append(detection)
    return detections, {
        **metadata, "fps": fps, "num_rgb_frames": len(frames), "episode_dir": str(episode.resolve()),
        "action_object": spec.name, "tracking_backend": "sam2_rgbd",
        "mask_manifest": str((directory / "manifest.json").resolve()),
        "mask_manifest_sha256": file_sha256(directory / "manifest.json"),
        "frame_diagnostics": diagnostics, "training_ready": False,
        "scene_object_coverage": "selected_object_only",
        "raw_timing": "capture_timestamps" if times is not None else "frame_index_fps",
        "planning_timing": "Existing Kalman/planner use frame-index/effective-FPS timing, not exact capture timestamps.",
        "geometry_policy": {"min_surface_height_m": 0.004, "max_surface_height_m": 0.6,
                            "workspace_tolerance_m": 0.003, "nominal_depth_fallback": False},
        "label_notes": "Approximate position-only centres; no yaw or independent absolute metrology. Confidence is depth/area support, not SAM score or pose accuracy. Missing/predicted/weak observations are not ground truth. Other scene objects are not tracked.",
    }


def endpoint_observation_check(detections: list[Detection], frame_count: int, fps: float) -> dict[str, Any]:
    """Do not infer a pick/place task from an unobserved recording boundary."""
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Endpoint checks require positive frame count and FPS")
    width = min(frame_count, max(1, int(np.ceil(0.4 * fps))))
    edge_width = min(width, max(1, int(np.ceil(0.1 * fps))))
    observed = {d.frame_idx for d in detections if d.visible and d.confidence >= 0.6}
    windows = {"initial": set(range(width)), "final": set(range(frame_count - width, frame_count))}
    edges = {"initial": set(range(edge_width)), "final": set(range(frame_count - edge_width, frame_count))}
    result: dict[str, Any] = {"window_frames": width, "required_observations": int(np.ceil(0.75 * width)),
                              "min_confidence": 0.6, "boundary_tolerance_frames": edge_width}
    for name, frames in windows.items():
        result[name] = {"observed_frames": sorted(frames & observed),
                        "passed": (len(frames & observed) >= result["required_observations"]
                                   and bool(edges[name] & observed))}
    result["passed"] = result["initial"]["passed"] and result["final"]["passed"]
    return result


def save_mask_tracking_video(episode: Path, mask_dir: Path, detections: list[Detection],
                             diagnostics: list[dict[str, Any]], out: Path, fps: float,
                             preview_width: int | None = None) -> None:
    frames = list_rgb_frames(episode)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Could not decode {frames[0]}")
    height, width = first.shape[:2]
    if preview_width is not None and preview_width <= 0:
        raise ValueError("Preview width must be positive")
    scale = min(1.0, preview_width / width) if preview_width is not None else 1.0
    size = (max(2, 2 * int(width * scale / 2)), max(2, 2 * int(height * scale / 2)))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError(f"Could not open video writer for {out}")
    rows = {d.frame_idx: d for d in detections}
    try:
        for index, path in enumerate(frames):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not decode {path}")
            mask = cv2.imread(str(mask_dir / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f"Missing or misaligned mask at frame {index}")
            mask = mask > 0
            image[mask] = np.round(0.6 * image[mask] + 0.4 * np.asarray([70, 220, 80])).astype(np.uint8)
            d = rows.get(index)
            label = f"frame {index} | {diagnostics[index]['status']}"
            if d is not None:
                xyz = np.asarray(d.desk_xyz_m) * 1000
                label += f" | XYZ mm: {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f} | support {d.confidence:.2f}"
                if not d.visible:
                    label += " | PREDICTED, NOT OBSERVED"
                elif d.confidence < 0.6:
                    label += " | WEAK"
                cv2.drawMarker(image, tuple(np.round(d.pixel_xy).astype(int)), (0, 190, 255),
                               cv2.MARKER_CROSS, 20, 2)
            cv2.rectangle(image, (0, height - 36), (width, height), (20, 20, 20), -1)
            cv2.putText(image, label, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA) if size != (width, height) else image)
    finally:
        writer.release()
