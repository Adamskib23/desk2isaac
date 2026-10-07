#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from humanoid_retarget.calibration import (
    calibration_from_depth_markers, calibration_from_single_marker, find_aruco_marker, find_aruco_markers,
)
from humanoid_retarget.io import list_rgb_frames, read_depth_m, read_episode_metadata, read_yaml
from humanoid_retarget.visualization import rectify_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a desk from flat, measured ArUco markers and optional RGB-D.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="RGB calibration image.")
    source.add_argument("--episode", type=Path, help="Stationary calibration recording; searches its first 60 frames.")
    parser.add_argument("--marker-size-mm", required=True, type=float, help="Outer black square side, excluding white margin.")
    parser.add_argument("--marker-id", type=int)
    parser.add_argument("--marker-ids", type=int, nargs="+", help="Use all these IDs for 3D RGB-D plane calibration; --marker-id selects the origin (default 0).")
    parser.add_argument("--dictionary", default="auto", help="OpenCV dictionary name, or auto.")
    parser.add_argument("--workspace-roi-px", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--workspace-shape", choices=("auto", "rectangle", "hull"), default="auto",
                        help="Four markers default to a metric rectangle; other layouts use their hull.")
    parser.add_argument("--workspace-inset-mm", type=float, default=5.0,
                        help="Minimum clearance from the four-marker boundary (default 5 mm).")
    parser.add_argument("--base-config", type=Path, default=Path("configs/workspace.yaml"))
    parser.add_argument("--out", type=Path, default=Path("configs/workspace.real.yaml"))
    args = parser.parse_args()
    if args.marker_ids and not args.episode:
        parser.error("--marker-ids requires an aligned RGB-D --episode with saved camera parameters")
    rectangle_requested = args.workspace_shape == "rectangle" or (
        args.workspace_shape == "auto" and args.marker_ids and len(args.marker_ids) == 4
    )
    if rectangle_requested and (not args.marker_ids or len(args.marker_ids) != 4 or args.workspace_roi_px):
        parser.error("Rectangular workspace needs exactly four --marker-ids and no --workspace-roi-px")

    paths = list_rgb_frames(args.episode)[:60] if args.episode else [args.image]
    metadata = read_episode_metadata(args.episode) if args.episode else {}
    marker = None
    markers = []
    last_error = "No readable calibration images"
    for image_path in paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        try:
            if args.marker_ids:
                markers = [m for m in find_aruco_markers(image, args.dictionary) if m.marker_id in args.marker_ids]
                found_ids = [m.marker_id for m in markers]
                if sorted(found_ids) != sorted(args.marker_ids):
                    raise ValueError(f"Expected marker IDs {args.marker_ids}; found {found_ids}")
                anchor_id = args.marker_id if args.marker_id is not None else 0
                marker = next((m for m in markers if m.marker_id == anchor_id), None)
                if marker is None:
                    raise ValueError(f"Anchor ID {anchor_id} must be included in --marker-ids")
            else:
                marker = find_aruco_marker(image, args.marker_id, args.dictionary)
                markers = [marker]
            break
        except ValueError as exc:
            last_error = str(exc)
    if marker is None:
        raise ValueError(last_error)

    height, width = image.shape[:2]
    roi = tuple(args.workspace_roi_px) if args.workspace_roi_px else None
    diagnostics = None
    if args.marker_ids:
        intrinsic = metadata.get("color_intrinsics")
        if (metadata.get("depth_aligned_to") != "color" or not intrinsic
                or metadata.get("color_distortion") is None):
            raise ValueError("3D calibration needs aligned depth, intrinsics and lens distortion; record a fresh episode")
        if (intrinsic["width"], intrinsic["height"]) != (width, height):
            raise ValueError("Saved intrinsics do not match the RGB resolution")
        matrix = np.asarray([[intrinsic["fx"], 0, intrinsic["cx"]],
                             [0, intrinsic["fy"], intrinsic["cy"]], [0, 0, 1]], dtype=float)
        depth = read_depth_m(args.episode, image_path.stem, metadata.get("depth_unit_m", 0.001))
        if depth is None:
            raise ValueError("Calibration RGB image has no matching aligned depth frame")
        calibration, diagnostics = calibration_from_depth_markers(
            (width, height), markers, args.marker_size_mm / 1000, depth, matrix,
            tuple(metadata["color_distortion"]), marker.marker_id, roi,
            args.workspace_shape, args.workspace_inset_mm / 1000,
        )
    else:
        calibration = calibration_from_single_marker(
            (width, height), marker.image_points_px, args.marker_size_mm / 1000, roi,
        )
    config = read_yaml(args.base_config)
    workspace = asdict(calibration)
    table_depth = None
    if args.episode and metadata.get("depth_aligned_to") == "color":
        depth = read_depth_m(args.episode, image_path.stem, metadata.get("depth_unit_m", 0.001))
        if depth is not None and depth.shape == image.shape[:2]:
            mask = np.zeros(depth.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(marker.image_points_px).astype(np.int32), 255)
            mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8))
            valid = depth[(mask > 0) & np.isfinite(depth) & (depth > 0.05)]
            if valid.size >= 20:
                table_depth = float(np.median(valid))
    workspace["table_depth_m"] = table_depth
    workspace["calibration_reference"] = {
        "type": "multi_aruco_rgbd_desk" if diagnostics else "single_aruco_desk_plane",
        "marker_id": marker.marker_id,
        "marker_ids": [m.marker_id for m in markers],
        "dictionary": marker.dictionary,
        "marker_size_m": args.marker_size_mm / 1000,
        "source_image": str(image_path),
        "workspace_roi_px": args.workspace_roi_px,
        "origin": "marker centre; x to decoded right, y to decoded top, z above desk",
        "assumptions": ("Markers are flat on desk; their relative layout is estimated from RGB-D."
                        if diagnostics else "Marker is flat on desk; homography applies to the desk plane only."),
        "diagnostics": diagnostics,
    }
    config["workspace"] = workspace
    if diagnostics:
        desk_center = np.asarray(calibration.workspace_outline_m).mean(axis=0)
        config.setdefault("retarget", {})["desk_center_m"] = desk_center.tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    preview = image.copy()
    outline_px = np.asarray([
        calibration.desk_to_pixel_xy(p) for p in calibration.workspace_outline_m
    ])
    cv2.polylines(preview, [np.round(outline_px).astype(np.int32)], True, (0, 210, 210), 2)
    corners = [np.asarray(m.image_points_px, dtype=np.float32).reshape(1, 4, 2) for m in markers]
    cv2.aruco.drawDetectedMarkers(preview, corners, np.asarray([[m.marker_id] for m in markers], dtype=np.int32))
    center = tuple(np.round(calibration.desk_to_pixel_xy((0, 0))).astype(int))
    for endpoint, color, label in [((0.10, 0), (0, 0, 255), "+x 100 mm"),
                                   ((0, 0.10), (0, 180, 0), "+y 100 mm")]:
        pixel = tuple(np.round(calibration.desk_to_pixel_xy(endpoint)).astype(int))
        cv2.arrowedLine(preview, center, pixel, color, 3)
        cv2.putText(preview, label, pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    preview_path = args.out.with_suffix(".png")
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"Could not write {preview_path}")
    print(f"Detected {marker.dictionary}, IDs {[m.marker_id for m in markers]}; black squares {args.marker_size_mm:.1f} mm")
    print(f"Saved {args.out} and {preview_path}")
    if diagnostics and diagnostics["workspace_shape"] == "rectangle":
        topdown = rectify_workspace(image, calibration)
        cv2.rectangle(topdown, (1, 1), (topdown.shape[1] - 2, topdown.shape[0] - 2), (0, 210, 210), 3)
        topdown_path = args.out.with_name(args.out.stem + "_topdown.png")
        if not cv2.imwrite(str(topdown_path), topdown):
            raise OSError(f"Could not write {topdown_path}")
        dimensions_mm = sorted(v * 1000 for v in diagnostics["workspace_size_m"])
        print(f"Rectangular workspace: {dimensions_mm[1]:.0f} x {dimensions_mm[0]:.0f} mm; top-down preview: {topdown_path}")
    print(f"Desk-plane bounds: {calibration.desk_limits_m}")
    print(f"Desk depth at marker: {table_depth:.3f} m" if table_depth else "No aligned desk depth; height uses configured object dimensions.")
    if diagnostics:
        print(f"Plane RMS: {diagnostics['plane_rms_m'] * 1000:.2f} mm; max marker-side discrepancy: {diagnostics['max_marker_size_error_fraction'] * 100:.1f}%")
    print("Check the preview and a second measured distance away from the marker before trusting metric positions.")


if __name__ == "__main__":
    main()
