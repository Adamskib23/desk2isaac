from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class WorkspaceCalibration:
    """Maps image pixels to a metric desk/workspace plane."""

    image_size_px: tuple[int, int]
    image_points_px: tuple[tuple[float, float], ...]
    desk_points_m: tuple[tuple[float, float], ...]
    table_z_m: float = 0.0
    table_depth_m: float | None = None
    workspace_outline_m: tuple[tuple[float, float], ...] | None = None
    camera_matrix_px: tuple[tuple[float, ...], ...] | None = None
    camera_distortion: tuple[float, ...] = ()
    camera_to_desk_m: tuple[tuple[float, ...], ...] | None = None
    workspace_roi_px: tuple[float, float, float, float] | None = None
    excluded_polygons_px: tuple[tuple[tuple[float, float], ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.image_points_px) < 4 or len(self.image_points_px) != len(self.desk_points_m):
            raise ValueError("Workspace calibration expects at least four matched image and desk points")
        if (self.camera_matrix_px is None) != (self.camera_to_desk_m is None):
            raise ValueError("3D calibration requires both a camera matrix and camera-to-desk transform")

    @property
    def pixel_to_desk_h(self) -> np.ndarray:
        h, _ = cv2.findHomography(
            np.asarray(self.image_points_px, dtype=np.float32),
            np.asarray(self.desk_points_m, dtype=np.float32),
            method=0,
        )
        if h is None:
            raise ValueError("Could not compute pixel-to-desk homography")
        return h

    @property
    def desk_to_pixel_h(self) -> np.ndarray:
        h, _ = cv2.findHomography(
            np.asarray(self.desk_points_m, dtype=np.float32),
            np.asarray(self.image_points_px, dtype=np.float32),
            method=0,
        )
        if h is None:
            raise ValueError("Could not compute desk-to-pixel homography")
        return h

    def pixel_to_desk_xy(self, pixel_xy: tuple[float, float] | np.ndarray) -> np.ndarray:
        if self.camera_to_desk_m is not None:
            return self.pixel_to_desk_at_height(pixel_xy, self.table_z_m)[:2].astype(np.float32)
        pt = np.asarray([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.pixel_to_desk_h)[0, 0]
        return mapped.astype(np.float32)

    def desk_to_pixel_xy(self, desk_xy_m: tuple[float, float] | np.ndarray) -> np.ndarray:
        if self.camera_to_desk_m is not None:
            transform = np.asarray(self.camera_to_desk_m)
            desk = np.asarray([desk_xy_m[0], desk_xy_m[1], self.table_z_m])
            camera = transform[:3, :3].T @ (desk - transform[:3, 3])
            distortion = np.asarray(self.camera_distortion)
            projected, _ = cv2.projectPoints(
                camera.reshape(1, 3), np.zeros(3), np.zeros(3),
                np.asarray(self.camera_matrix_px), distortion if distortion.size else None,
            )
            return projected[0, 0].astype(np.float32)
        pt = np.asarray([[[float(desk_xy_m[0]), float(desk_xy_m[1])]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.desk_to_pixel_h)[0, 0]
        return mapped.astype(np.float32)

    def pixels_depth_to_desk_xyz(self, pixels_xy: np.ndarray, depths_m: np.ndarray) -> np.ndarray:
        if self.camera_to_desk_m is None:
            raise ValueError("Depth deprojection requires a calibrated camera-to-desk transform")
        rays = camera_rays(pixels_xy, np.asarray(self.camera_matrix_px), self.camera_distortion)
        camera_points = rays * np.asarray(depths_m).reshape(-1, 1)
        transform = np.asarray(self.camera_to_desk_m)
        return camera_points @ transform[:3, :3].T + transform[:3, 3]

    def pixel_to_desk_at_height(self, pixel_xy: tuple[float, float], height_m: float) -> np.ndarray:
        if self.camera_to_desk_m is None:
            xy = self.pixel_to_desk_xy(pixel_xy)
            return np.asarray([*xy, height_m])
        ray = camera_rays(np.asarray([pixel_xy]), np.asarray(self.camera_matrix_px), self.camera_distortion)[0]
        transform = np.asarray(self.camera_to_desk_m)
        direction = transform[:3, :3] @ ray
        origin = transform[:3, 3]
        if abs(direction[2]) < 1e-8:
            raise ValueError("Pixel ray is parallel to the desk plane")
        distance = (height_m - origin[2]) / direction[2]
        if distance <= 0:
            raise ValueError("Desk-plane intersection lies behind the camera")
        return origin + distance * direction

    @property
    def desk_limits_m(self) -> dict[str, list[float]]:
        pts = np.asarray(
            self.workspace_outline_m if self.workspace_outline_m is not None else self.desk_points_m,
            dtype=np.float32,
        )
        return {
            "x": [float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))],
            "y": [float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))],
            "z": [float(self.table_z_m), float(self.table_z_m + 0.6)],
        }


def calibration_from_config(config: dict[str, Any]) -> WorkspaceCalibration:
    workspace = config.get("workspace", config)
    image_size = tuple(workspace.get("image_size_px", (1280, 720)))
    image_points = tuple(tuple(p) for p in workspace["image_points_px"])
    desk_points = tuple(tuple(p) for p in workspace["desk_points_m"])
    return WorkspaceCalibration(
        image_size_px=(int(image_size[0]), int(image_size[1])),
        image_points_px=image_points,
        desk_points_m=desk_points,
        table_z_m=float(workspace.get("table_z_m", 0.0)),
        table_depth_m=(
            None if workspace.get("table_depth_m") is None else float(workspace.get("table_depth_m"))
        ),
        workspace_outline_m=(
            None if workspace.get("workspace_outline_m") is None
            else tuple(tuple(p) for p in workspace["workspace_outline_m"])
        ),
        camera_matrix_px=(None if workspace.get("camera_matrix_px") is None
                          else tuple(tuple(row) for row in workspace["camera_matrix_px"])),
        camera_distortion=tuple(workspace.get("camera_distortion", ())),
        camera_to_desk_m=(None if workspace.get("camera_to_desk_m") is None
                          else tuple(tuple(row) for row in workspace["camera_to_desk_m"])),
        workspace_roi_px=(None if workspace.get("workspace_roi_px") is None
                          else tuple(workspace["workspace_roi_px"])),
        excluded_polygons_px=tuple(tuple(tuple(p) for p in polygon)
                                  for polygon in workspace.get("excluded_polygons_px", ())),
    )


def _detect_aruco(image_bgr: np.ndarray, dictionary_id: int):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("ArUco requires opencv-contrib-python in your capture environment")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


@dataclass(frozen=True)
class ArucoMarker:
    marker_id: int
    dictionary: str
    image_points_px: tuple[tuple[float, float], ...]


def find_aruco_marker(
    image_bgr: np.ndarray,
    marker_id: int | None = None,
    dictionary: str = "auto",
) -> ArucoMarker:
    """Find one marker, retaining its decoded corner order even when rotated."""
    matches = find_aruco_markers(image_bgr, dictionary)
    if marker_id is not None:
        matches = [marker for marker in matches if marker.marker_id == marker_id]
    if not matches:
        raise ValueError(
            "No matching ArUco marker found. Keep the complete black border and white margin "
            "visible; supply --dictionary if you used a custom marker family."
        )
    if len(matches) != 1:
        found = ", ".join(f"{m.dictionary}:ID {m.marker_id}" for m in matches)
        raise ValueError(f"Ambiguous marker detection ({found}); specify --dictionary and --marker-id")
    return matches[0]


def find_aruco_markers(image_bgr: np.ndarray, dictionary: str = "auto") -> list[ArucoMarker]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("ArUco requires opencv-contrib-python in your capture environment")
    names = (
        ["DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000", "DICT_7X7_1000",
         "DICT_ARUCO_ORIGINAL", "DICT_ARUCO_MIP_36h12"]
        if dictionary == "auto" else [dictionary]
    )
    matches: list[ArucoMarker] = []
    for name in names:
        if not hasattr(cv2.aruco, name):
            if dictionary != "auto":
                raise ValueError(f"Unknown ArUco dictionary: {name}")
            continue
        corners, ids, _ = _detect_aruco(image_bgr, getattr(cv2.aruco, name))
        if ids is None:
            continue
        for points, detected_id in zip(corners, ids.flatten()):
            matches.append(ArucoMarker(
                marker_id=int(detected_id),
                dictionary=name,
                image_points_px=tuple(tuple(float(v) for v in p) for p in points.reshape(4, 2)),
            ))
    return matches


def camera_rays(pixels_xy: np.ndarray, matrix: np.ndarray, distortion=()) -> np.ndarray:
    coeffs = np.asarray(distortion, dtype=np.float64)
    normalized = cv2.undistortPoints(
        np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(matrix, dtype=np.float64), coeffs if coeffs.size else None,
    ).reshape(-1, 2)
    return np.column_stack((normalized, np.ones(len(normalized))))


def rectangular_workspace_from_markers(
    marker_corners_m: np.ndarray,
    inset_m: float = 0.005,
) -> tuple[tuple[float, float], ...]:
    """Fit a conservative metric rectangle inside four corner markers' hull."""
    corners = np.asarray(marker_corners_m, dtype=np.float64)
    if corners.shape != (4, 4, 2) or not np.isfinite(corners).all():
        raise ValueError("Rectangular workspace requires four markers with finite desk-plane corners")
    if not np.isfinite(inset_m) or inset_m < 0:
        raise ValueError("Workspace inset must be finite and non-negative")
    centres = corners.mean(axis=1).astype(np.float32)
    if len(cv2.convexHull(centres)) != 4:
        raise ValueError("Place one marker at each of four distinct workspace corners")
    hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.float32)).reshape(-1, 2)
    rectangle = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
    centre = rectangle.mean(axis=0)
    offsets = rectangle - centre
    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.column_stack((edges[:, 1], -edges[:, 0])).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    distances = np.sum(normals * (hull - centre), axis=1)
    # Uniformly shrink the enclosing rectangle until every corner satisfies
    # every hull half-plane; a bounding box alone would extend outside markers.
    support = (normals @ offsets.T).max(axis=1)
    scale = min(1.0, float(np.min((distances - inset_m) / support)))
    if scale <= 0 or np.min(np.linalg.norm(edges, axis=1)) < 1e-6:
        raise ValueError("Markers enclose no usable rectangle at the requested inset")
    outline = centre + scale * offsets
    if np.min(np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1)) < 0.01:
        raise ValueError("Markers enclose too small a rectangular workspace")
    return tuple(tuple(float(v) for v in p) for p in outline)


def calibration_from_depth_markers(
    image_size_px: tuple[int, int],
    markers: list[ArucoMarker],
    marker_size_m: float,
    depth_m: np.ndarray,
    camera_matrix_px: np.ndarray,
    camera_distortion: tuple[float, ...] = (),
    anchor_id: int = 0,
    workspace_roi_px: tuple[float, float, float, float] | None = None,
    workspace_shape: str = "auto",
    workspace_inset_m: float = 0.005,
) -> tuple[WorkspaceCalibration, dict[str, Any]]:
    """Fit a common desk plane without assuming the markers form a measured board.

    Marker-to-marker placement comes from RGB-D. Their known side lengths provide
    an independent scale check, not invented board coordinates.
    """
    if not np.isfinite(marker_size_m) or marker_size_m <= 0:
        raise ValueError("Marker side length must be positive and finite")
    width, height = image_size_px
    if depth_m.shape != (height, width):
        raise ValueError("Calibration depth must be aligned to the colour image")
    ids = [m.marker_id for m in markers]
    if anchor_id not in ids or len(ids) != len(set(ids)):
        raise ValueError("Need one unambiguous anchor marker and unique marker IDs")
    if workspace_shape not in ("auto", "rectangle", "hull"):
        raise ValueError("Workspace shape must be auto, rectangle or hull")
    shape = "rectangle" if workspace_shape == "auto" and len(markers) == 4 else workspace_shape
    if shape == "auto":
        shape = "hull"
    if shape == "rectangle" and (len(markers) != 4 or workspace_roi_px is not None):
        raise ValueError("Rectangular workspace requires exactly four markers and no image ROI")
    anchor = markers[ids.index(anchor_id)]
    calibration_from_single_marker(image_size_px, anchor.image_points_px, marker_size_m, workspace_roi_px)
    clouds = []
    exclusions = []
    for marker in markers:
        corners = np.asarray(marker.image_points_px)
        centre = corners.mean(axis=0)
        mask = np.zeros(depth_m.shape, dtype=np.uint8)
        inset = centre + 0.8 * (corners - centre)
        cv2.fillConvexPoly(mask, np.round(inset).astype(np.int32), 255)
        valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 3)
        py, px = np.nonzero(valid)
        if len(px) < 40:
            raise ValueError(f"Marker ID {marker.marker_id} has too few valid aligned depth pixels")
        stride = max(1, len(px) // 1200)
        px, py = px[::stride], py[::stride]
        rays = camera_rays(np.column_stack((px, py)), camera_matrix_px, camera_distortion)
        clouds.append(rays * depth_m[py, px, None])
        exclusions.append(tuple(tuple(float(v) for v in p) for p in centre + 1.18 * (corners - centre)))
    points = np.vstack(clouds)
    inliers = np.ones(len(points), dtype=bool)
    for _ in range(5):
        centre = points[inliers].mean(axis=0)
        _, _, axes = np.linalg.svd(points[inliers] - centre, full_matrices=False)
        normal = axes[-1]
        residuals = np.abs((points - centre) @ normal)
        median = np.median(residuals[inliers])
        mad = np.median(np.abs(residuals[inliers] - median))
        next_inliers = residuals <= max(0.003, median + 3 * 1.4826 * mad)
        if next_inliers.sum() < len(points) * 0.6:
            raise ValueError("Marker depth samples do not support a consistent tabletop plane")
        if np.array_equal(inliers, next_inliers):
            break
        inliers = next_inliers
    centre = points[inliers].mean(axis=0)
    _, _, axes = np.linalg.svd(points[inliers] - centre, full_matrices=False)
    normal = axes[-1]
    if normal @ centre > 0:
        normal = -normal
    offset = -normal @ centre

    def corners_on_plane(marker: ArucoMarker) -> np.ndarray:
        rays = camera_rays(np.asarray(marker.image_points_px), camera_matrix_px, camera_distortion)
        return rays * (-offset / (rays @ normal))[:, None]

    anchor_corners = corners_on_plane(anchor)
    origin = anchor_corners.mean(axis=0)
    x_axis = (anchor_corners[1] - anchor_corners[0]) + (anchor_corners[2] - anchor_corners[3])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    rotation = np.vstack((x_axis, y_axis, normal))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin
    image_points, desk_points = [], []
    lengths, marker_centres, marker_residuals = {}, {}, {}
    for marker, cloud in zip(markers, clouds):
        camera_corners = corners_on_plane(marker)
        desk_corners = (camera_corners - origin) @ rotation.T
        edges = np.linalg.norm(np.roll(camera_corners, -1, axis=0) - camera_corners, axis=1)
        lengths[marker.marker_id] = edges.tolist()
        marker_centres[marker.marker_id] = desk_corners.mean(axis=0).tolist()
        marker_residuals[marker.marker_id] = float(np.median(np.abs(cloud @ normal + offset)))
        image_points.extend(marker.image_points_px)
        desk_points.extend(tuple(float(v) for v in p[:2]) for p in desk_corners)
    max_size_error = max(abs(edge / marker_size_m - 1) for edges in lengths.values() for edge in edges)
    if max_size_error > 0.12 or max(marker_residuals.values()) > 0.010:
        raise ValueError("Marker geometry disagrees with measured size or desk plane; check print size, depth and intrinsics")
    calibration = WorkspaceCalibration(
        image_size_px, tuple(image_points), tuple(desk_points),
        camera_matrix_px=tuple(tuple(float(v) for v in row) for row in camera_matrix_px),
        camera_distortion=camera_distortion,
        camera_to_desk_m=tuple(tuple(float(v) for v in row) for row in transform),
        workspace_roi_px=workspace_roi_px,
        excluded_polygons_px=tuple(exclusions),
    )
    if shape == "rectangle":
        outline = rectangular_workspace_from_markers(np.asarray(desk_points).reshape(4, 4, 2), workspace_inset_m)
    else:
        hull = cv2.convexHull(np.asarray(image_points, dtype=np.float32)).reshape(-1, 2)
        if workspace_roi_px is not None:
            x, y, roi_width, roi_height = workspace_roi_px
            rectangle = np.asarray([[x, y], [x + roi_width, y],
                                    [x + roi_width, y + roi_height], [x, y + roi_height]], dtype=np.float32)
            area, intersection = cv2.intersectConvexConvex(hull, rectangle)
            if intersection is None or area < 16:
                raise ValueError("Workspace ROI does not overlap the area enclosed by the markers")
            hull = intersection.reshape(-1, 2)
        outline = tuple(tuple(float(v) for v in calibration.pixel_to_desk_xy(p)) for p in hull)
    calibration = replace(calibration, workspace_outline_m=outline)
    diagnostics = {
        "method": "rgbd_marker_plane",
        "marker_ids": ids,
        "plane_rms_m": float(np.sqrt(np.mean(((points[inliers] - centre) @ normal) ** 2))),
        "depth_inlier_fraction": float(inliers.mean()),
        "marker_edge_lengths_m": lengths,
        "marker_centres_desk_m": marker_centres,
        "marker_plane_residual_m": marker_residuals,
        "max_marker_size_error_fraction": float(max_size_error),
        "workspace_shape": shape,
    }
    if shape == "rectangle":
        diagnostics["workspace_inset_m"] = workspace_inset_m
        diagnostics["workspace_size_m"] = np.linalg.norm(
            np.roll(np.asarray(outline), -1, axis=0) - outline, axis=1,
        )[:2].tolist()
    return calibration, diagnostics


def calibration_from_single_marker(
    image_size_px: tuple[int, int],
    image_points_px: tuple[tuple[float, float], ...],
    marker_size_m: float,
    workspace_roi_px: tuple[float, float, float, float] | None = None,
) -> WorkspaceCalibration:
    """Calibrate the marker's plane; origin is its centre, +y is its decoded top."""
    if not np.isfinite(marker_size_m) or marker_size_m <= 0:
        raise ValueError("Marker side length must be positive and finite")
    points = np.asarray(image_points_px, dtype=np.float32)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("Expected four finite marker corners in decoded order")
    if not cv2.isContourConvex(points) or cv2.contourArea(points) < 16:
        raise ValueError("Marker corners must form a non-degenerate convex quadrilateral")
    width, height = image_size_px
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x, y, roi_width, roi_height = workspace_roi_px or (0, 0, width - 1, height - 1)
    if (roi_width <= 0 or roi_height <= 0 or x < 0 or y < 0
            or x + roi_width > width - 1 or y + roi_height > height - 1):
        raise ValueError("Workspace ROI must be a positive rectangle within the image")
    half = float(marker_size_m) / 2
    desk_points = ((-half, half), (half, half), (half, -half), (-half, -half))
    calibration = WorkspaceCalibration(image_size_px, image_points_px, desk_points)
    # Calibration references and the working area are distinct: a small marker
    # must not shrink the simulator's workspace to the marker's footprint.
    roi_points = ((x, y), (x + roi_width, y),
                  (x + roi_width, y + roi_height), (x, y + roi_height))
    outline = tuple(tuple(float(v) for v in calibration.pixel_to_desk_xy(p)) for p in roi_points)
    centre = points.mean(axis=0)
    excluded = tuple(tuple(float(v) for v in p) for p in centre + 1.18 * (points - centre))
    return WorkspaceCalibration(
        image_size_px, image_points_px, desk_points, workspace_outline_m=outline,
        workspace_roi_px=workspace_roi_px, excluded_polygons_px=(excluded,),
    )


def detect_aruco_workspace_points(
    image_bgr: np.ndarray,
    marker_id_to_desk_xy_m: dict[int, tuple[float, float]],
    dictionary_name: int = 0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return matched pixel/desk points using ArUco marker centers.

    This is optional; the default project config uses fixed points so the full
    pipeline can run without a GUI calibration step.
    """

    corners, ids, _ = _detect_aruco(image_bgr, dictionary_name)
    if ids is None:
        return [], []

    image_points: list[tuple[float, float]] = []
    desk_points: list[tuple[float, float]] = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id_int = int(marker_id)
        if marker_id_int not in marker_id_to_desk_xy_m:
            continue
        center = np.mean(marker_corners.reshape(-1, 2), axis=0)
        image_points.append((float(center[0]), float(center[1])))
        desk_points.append(tuple(marker_id_to_desk_xy_m[marker_id_int]))
    return image_points, desk_points
