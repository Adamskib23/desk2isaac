from __future__ import annotations

import cv2
import numpy as np
import pytest

from humanoid_retarget.calibration import (
    ArucoMarker, WorkspaceCalibration, calibration_from_config, calibration_from_depth_markers,
    calibration_from_single_marker, find_aruco_marker, rectangular_workspace_from_markers,
)
from humanoid_retarget.visualization import rectify_workspace


def test_workspace_homography_round_trip() -> None:
    calibration = WorkspaceCalibration(
        image_size_px=(100, 100),
        image_points_px=((10, 10), (90, 10), (90, 90), (10, 90)),
        desk_points_m=((-0.2, 0.2), (0.2, 0.2), (0.2, -0.2), (-0.2, -0.2)),
    )
    pixel = np.asarray([42.0, 73.0], dtype=np.float32)
    desk = calibration.pixel_to_desk_xy(pixel)
    pixel_back = calibration.desk_to_pixel_xy(desk)
    assert np.allclose(pixel_back, pixel, atol=1e-3)


def _marker_image() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    else:
        marker = cv2.aruco.drawMarker(dictionary, 0, 200)
    image = np.full((400, 600), 255, dtype=np.uint8)
    image[100:300, 200:400] = marker
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_single_marker_scale_and_decoded_axes(rotation: int) -> None:
    image = np.rot90(_marker_image(), rotation).copy()
    marker = find_aruco_marker(image, marker_id=0)
    height, width = image.shape[:2]
    calibration = calibration_from_single_marker((width, height), marker.image_points_px, 0.1064)
    centre_px = calibration.desk_to_pixel_xy((0, 0))
    assert np.allclose(centre_px, ((width - 1) / 2, (height - 1) / 2), atol=1)
    expected = ((-0.0532, 0.0532), (0.0532, 0.0532), (0.0532, -0.0532), (-0.0532, -0.0532))
    for pixel, desk in zip(marker.image_points_px, expected):
        assert np.allclose(calibration.pixel_to_desk_xy(pixel), desk, atol=1e-6)
    assert calibration.desk_limits_m["x"][1] - calibration.desk_limits_m["x"][0] > 0.1064
    right_edge_midpoint = np.mean(np.asarray(marker.image_points_px)[[1, 2]], axis=0)
    assert np.allclose(calibration.pixel_to_desk_xy(right_edge_midpoint), (0.0532, 0), atol=1e-6)


def test_single_marker_missing_or_wrong_id_rejected() -> None:
    with pytest.raises(ValueError, match="No matching"):
        find_aruco_marker(_marker_image(), marker_id=1)
    with pytest.raises(ValueError, match="No matching"):
        find_aruco_marker(np.full((400, 600, 3), 255, dtype=np.uint8), marker_id=0)


def test_marker_size_and_degenerate_corners_rejected() -> None:
    corners = ((100, 100), (300, 100), (300, 300), (100, 300))
    for side in (0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="side length"):
            calibration_from_single_marker((400, 400), corners, side)
    with pytest.raises(ValueError, match="non-degenerate"):
        calibration_from_single_marker((400, 400), ((0, 0), (1, 1), (2, 2), (3, 3)), 0.1064)


def test_workspace_outline_survives_config_loading() -> None:
    config = {"workspace": {
        "image_size_px": [400, 400],
        "image_points_px": [[100, 100], [300, 100], [300, 300], [100, 300]],
        "desk_points_m": [[-0.05, 0.05], [0.05, 0.05], [0.05, -0.05], [-0.05, -0.05]],
        "workspace_outline_m": [[-0.3, 0.2], [0.3, 0.2], [0.3, -0.2], [-0.3, -0.2]],
    }}
    calibration = calibration_from_config(config)
    assert np.allclose(calibration.desk_limits_m["x"], (-0.3, 0.3))
    assert np.allclose(calibration.desk_limits_m["y"], (-0.2, 0.2))


def _tilted_rgbd_markers(layout=None):
    width, height = 960, 720
    matrix = np.asarray([[750.0, 0, 480], [0, 755.0, 360], [0, 0, 1]])
    distortion = (0.02, -0.01, 0.001, -0.002, 0.0)
    rotation = cv2.Rodrigues(np.asarray([0.25, -0.07, 0.10]))[0] @ np.diag([1.0, -1.0, -1.0])
    translation = np.asarray([0.0, 0.0, 0.8])
    rvec = cv2.Rodrigues(rotation)[0]
    side = 0.1064
    half = side / 2
    local = np.asarray([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    markers = []
    if layout is None:
        layout = [(0, 0, 0), (-0.20, 0.13, 0.6), (0.21, -0.12, -0.9)]
    for marker_id, (x, y, yaw) in enumerate(layout):
        orientation = cv2.Rodrigues(np.asarray([0.0, 0.0, yaw]))[0]
        world = local @ orientation.T + np.asarray([x, y, 0])
        corners = cv2.projectPoints(world, rvec, translation, matrix, np.asarray(distortion))[0].reshape(4, 2)
        markers.append(ArucoMarker(marker_id, "DICT_4X4_1000", tuple(tuple(p) for p in corners)))
    py, px = np.indices((height, width))
    pixels = np.column_stack((px.ravel(), py.ravel())).astype(float)
    undistorted = cv2.undistortPoints(pixels.reshape(-1, 1, 2), matrix, np.asarray(distortion)).reshape(-1, 2)
    rays = np.column_stack((undistorted, np.ones(len(undistorted))))
    normal = rotation[:, 2]
    depth = ((normal @ translation) / (rays @ normal)).reshape(height, width)
    random = np.random.default_rng(42)
    depth += random.normal(0, 0.0004, depth.shape)
    depth[random.random(depth.shape) < 0.05] = 0
    depth[random.random(depth.shape) < 0.01] += 0.050
    return (width, height), matrix, distortion, rotation, translation, markers, depth


def test_rgbd_calibration_corrects_tilt_distortion_and_lift_parallax() -> None:
    size, matrix, distortion, rotation, translation, markers, depth = _tilted_rgbd_markers()
    calibration, diagnostics = calibration_from_depth_markers(size, markers, 0.1064, depth, matrix, distortion)
    assert diagnostics["plane_rms_m"] < 0.001
    assert diagnostics["max_marker_size_error_fraction"] < 0.01
    assert np.linalg.det(np.asarray(calibration.camera_to_desk_m)[:3, :3]) == pytest.approx(1, abs=1e-6)
    lifted = np.asarray([0.12, -0.06, 0.15])
    camera_point = rotation @ lifted + translation
    pixel = cv2.projectPoints(lifted.reshape(1, 3), cv2.Rodrigues(rotation)[0], translation,
                              matrix, np.asarray(distortion))[0].reshape(2)
    recovered = calibration.pixels_depth_to_desk_xyz(pixel.reshape(1, 2), [camera_point[2]])[0]
    assert np.allclose(recovered, lifted, atol=0.001)
    assert np.linalg.norm(calibration.pixel_to_desk_xy(pixel) - lifted[:2]) > 0.01
    desk_pixel = calibration.desk_to_pixel_xy((0.10, -0.04))
    assert np.allclose(calibration.pixel_to_desk_xy(desk_pixel), (0.10, -0.04), atol=1e-5)


def test_rgbd_calibration_rejects_wrong_print_size() -> None:
    size, matrix, distortion, _, _, markers, depth = _tilted_rgbd_markers()
    with pytest.raises(ValueError, match="measured size"):
        calibration_from_depth_markers(size, markers, 0.05, depth, matrix, distortion)


@pytest.mark.parametrize("yaw", [0.0, 0.55, 1.7])
def test_four_markers_define_an_inscribed_metric_rectangle(yaw: float) -> None:
    centres = np.asarray([[0.20, -0.13], [-0.20, 0.13], [0.21, 0.12], [-0.19, -0.14]])
    orientation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    centres = centres @ orientation.T
    layout = [(x, y, marker_yaw + yaw) for (x, y), marker_yaw in zip(centres, [0, 0.6, -0.9, 1.3])]
    size, matrix, distortion, _, _, markers, depth = _tilted_rgbd_markers(layout)
    calibration, diagnostics = calibration_from_depth_markers(size, markers, 0.1064, depth, matrix, distortion)
    outline = np.asarray(calibration.workspace_outline_m)
    assert outline.shape == (4, 2)
    edges = np.roll(outline, -1, axis=0) - outline
    assert abs(edges[0] @ edges[1]) < 1e-7
    assert np.allclose(edges[:2], -edges[2:], atol=1e-7)
    hull = cv2.convexHull(np.asarray(calibration.desk_points_m, dtype=np.float32))
    assert all(cv2.pointPolygonTest(hull, tuple(p), True) >= 0.005 - 1e-6 for p in outline)
    assert diagnostics["workspace_shape"] == "rectangle"
    assert len(calibration.excluded_polygons_px) == 4
    shuffled, _ = calibration_from_depth_markers(size, markers[::-1], 0.1064, depth, matrix, distortion)
    assert np.allclose(shuffled.workspace_outline_m, outline, atol=1e-6)


def test_rectangular_workspace_rejects_non_corner_layout_or_excessive_inset() -> None:
    local = np.asarray([[-0.05, 0.05], [0.05, 0.05], [0.05, -0.05], [-0.05, -0.05]])
    collinear = local + np.asarray([[0, 0], [0.2, 0], [0.4, 0], [0.6, 0]])[:, None, :]
    with pytest.raises(ValueError, match="four distinct workspace corners"):
        rectangular_workspace_from_markers(collinear)
    valid = local + np.asarray([[0, 0], [0.4, 0], [0.4, 0.3], [0, 0.3]])[:, None, :]
    with pytest.raises(ValueError, match="no usable rectangle"):
        rectangular_workspace_from_markers(valid, inset_m=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        rectangular_workspace_from_markers(valid, inset_m=-0.01)


def test_explicit_rectangle_rejects_three_markers_and_image_roi() -> None:
    size, matrix, distortion, _, _, markers, depth = _tilted_rgbd_markers()
    with pytest.raises(ValueError, match="exactly four markers and no image ROI"):
        calibration_from_depth_markers(size, markers, 0.1064, depth, matrix, distortion,
                                       workspace_shape="rectangle")
    layout = [(0.2, -0.13, 0), (-0.2, 0.13, 0.6), (0.2, 0.13, -0.9), (-0.2, -0.13, 1.3)]
    size, matrix, distortion, _, _, markers, depth = _tilted_rgbd_markers(layout)
    with pytest.raises(ValueError, match="exactly four markers and no image ROI"):
        calibration_from_depth_markers(size, markers, 0.1064, depth, matrix, distortion,
                                       workspace_roi_px=(10, 10, 900, 600))


def test_topdown_rectification_preserves_plane_scale_orientation_and_pixels() -> None:
    outline = ((0, 0.1), (0.2, 0.1), (0.2, 0), (0, 0))
    calibration = WorkspaceCalibration(
        (201, 101), ((0, 0), (200, 0), (200, 100), (0, 100)), outline,
        workspace_outline_m=outline,
    )
    y, x = np.indices((101, 201))
    image = np.stack((x, y, np.full_like(x, 100)), axis=-1).astype(np.uint8)
    rectified = rectify_workspace(image, calibration, width_px=201)
    assert rectified.shape == image.shape
    assert np.allclose(rectified, image, atol=1)
    with pytest.raises(ValueError, match="Image dimensions"):
        rectify_workspace(image[:100], calibration)


def test_topdown_rectification_rejects_non_rectangular_quadrilateral() -> None:
    calibration = WorkspaceCalibration(
        (201, 101), ((0, 0), (200, 0), (200, 100), (0, 100)),
        ((0, 0.1), (0.2, 0.1), (0.2, 0), (0, 0)),
        workspace_outline_m=((0, 0.1), (0.2, 0.1), (0.17, 0), (0, 0)),
    )
    with pytest.raises(ValueError, match="metric rectangle"):
        rectify_workspace(np.zeros((101, 201, 3), dtype=np.uint8), calibration)


@pytest.mark.parametrize("distortion", [(), (0.1, -0.05, 0.001, -0.002, 0.0)])
def test_3d_topdown_rectification_matches_lens_correction(distortion) -> None:
    outline = ((-0.1, 0.05), (0.1, 0.05), (0.1, -0.05), (-0.1, -0.05))
    matrix = np.asarray([[200.0, 0, 100], [0, 200.0, 50], [0, 0, 1]])
    calibration = WorkspaceCalibration(
        (201, 101), ((0, 0), (200, 0), (200, 100), (0, 100)), outline,
        workspace_outline_m=outline,
        camera_matrix_px=tuple(tuple(row) for row in matrix),
        camera_distortion=distortion,
        camera_to_desk_m=((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0.2), (0, 0, 0, 1)),
    )
    y, x = np.indices((101, 201))
    image = np.stack((x, y, np.full_like(x, 100)), axis=-1).astype(np.uint8)
    rectified = rectify_workspace(image, calibration, width_px=201)
    if distortion:
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrix, np.asarray(distortion), None, matrix, (201, 101), cv2.CV_32FC1,
        )
        expected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    else:
        expected = image
    assert rectified.shape == image.shape
    assert np.allclose(rectified, expected, atol=1)
