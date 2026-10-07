from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.physical import build_physical_plan
from humanoid_retarget.planner import build_trajectory
from humanoid_retarget.tracking import detect_object, load_object_specs
from humanoid_retarget.types import Detection


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_pilot_replaces_glass_without_changing_other_object_specs() -> None:
    original = {spec.name: spec for spec in load_object_specs(CONFIGS / "objects.real.yaml")}
    pilot = {spec.name: spec for spec in load_object_specs(CONFIGS / "objects.rl_pilot.yaml")}
    assert "green_shot_glass" in original
    assert set(pilot) == {"blue_sphere", "black_box", "yellow_mustard_tin"}
    for name in ("blue_sphere", "black_box"):
        assert pilot[name] == original[name]
    tin = pilot["yellow_mustard_tin"]
    assert tin.shape == "cuboid"
    assert tin.size_m == (0.051, 0.035, 0.085)
    assert tin.color_bgr == (35, 195, 235)


@pytest.mark.parametrize("hue,value", [(20, 90), (28, 180), (36, 245)])
def test_pilot_yellow_tin_mask_separates_saturated_yellow_from_background(
    hue: int, value: int,
) -> None:
    specs = load_object_specs(CONFIGS / "objects.rl_pilot.yaml")
    hsv = np.full((200, 300, 3), (20, 65, 175), dtype=np.uint8)
    hsv[80:140, 130:170] = (hue, 210, value)
    hsv[30:65, 30:65] = (100, 210, 200)
    hsv[130:165, 220:255] = (10, 25, 90)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    calibration = WorkspaceCalibration(
        (300, 200), ((0, 0), (299, 0), (299, 199), (0, 199)),
        ((0, 0.2), (0.3, 0.2), (0.3, 0), (0, 0)), table_depth_m=0.8,
    )
    depth = np.full((200, 300), 0.8, dtype=np.float32)
    depth[80:140, 130:170] = 0.715
    expected = {
        "yellow_mustard_tin": (149.5, 109.5),
        "blue_sphere": (47, 47),
        "black_box": (237, 147),
    }
    for spec in specs:
        detection = detect_object(frame, depth, spec, calibration, 0, 0)
        assert detection is not None
        assert np.allclose(detection.pixel_xy, expected[spec.name], atol=1)
        if spec.name == "yellow_mustard_tin":
            assert detection.desk_xyz_m[2] == pytest.approx(0.0425, abs=1e-6)


def test_pilot_tin_dimensions_reach_scene_and_physical_plan() -> None:
    specs = load_object_specs(CONFIGS / "objects.rl_pilot.yaml")
    calibration = WorkspaceCalibration(
        (300, 200), ((0, 0), (299, 0), (299, 199), (0, 199)),
        ((0, 0.2), (0.3, 0.2), (0.3, 0), (0, 0)),
    )
    rows = []
    for i in range(120):
        alpha = float(np.clip((i - 25) / 30, 0, 1))
        for j, spec in enumerate(specs):
            xyz = (0.03 + j * 0.05, 0.04, spec.size_m[2] / 2)
            if spec.name == "yellow_mustard_tin":
                xyz = (0.1 + 0.15 * alpha, 0.1, 0.0425 + 0.08 * np.sin(np.pi * alpha))
            rows.append(Detection(i, i / 30, spec.name, (100, 100), xyz, 2000, 0.9))
    trajectory = build_trajectory(rows, specs, calibration, {"fps": 30}, {
        "sim_origin_m": [0.55, 0, 1.05], "sim_table_z_m": 1.05, "mapping": "identity",
    }, object_name="yellow_mustard_tin")
    assert trajectory.object["size_m"] == (0.051, 0.035, 0.085)
    scene = {obj["name"]: obj for obj in trajectory.scene_objects}
    assert set(scene) == {spec.name for spec in specs}
    assert scene["yellow_mustard_tin"]["size_m"] == (0.051, 0.035, 0.085)
    plan = build_physical_plan(trajectory.to_dict())
    assert plan["task_input"]["start_xyz_m"][2] == pytest.approx(1.0925)
    assert plan["task_input"]["goal_xyz_m"][2] == pytest.approx(1.0925)
