from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def frame_tool():
    path = Path(__file__).resolve().parents[1] / "scripts/sam2_segment_frame.py"
    spec = importlib.util.spec_from_file_location("sam2_frame_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_box_retains_original_pixel_coordinates(frame_tool):
    box = frame_tool.validate_box([560, 365, 705, 510], 1920, 1080)
    assert box.dtype == np.float32
    np.testing.assert_array_equal(box, [560, 365, 705, 510])


@pytest.mark.parametrize("box", [
    [-1, 0, 10, 10], [0, 0, 21, 10], [0, 0, 10, 21],
    [10, 0, 10, 10], [10, 0, 0, 10], [0, 10, 10, 0],
    [0, 0, float("nan"), 10], [0, 0, float("inf"), 10], [0, 0, 10],
])
def test_invalid_boxes_are_rejected(frame_tool, box):
    with pytest.raises(ValueError):
        frame_tool.validate_box(box, 20, 20)


def test_preview_preserves_source_and_marks_selected_pixels(frame_tool):
    image = np.zeros((30, 30, 3), np.uint8)
    mask = np.zeros((30, 30), bool)
    mask[10:20, 10:20] = True
    preview = frame_tool.mask_preview(image, mask, np.asarray([5, 5, 25, 25]))
    assert not image.any()
    assert preview[15, 15, 1] > 0
    assert not preview[0, 0].any()
    assert preview.shape == image.shape


def test_preview_rejects_resized_mask(frame_tool):
    with pytest.raises(ValueError, match="dimensions"):
        frame_tool.mask_preview(np.zeros((30, 30, 3), np.uint8), np.zeros((15, 15), bool),
                                np.asarray([0, 0, 10, 10]))
