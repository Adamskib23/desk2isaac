from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObjectSpec:
    """Visual and physical description of an object that exists in the camera view and sim."""

    name: str
    shape: str
    size_m: tuple[float, float, float]
    hsv_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]]
    color_bgr: tuple[int, int, int] = (40, 40, 220)
    min_area_px: int = 160
    base_diameter_m: float | None = None


@dataclass(frozen=True)
class Detection:
    """Single-frame object observation in both image and desk coordinates."""

    frame_idx: int
    t: float
    object_name: str
    pixel_xy: tuple[float, float]
    desk_xyz_m: tuple[float, float, float]
    mask_area_px: float
    confidence: float
    visible: bool = True
    position_source: str = "desk_plane"


@dataclass(frozen=True)
class Waypoint:
    """End-effector command generated from real object motion."""

    t: float
    phase: str
    ee_xyz_m: tuple[float, float, float]
    ee_quat_wxyz: tuple[float, float, float, float]
    gripper: str
    object_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class Trajectory:
    """Portable artifact consumed by the top-down and Isaac Lab replay scripts."""

    schema_version: str
    source_episode: str
    object_name: str
    fps: float
    events: dict[str, Any]
    object: dict[str, Any]
    workspace: dict[str, Any]
    object_trace: list[dict[str, Any]] = field(default_factory=list)
    waypoints: list[dict[str, Any]] = field(default_factory=list)
    scene_objects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
