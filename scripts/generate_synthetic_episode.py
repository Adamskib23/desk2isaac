#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from humanoid_retarget.calibration import calibration_from_config
from humanoid_retarget.io import read_yaml
from humanoid_retarget.tracking import load_object_specs


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def draw_rect_on_desk(
    frame: np.ndarray,
    depth_m: np.ndarray,
    calibration,
    center_xy: np.ndarray,
    size_xy: tuple[float, float],
    height_m: float,
    color_bgr: tuple[int, int, int],
    table_depth_m: float,
) -> None:
    sx, sy = size_xy[0] * 0.5, size_xy[1] * 0.5
    corners = np.asarray(
        [
            [center_xy[0] - sx, center_xy[1] - sy],
            [center_xy[0] + sx, center_xy[1] - sy],
            [center_xy[0] + sx, center_xy[1] + sy],
            [center_xy[0] - sx, center_xy[1] + sy],
        ],
        dtype=np.float32,
    )
    px = np.asarray([calibration.desk_to_pixel_xy(corner) for corner in corners], dtype=np.int32)
    cv2.fillConvexPoly(frame, px, color_bgr)
    cv2.polylines(frame, [px.reshape(-1, 1, 2)], True, (35, 35, 35), 2)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, px, 255)
    depth_m[mask > 0] = table_depth_m - height_m


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic RGB-D manipulation episode.")
    parser.add_argument("--out", default="data/synthetic/red_cube_slide", help="Output episode directory.")
    parser.add_argument("--workspace-config", default="configs/workspace.yaml")
    parser.add_argument("--objects-config", default="configs/objects.yaml")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    out = Path(args.out)
    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    workspace_cfg = read_yaml(args.workspace_config)
    calibration = calibration_from_config(workspace_cfg)
    specs = {spec.name: spec for spec in load_object_specs(args.objects_config)}
    width, height = calibration.image_size_px
    table_depth_m = calibration.table_depth_m or 1.2

    static_objects = [
        ("blue_block", np.asarray([-0.18, 0.10], dtype=np.float32)),
        ("green_cylinder", np.asarray([0.18, -0.12], dtype=np.float32)),
    ]
    start_xy = np.asarray([-0.18, -0.10], dtype=np.float32)
    end_xy = np.asarray([0.17, 0.11], dtype=np.float32)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(out / "source_preview.mp4"), fourcc, args.fps, (width, height))

    for idx in range(args.frames):
        frame = np.full((height, width, 3), (184, 181, 170), dtype=np.uint8)
        depth = np.full((height, width), table_depth_m, dtype=np.float32)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (155, 152, 143), 8)

        for name, xy in static_objects:
            spec = specs[name]
            draw_rect_on_desk(frame, depth, calibration, xy, spec.size_m[:2], spec.size_m[2], spec.color_bgr, table_depth_m)

        alpha = smoothstep((idx - 38) / 96.0)
        cube_xy = (1.0 - alpha) * start_xy + alpha * end_xy
        lift_m = 0.12 * np.sin(np.pi * alpha) if 0.0 < alpha < 1.0 else 0.0
        hand_px = calibration.desk_to_pixel_xy(cube_xy + np.asarray([-0.018, -0.012], dtype=np.float32))
        if 28 < idx < 145:
            overlay = frame.copy()
            cv2.ellipse(
                overlay,
                tuple(int(v) for v in hand_px),
                (72, 45),
                -18,
                0,
                360,
                (120, 160, 205),
                -1,
            )
            frame = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

        spec = specs["red_cube"]
        draw_rect_on_desk(
            frame,
            depth,
            calibration,
            cube_xy,
            spec.size_m[:2],
            spec.size_m[2] + lift_m,
            spec.color_bgr,
            table_depth_m,
        )

        cv2.putText(frame, f"synthetic frame {idx:03d}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (55, 55, 55), 2)
        cv2.imwrite(str(rgb_dir / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        depth_mm = np.clip(np.round(depth * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{idx:06d}.png"), depth_mm)
        video.write(frame)

    video.release()
    metadata = {
        "schema_version": "hmnd_rgbd_episode_v1",
        "fps": args.fps,
        "width": width,
        "height": height,
        "depth_unit_m": 0.001,
        "camera": "synthetic_overhead_rgbd",
        "notes": "Generated only for testing. Replace with personally collected Gemini 2 data for submission.",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Wrote synthetic episode to {out}")


if __name__ == "__main__":
    main()
