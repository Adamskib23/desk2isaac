#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from humanoid_retarget.calibration import calibration_from_config
from humanoid_retarget.io import list_rgb_frames, read_depth_m, read_episode_metadata, read_yaml, write_json
from humanoid_retarget.tracking import detect_object, load_object_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect object tracking on a saved RGB-D frame.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--workspace-config", default="configs/workspace.real.yaml")
    parser.add_argument("--objects-config", default="configs/objects.real.yaml")
    parser.add_argument("--out", type=Path, default=Path("outputs/real_setup/tracking.png"))
    args = parser.parse_args()
    metadata = read_episode_metadata(args.episode)
    calibration = calibration_from_config(read_yaml(args.workspace_config))
    if calibration.camera_to_desk_m is not None and metadata.get("depth_aligned_to") != "color":
        raise ValueError("3D tracking needs explicitly aligned RGB-D frames")
    path = list_rgb_frames(args.episode)[args.frame]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    depth = read_depth_m(args.episode, path.stem, metadata.get("depth_unit_m", 0.001))
    specs = load_object_specs(args.objects_config)
    preview = image.copy()
    if calibration.workspace_outline_m:
        polygon = np.asarray([calibration.desk_to_pixel_xy(p) for p in calibration.workspace_outline_m])
        cv2.polylines(preview, [np.round(polygon).astype(np.int32)], True, (0, 210, 210), 2)
    detections = []
    for spec in specs:
        detection = detect_object(image, depth, spec, calibration, args.frame, args.frame / metadata.get("fps", 30))
        if detection is None:
            print(f"{spec.name}: NOT DETECTED")
            continue
        detections.append(asdict(detection))
        x, y = (int(round(v)) for v in detection.pixel_xy)
        cv2.circle(preview, (x, y), 12, (255, 255, 255), 2)
        mm = np.asarray(detection.desk_xyz_m) * 1000
        label = f"{spec.name}: ({mm[0]:.0f}, {mm[1]:.0f}, {mm[2]:.0f}) mm"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_x = max(4, min(x + 18, preview.shape[1] - text_width - 8))
        label_y = max(text_height + 8, y - 12)
        cv2.rectangle(preview, (label_x - 4, label_y - text_height - 4),
                      (label_x + text_width + 4, label_y + baseline + 4), (25, 25, 25), -1)
        cv2.putText(preview, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        print(f"{label} | {detection.position_source} | confidence {detection.confidence:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), preview):
        raise OSError(f"Could not write {args.out}")
    write_json(args.out.with_suffix(".json"), {"source_image": str(path), "detections": detections})
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
