#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.calibration import calibration_from_config
from humanoid_retarget.dataset_review import load_review_manifest
from humanoid_retarget.io import read_json, read_yaml, write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.physical import build_physical_plan
from humanoid_retarget.tracking import load_object_specs
from humanoid_retarget.yaw_tracking import estimate_episode_yaw


def boundary_preview(task, calibration, output):
    episode = Path(task["source_episode"])
    frames = task["orientation"]["frames"]
    panels = []
    for label, index in (("START", 0), ("GOAL", len(frames) - 1)):
        image = cv2.imread(str(episode / "rgb" / f"{index:06d}.jpg"))
        if image is None:
            raise ValueError("Cannot decode yaw boundary preview")
        mask = cv2.imread(str(Path(task["events"]["mask_manifest"]).parent / f"{index:06d}.png"), 0)
        py, px = np.nonzero(mask)
        estimate = frames[index]
        if len(px):
            height = estimate.get("upper_face_height_m", task["object"]["size_m"][2])
            centre = calibration.pixel_to_desk_at_height((float(np.median(px)), float(np.median(py))), height)
            if estimate["status"] == "observed":
                angle = estimate["desk_yaw_rad"]
                direction = np.array([np.cos(angle), np.sin(angle), 0]) * task["object"]["size_m"][0] / 2
                transform = np.asarray(calibration.camera_to_desk_m)
                camera = (np.asarray([centre - direction, centre + direction]) - transform[:3, 3]) @ transform[:3, :3]
                pixels, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), np.asarray(calibration.camera_matrix_px),
                                              np.asarray(calibration.camera_distortion))
                a, b = np.round(pixels.reshape(-1, 2)).astype(int)
                cv2.line(image, tuple(a), tuple(b), (20, 240, 20), 3, cv2.LINE_AA)
            cx, cy = int(np.median(px)), int(np.median(py))
            radius = max(120, int(max(np.ptp(px), np.ptp(py))) + 40)
            image = image[max(0, cy - radius):min(image.shape[0], cy + radius),
                          max(0, cx - radius):min(image.shape[1], cx + radius)]
        image = cv2.resize(image, (480, 480))
        angle_text = f"{np.rad2deg(estimate['desk_yaw_rad']):.1f} deg" if estimate["status"] == "observed" else estimate["status"]
        cv2.rectangle(image, (0, 0), (480, 64), (20, 20, 20), -1)
        cv2.putText(image, label + " desk yaw: " + angle_text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1)
        cv2.putText(image, task["object_name"] + " | " + task["orientation"]["mode"], (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1)
        panels.append(image)
    if not cv2.imwrite(str(output), np.hstack(panels)):
        raise OSError("Could not write yaw preview")


def prepare(manifest_path: Path, objects_path: Path, directory: Path) -> Path:
    directory = directory.resolve()
    if (not directory.is_relative_to(ROOT) or directory.is_relative_to(ROOT / "data")
            or directory.is_relative_to(manifest_path.parent.resolve())
            or (directory.exists() and any(directory.iterdir()))):
        raise ValueError("Use a fresh repository output outside captures and the original dataset")
    manifest = load_review_manifest(manifest_path, ROOT)
    specs = {s.name: s for s in load_object_specs(objects_path)}
    output = deepcopy(manifest)
    output.update(parent_manifest=str(manifest_path.resolve().relative_to(ROOT)),
                  parent_manifest_sha256=file_sha256(manifest_path), training_ready=False,
                  notes="Derived upright shape-axis yaw tasks. Position-only fallback remains explicit. "
                        "Original 30 tasks/labels/captures unchanged; no teacher labels approved, no learned policy.")
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for row in output["clips"]:
        source_path = ROOT / row["trajectory"]
        task = read_json(source_path)
        workspace_path = ROOT / row["calibration"]["config"]
        if (file_sha256(workspace_path) != task["events"]["workspace_config_sha256"]
                or file_sha256(objects_path) != task["events"]["objects_config_sha256"]):
            raise ValueError("Yaw configuration differs from source task")
        calibration = calibration_from_config(read_yaml(workspace_path))
        augmented = estimate_episode_yaw(task, specs[row["object_name"]], calibration)
        build_physical_plan(augmented)
        orientation = augmented["orientation"]
        new_path = directory / "tasks" / f"{row['index']:03d}_{row['object_name']}.json"
        write_json(new_path, augmented)
        preview_path = directory / "previews" / f"{row['index']:03d}_{row['object_name']}.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        boundary_preview(augmented, calibration, preview_path)
        row["parent_trajectory"] = row["trajectory"]
        row["trajectory"] = str(new_path.relative_to(ROOT))
        row["yaw_preview"] = str(preview_path.relative_to(ROOT))
        row["yaw_mode"] = orientation["mode"]
        for path in (new_path, preview_path, manifest_path, objects_path, Path(__file__),
                     ROOT / "src/humanoid_retarget/yaw_tracking.py"):
            row["file_sha256"][str(path.resolve().relative_to(ROOT))] = file_sha256(path)
        yaw_frames = sum(r["status"] == "observed" for r in orientation["frames"])
        report = {"index": row["index"], "object_name": row["object_name"], "mode": orientation["mode"],
                  "observed_yaw_frames": yaw_frames, "total_frames": len(orientation["frames"]),
                  "initial": orientation["initial"], "final": orientation["final"],
                  "status_counts": dict(Counter(r["status"] for r in orientation["frames"]))}
        reports.append(report)
        print(f"{row['index']:02d}/{output['clip_count']} {row['object_name']}: {orientation['mode']}, "
              f"yaw observed {yaw_frames}/{len(orientation['frames'])}", flush=True)
    summary = {"clip_count": len(reports), "yaw_enabled_clips": sum(r["mode"] == "observed_yaw" for r in reports),
               "position_only_clips": sum(r["mode"] != "observed_yaw" for r in reports), "clips": reports,
               "training_ready": False}
    output["yaw_summary"] = {k: v for k, v in summary.items() if k != "clips"}
    write_json(directory / "yaw_report.json", summary)
    path = directory / "manifest.json"
    write_json(path, output)
    load_review_manifest(path, ROOT)
    return path


def main():
    parser = argparse.ArgumentParser(description="Add confidence-gated upright yaw to a fresh copy of a frozen SAM RGB-D dataset")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/datasets/pilot_030/manifest.json")
    parser.add_argument("--objects-config", type=Path, default=ROOT / "configs/objects.rl_pilot.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(f"Yaw manifest: {prepare(args.manifest, args.objects_config, args.out)}")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Yaw preparation failed: {error}. Partial artifacts preserved; choose a fresh output on retry.\n")


if __name__ == "__main__":
    main()
