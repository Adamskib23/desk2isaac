#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from humanoid_retarget.calibration import calibration_from_config
from humanoid_retarget.io import read_yaml, write_json
from humanoid_retarget.mask_tracking import (
    endpoint_observation_check, file_sha256, save_mask_tracking_video, track_mask_episode,
)
from humanoid_retarget.planner import build_trajectory
from humanoid_retarget.tracking import filter_detections, load_object_specs, track_episode
from humanoid_retarget.visualization import save_detection_video
from humanoid_retarget.yaw_tracking import estimate_episode_yaw


def validate_mask_output(out: Path, episode: Path, mask_dir: Path, save_video: bool) -> None:
    reserved = {"raw_detections.json", "detections.json", "tracking_quality.json", "detections.mp4"}
    if out.name in reserved:
        raise ValueError("Trajectory filename conflicts with a tracking sidecar")
    targets = [out, *(out.parent / name for name in reserved if save_video or name != "detections.mp4")]
    for target in targets:
        if any(target.resolve().is_relative_to(root.resolve()) for root in (episode, mask_dir)):
            raise ValueError("Write derived SAM tracking outputs outside the original episode and mask directory")
        if target.exists():
            raise ValueError(f"Output already exists: {target}; choose a fresh --out directory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a recorded RGB-D episode into retargeted robot waypoints.")
    parser.add_argument("--episode", required=True, help="Episode directory containing rgb/, optional depth/, metadata.json.")
    parser.add_argument("--workspace-config", default="configs/workspace.yaml")
    parser.add_argument("--objects-config", default="configs/objects.yaml")
    parser.add_argument("--object-name", help="Known object label for a single-object pilot; otherwise infer from movement.")
    parser.add_argument("--out", default=None, help="Output trajectory JSON.")
    parser.add_argument("--fps", type=float, default=None, help="Override episode FPS.")
    parser.add_argument("--no-video", action="store_true", help="Skip annotated source video.")
    parser.add_argument("--no-filter", action="store_true", help="Use raw tracking without temporal stabilisation.")
    parser.add_argument("--mask-dir", type=Path, help="Fingerprint-checked SAM video masks; require calibrated aligned depth.")
    parser.add_argument("--with-yaw", action="store_true", help="Add gated upright cuboid yaw from SAM masks/aligned depth (fresh outputs)")
    parser.add_argument("--preview-width", type=int, help="SAM annotated-video width only; masks/depth remain full resolution.")
    args = parser.parse_args()
    if args.with_yaw and args.mask_dir is None:
        parser.error("--with-yaw requires --mask-dir and calibrated RGB-D")
    if args.preview_width is not None and (args.preview_width <= 0 or args.mask_dir is None):
        parser.error("--preview-width requires SAM masks and a positive width")

    episode = Path(args.episode)
    if args.mask_dir is not None and args.out is None:
        parser.error("SAM tracking requires explicit --out to preserve the existing HSV outputs")
    out = Path(args.out) if args.out else Path("outputs") / episode.name / "trajectory.json"
    output_dir = out.parent
    if args.mask_dir is not None:
        try:
            validate_mask_output(out, episode, args.mask_dir, not args.no_video)
        except ValueError as error:
            parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_cfg = read_yaml(args.workspace_config)
    calibration = calibration_from_config(workspace_cfg)
    retarget_cfg = workspace_cfg.get("retarget", {})
    specs = load_object_specs(args.objects_config)
    if args.object_name is not None and args.object_name not in {spec.name for spec in specs}:
        parser.error(f"Unknown --object-name {args.object_name!r}; configured: {', '.join(spec.name for spec in specs)}")
    if args.mask_dir is None:
        raw_detections, metadata = track_episode(episode, specs, calibration, fps_override=args.fps)
    else:
        raw_detections, metadata = track_mask_episode(episode, specs, calibration, args.mask_dir, fps_override=args.fps)
        if args.object_name is not None and args.object_name != metadata["action_object"]:
            parser.error("--object-name does not match the prompted SAM mask object")
        metadata["workspace_config_sha256"] = file_sha256(args.workspace_config)
        metadata["objects_config_sha256"] = file_sha256(args.objects_config)
    detections = raw_detections if args.no_filter else filter_detections(raw_detections, float(metadata["fps"]))
    metadata["tracking_filter"] = "none" if args.no_filter else "opencv_constant_velocity_kalman"
    trajectory, task_error = None, None
    try:
        if args.mask_dir is not None:
            metadata["endpoint_observations"] = endpoint_observation_check(
                detections, metadata["num_rgb_frames"], float(metadata["fps"]),
            )
            if not metadata["endpoint_observations"]["passed"]:
                raise ValueError("Start/end recording windows lack reliable observed positions; review tracking_quality.json")
        trajectory = build_trajectory(detections, specs, calibration, metadata, retarget_cfg, object_name=args.object_name)
    except ValueError as error:
        if args.mask_dir is None:
            raise
        task_error = str(error)
    write_json(output_dir / "detections.json", {"detections": [asdict(d) for d in detections], "metadata": metadata})
    write_json(output_dir / "raw_detections.json", {"detections": [asdict(d) for d in raw_detections], "metadata": metadata})
    if not args.no_video:
        if args.mask_dir is None:
            save_detection_video(episode, detections, specs, output_dir / "detections.mp4", float(metadata["fps"]))
        else:
            save_mask_tracking_video(episode, args.mask_dir, detections, metadata["frame_diagnostics"],
                                     output_dir / "detections.mp4", float(metadata["fps"]), args.preview_width)
    if args.mask_dir is not None:
        summary = {
            "tracking_backend": "sam2_rgbd", "training_ready": False,
            "frames": metadata["num_rgb_frames"], "raw_observations": len(raw_detections),
            "missing_frames": [row["frame_idx"] for row in metadata["frame_diagnostics"] if row["status"] != "observed"],
            "predicted_frames": [d.frame_idx for d in detections if not d.visible],
            "weak_frames": [d.frame_idx for d in detections if d.visible and d.confidence < 0.6],
            "endpoint_observations": metadata["endpoint_observations"],
            "task_generated": trajectory is not None, "task_error": task_error,
            "notes": metadata["label_notes"],
        }
        write_json(output_dir / "tracking_quality.json", summary)
    if trajectory is None:
        print(f"Saved tracking diagnostics to {output_dir}; no trajectory: {task_error}", file=sys.stderr)
        raise SystemExit(2)
    payload = trajectory.to_dict()
    if args.mask_dir is not None:
        payload["events"].update(tracking_backend="sam2_rgbd", training_ready=False,
                                  scene_object_coverage="selected_object_only",
                                  mask_manifest=metadata["mask_manifest"],
                                  mask_manifest_sha256=metadata["mask_manifest_sha256"],
                                  endpoint_observations=metadata["endpoint_observations"],
                                  workspace_config_sha256=metadata["workspace_config_sha256"],
                                  objects_config_sha256=metadata["objects_config_sha256"])
    if args.with_yaw:
        spec = next(s for s in specs if s.name == payload["object_name"])
        payload = estimate_episode_yaw(payload, spec, calibration)
        print(f"Orientation: {payload['orientation']['mode']} (shape-axis yaw, upright assumption)")
    write_json(out, payload)

    print(f"Selected object: {trajectory.object_name}")
    if trajectory.events["selection_method"] == "declared_object" and trajectory.events["automatic_object_name"] != trajectory.object_name:
        print(f"Declared label used; automatic motion selection suggested {trajectory.events['automatic_object_name']} (review other-object tracks).")
    print(f"Pick frame: {trajectory.events['pick_frame']} | place frame: {trajectory.events['place_frame']}")
    print(f"Placement settled: {trajectory.events['placement_settled']}")
    print(f"Max displacement: {trajectory.events['max_displacement_m']:.3f} m")
    print(f"Waypoints: {len(trajectory.waypoints)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
