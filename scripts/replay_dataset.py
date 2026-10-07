#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.dataset_review import load_review_manifest
from humanoid_retarget.io import read_json, write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.scene_context import prepare_scene_context


def main():
    parser = argparse.ArgumentParser(description="Sequential dynamic Isaac playback with the source RGB alongside.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/datasets/pilot_030/manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="Fresh result directory; use --resume after interruption")
    parser.add_argument("--indices", type=int, nargs="+", help="One-based subset, e.g. a three-object smoke check")
    parser.add_argument("--headless", action="store_true", help="Render comparison videos without the Isaac GUI")
    parser.add_argument("--no-video", action="store_true", help="Physics/metrics only; no source panel or camera")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--objects-config", type=Path, default=ROOT / "configs/objects.rl_pilot.yaml")
    parser.add_argument("--no-scene-context", action="store_true", help="Omit the calibrated markers and visual-only background references")
    parser.add_argument("--timeout-seconds", type=float, default=240)
    args = parser.parse_args()
    try:
        manifest = load_review_manifest(args.manifest, ROOT)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot replay dataset: {error}\n")
    chosen = set(args.indices or range(1, manifest["clip_count"] + 1))
    if not chosen or not chosen.issubset(range(1, manifest["clip_count"] + 1)) or args.timeout_seconds <= 0:
        parser.error("Indices must be within the manifest and timeout positive")
    rows = [r for r in manifest["clips"] if r["index"] in chosen]
    if args.dry_run:
        for row in rows:
            print(f"{row['index']:02d}/{manifest['clip_count']} {row['episode']} -> dynamic contact grasp")
        return
    directory = args.out.resolve()
    if directory.is_relative_to(ROOT / "data") or directory.is_relative_to(args.manifest.parent.resolve()):
        parser.error("Replay outputs must be outside recordings and the frozen dataset")
    state_path = directory / "run.json"
    identity = {"manifest_sha256": file_sha256(args.manifest), "indices": sorted(chosen),
                "headless": args.headless, "video": not args.no_video,
                "replay_script_sha256": file_sha256(ROOT / "scripts/replay_in_isaac_lab.py"),
                "physical_controller_sha256": file_sha256(ROOT / "src/humanoid_retarget/physical.py"),
                "scene_context": not args.no_scene_context,
                "scene_context_code_sha256": file_sha256(ROOT / "src/humanoid_retarget/scene_context.py"),
                "yaw_code_sha256": file_sha256(ROOT / "src/humanoid_retarget/yaw_tracking.py"),
                "objects_config_sha256": file_sha256(args.objects_config)}
    if args.resume:
        state = read_json(state_path)
        if state["identity"] != identity:
            parser.error("Manifest/controller/run settings changed; use a separate result directory")
    else:
        if directory.exists() and any(directory.iterdir()):
            parser.error("Choose a fresh output directory or --resume")
        state = {"identity": identity, "mode": "dynamic", "clips": [], "completed": False,
                 "notes": "Automatic review does not approve teacher labels or replace held-out evaluation."}
        write_json(state_path, state)
    completed = {r["index"] for r in state["clips"] if r.get("completed")}
    try:
        for row in rows:
            if row["index"] in completed:
                print(f"SKIP completed {row['index']:02d} {row['episode']}", flush=True)
                continue
            clip_dir = directory / f"{row['index']:03d}_{row['object_name']}"
            if clip_dir.exists() and any(clip_dir.iterdir()):
                attempts = 2
                while (directory / f"{clip_dir.name}_attempt_{attempts}").exists():
                    attempts += 1
                clip_dir = directory / f"{clip_dir.name}_attempt_{attempts}"
            clip_dir.mkdir(parents=True)
            command = ["bash", str(ROOT / "scripts/run_physical_grasp.sh"), "--trajectory", str(ROOT / row["trajectory"]),
                       "--report", str(clip_dir / "metrics.json"), "--review-title", f"{row['index']:02d}/{manifest['clip_count']} {row['object_name']}"]
            if args.headless:
                command.append("--headless")
            if not args.no_scene_context:
                context_path = clip_dir / "scene_context.json"
                context = prepare_scene_context(ROOT / row["trajectory"], ROOT / row["calibration"]["config"],
                                                args.objects_config, context_path)
                print(f"SCENE {len(context['markers'])} markers, {len(context['background_objects'])} static visual objects", flush=True)
                command.extend(["--scene-context", str(context_path)])
            if not args.no_video:
                command.extend(["--video", str(clip_dir / "comparison.mp4"), "--reference-episode", str(ROOT / row["episode"])])
            print(f"START {row['index']:02d}/{manifest['clip_count']} {row['episode']}", flush=True)
            with (clip_dir / "isaac.log").open("w") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds, env=os.environ.copy())
            metrics_path = clip_dir / "metrics.json"
            metrics = read_json(metrics_path) if metrics_path.is_file() else {}
            result = {"index": row["index"], "episode": row["episode"], "directory": str(clip_dir), "exit_code": process.returncode,
                      "completed": process.returncode == 0 and metrics.get("completed", False),
                      "pick_place_passed": metrics.get("pick_place_passed", False),
                      "tracking_passed": metrics.get("tracking_passed", False),
                      "physical_results": metrics.get("physical_results", [])}
            if "pick_place_with_yaw_passed" in metrics:
                result["pick_place_with_yaw_passed"] = metrics["pick_place_with_yaw_passed"]
            state["clips"] = [r for r in state["clips"] if r["index"] != row["index"]] + [result]
            write_json(state_path, state)
            print(f"DONE {row['index']:02d} completed={result['completed']} pick/place={result['pick_place_passed']} tracking={result['tracking_passed']}", flush=True)
            if not result["completed"]:
                parser.exit(1, f"Replay incomplete; inspect {clip_dir / 'isaac.log'} then resume.\n")
        state["completed"] = True
        state["pick_place_passed_count"] = sum(r["pick_place_passed"] for r in state["clips"])
        write_json(state_path, state)
        print(f"Review complete: {len(rows)} episodes; pick/place passed {state['pick_place_passed_count']}/{len(rows)}. Results: {state_path}")
    except (ValueError, OSError) as error:
        write_json(state_path, state)
        parser.exit(1, f"Scene/replay error: {error}. Artifacts preserved.\n")
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        write_json(state_path, state)
        parser.exit(1, "Review interrupted/timed out. Artifacts preserved; use --resume with identical settings.\n")


if __name__ == "__main__":
    main()
