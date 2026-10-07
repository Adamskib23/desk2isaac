#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from humanoid_retarget.io import read_json, read_yaml, write_json
from humanoid_retarget.recording import CaptureTimer
from humanoid_retarget.tracking import object_specs_from_config
from record_orbbec_episode import record_opencv, record_orbbec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def recording_settings(args: argparse.Namespace) -> dict:
    return {name: getattr(args, name) for name in (
        "count", "seconds", "fps", "backend", "camera_index", "countdown",
        "initial_hold_seconds", "final_hold_seconds", "beep",
    )}


def make_timer(args: argparse.Namespace) -> CaptureTimer:
    return CaptureTimer(
        args.seconds, guided=True, countdown_seconds=args.countdown,
        initial_hold_seconds=args.initial_hold_seconds,
        final_hold_seconds=args.final_hold_seconds, beep=args.beep,
    )


def choose_object(names: list[str], clip_number: int, count: int) -> str | None:
    print(f"\nCLIP {clip_number}/{count} | Prepare the start position; move only the selected object.", flush=True)
    for index, name in enumerate(names, 1):
        print(f"  {index}: {name}", flush=True)
    while True:
        try:
            choice = input("Object number/name, or q to finish: ").strip()
        except EOFError:
            return None
        if choice.lower() in ("q", "quit"):
            return None
        if choice in names:
            return choice
        if choice.isdecimal() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Choose a listed object number/name, or q. Recording has not started.", flush=True)


def annotate_episode(episode: Path, manifest: dict, entry: dict) -> dict | None:
    metadata_path = episode / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    label = metadata.get("action_object")
    if label is not None and label != entry["object_name"]:
        raise ValueError(f"Episode label disagrees with the manifest in {episode}; it was not overwritten")
    batch_info = {
        "batch_id": manifest["batch_id"], "clip_index": entry["clip_index"],
        "manifest": "../batch.json", "label_source": "operator_declared",
    }
    if metadata.get("recording_batch") not in (None, batch_info):
        raise ValueError(f"Episode batch annotation disagrees with the manifest in {episode}")
    if label != entry["object_name"] or metadata.get("recording_batch") != batch_info:
        metadata["action_object"] = entry["object_name"]
        metadata["recording_batch"] = batch_info
        write_atomic_json(metadata_path, metadata)
    return metadata


def completed_capture(episode: Path, metadata: dict | None, backend: str) -> bool:
    if (metadata is None or metadata.get("interrupted") is not False
            or metadata.get("recording_guidance", {}).get("completed") is not True):
        return False
    rgb = {path.stem for path in (episode / "rgb").glob("*.jpg")}
    timestamps = metadata.get("timestamps_relative_s", [])
    if (len(rgb) < 3 or len(rgb) != len(timestamps)
            or not all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in timestamps)
            or not all(b > a for a, b in zip(timestamps, timestamps[1:]))):
        return False
    seconds = metadata["recording_guidance"].get("requested_seconds")
    fps = metadata.get("requested_fps", metadata.get("fps", 30))
    if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
            or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
            or timestamps[-1] < seconds - max(0.25, 2 / fps)):
        return False
    if backend == "orbbec":
        depth = {path.stem for path in (episode / "depth").glob("*.png")}
        return rgb == depth and metadata.get("depth_aligned_to") == "color"
    return True


def prepare_manifest(out: Path, args: argparse.Namespace, objects: dict) -> dict:
    path = out / "batch.json"
    if args.resume:
        if not path.is_file():
            raise ValueError("--resume requires an existing batch.json")
        manifest = read_json(path)
        if manifest.get("schema_version") != "hmnd_rgbd_batch_v1":
            raise ValueError("Unrecognised batch manifest schema")
        if manifest["recording_settings"] != recording_settings(args) or manifest["objects"] != objects["objects"]:
            raise ValueError("Resume with the original recording settings and object definitions")
        names = {obj["name"] for obj in manifest["objects"]}
        indices, episodes = set(), set()
        for entry in manifest["clips"]:
            if (not isinstance(entry["clip_index"], int) or entry["clip_index"] < 1
                    or entry["clip_index"] in indices or entry["episode"] in episodes
                    or Path(entry["episode"]).name != entry["episode"] or entry["episode"] in ("", ".", "..")
                    or entry["object_name"] not in names
                    or entry["status"] not in ("recording", "completed", "interrupted", "incomplete", "failed", "rejected")):
                raise ValueError("Invalid or duplicated clip entry in batch manifest")
            indices.add(entry["clip_index"])
            episodes.add(entry["episode"])
        retake = None
        if args.retake_clip is not None:
            retake = next((entry for entry in manifest["clips"] if entry["clip_index"] == args.retake_clip), None)
            if retake is None or retake["status"] not in ("completed", "rejected"):
                raise ValueError("--retake-clip must name a completed or already rejected clip index")
        for entry in manifest["clips"]:
            if entry is retake or entry["status"] not in ("recording", "completed"):
                continue
            episode = out / entry["episode"]
            metadata = annotate_episode(episode, manifest, entry)
            if completed_capture(episode, metadata, args.backend):
                entry["status"] = "completed"
                entry["frames"] = len(metadata["timestamps_relative_s"])
            elif entry["status"] == "completed":
                raise ValueError(f"Completed clip {episode} is incomplete or missing; original files were not changed")
            else:
                entry["status"] = "incomplete"
        if retake is not None:
            if retake["status"] != "rejected":
                retake["status"] = "rejected"
                retake["rejected_at_utc"] = utc_now()
                retake["rejection_reason"] = "operator_requested_retake"
            print(f"RETAKE | {retake['episode']} excluded from the count; original files preserved. "
                  f"Choose {retake['object_name']} at the next prompt to replace it.", flush=True)
        write_atomic_json(path, manifest)
        return manifest
    if any(item.name != ".batch.lock" for item in out.iterdir()):
        raise ValueError(f"Batch directory {out} is not empty; use --resume or choose a fresh --out")
    manifest = {
        "schema_version": "hmnd_rgbd_batch_v1", "batch_id": uuid.uuid4().hex,
        "created_at_utc": utc_now(), "objects_config": str(Path(args.objects_config).resolve()),
        "objects": objects["objects"], "recording_settings": recording_settings(args), "clips": [],
        "notes": "Completed means capture finished, not tracking or physical-grasp validation. One action object per clip.",
    }
    write_atomic_json(path, manifest)
    return manifest


def run_batch(args: argparse.Namespace) -> int:
    if args.retake_clip is not None and (not args.resume or args.retake_clip < 1):
        raise ValueError("--retake-clip requires --resume and a positive clip index")
    if args.count < 1 or not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--count must be positive and --fps finite and positive")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("--min-free-gb must be finite and non-negative")
    make_timer(args)
    objects = read_yaml(args.objects_config)
    names = [spec.name for spec in object_specs_from_config(objects)]
    if len(set(names)) != len(names) or any(not name or Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError("Object names must be unique, non-empty directory-safe names")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another recorder is using this batch directory") from exc
        manifest = prepare_manifest(out, args, objects)
        manifest_path = out / "batch.json"
        print(f"Batch: {out}\nEach clip: countdown, initial HOLD, MOVE, final HOLD. Ctrl+C stops safely.", flush=True)
        try:
            while sum(entry["status"] == "completed" for entry in manifest["clips"]) < args.count:
                completed = sum(entry["status"] == "completed" for entry in manifest["clips"])
                if shutil.disk_usage(out).free < args.min_free_gb * 1024 ** 3:
                    raise ValueError(f"Less than {args.min_free_gb:g} GiB free; stopped before the next clip")
                object_name = choose_object(names, completed + 1, args.count)
                if object_name is None:
                    print(f"Stopped with {completed}/{args.count} captures. Rerun with --resume to continue.", flush=True)
                    return 0
                index = max((entry["clip_index"] for entry in manifest["clips"]), default=0) + 1
                episode = out / f"clip_{index:03d}_{object_name}"
                entry = {
                    "clip_index": index, "object_name": object_name, "episode": episode.name,
                    "status": "recording", "started_at_utc": utc_now(),
                }
                manifest["clips"].append(entry)
                write_atomic_json(manifest_path, manifest)
                print(f"Recording {object_name} -> {episode.name}", flush=True)
                created = False
                try:
                    episode.mkdir(exist_ok=False)
                    created = True
                    timer = make_timer(args)
                    if args.backend == "orbbec":
                        record_orbbec(episode, args.seconds, args.fps, timer=timer)
                    else:
                        record_opencv(episode, args.seconds, args.fps, args.camera_index, timer=timer)
                    metadata = annotate_episode(episode, manifest, entry)
                    if not completed_capture(episode, metadata, args.backend):
                        raise RuntimeError("Capture is incomplete or has unmatched frames; it was preserved, not counted")
                    entry["status"] = "completed"
                    entry["frames"] = len(metadata["timestamps_relative_s"])
                except (Exception, KeyboardInterrupt) as exc:
                    entry["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    entry["error"] = str(exc) or "KeyboardInterrupt"
                    if created:
                        annotate_episode(episode, manifest, entry)
                    raise
                finally:
                    entry["finished_at_utc"] = utc_now()
                    write_atomic_json(manifest_path, manifest)
                print(f"SAVED {completed + 1}/{args.count} | {object_name} | tracking not yet reviewed", flush=True)
        except KeyboardInterrupt:
            print("Stopped safely. Completed and partial clips were preserved; use --resume to continue.", flush=True)
            return 130
        print(f"BATCH DONE | {args.count} captures saved. Manifest: {manifest_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Record labelled, guided single-object clips back-to-back.")
    parser.add_argument("--out", required=True, help="Batch directory; keep existing single clips separate.")
    parser.add_argument("--count", type=int, default=10, help="Total completed captures in this batch, including resumed clips.")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--objects-config", default=str(ROOT / "configs/objects.rl_pilot.yaml"))
    parser.add_argument("--backend", choices=("orbbec", "opencv"), default="orbbec")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--initial-hold-seconds", type=float, default=2.0)
    parser.add_argument("--final-hold-seconds", type=float, default=2.0)
    parser.add_argument("--beep", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an existing batch; never overwrite incomplete clips.")
    parser.add_argument("--retake-clip", type=int, help="With --resume, reject this clip index so a fresh attempt can replace it; preserve original files.")
    parser.add_argument("--min-free-gb", type=float, default=1.0, help="Minimum free GiB checked before each clip.")
    args = parser.parse_args()
    try:
        return run_batch(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Batch stopped: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
