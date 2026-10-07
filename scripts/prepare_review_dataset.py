#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.dataset_review import checked_replay_task, load_review_manifest
from humanoid_retarget.io import list_rgb_frames, read_json, read_yaml, write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.physical import build_physical_plan
from humanoid_retarget.sam2_dataset import capture_issues


def prepare(args) -> Path:
    selection = read_json(args.selection)
    selected = selection["retained_clips"]
    counts = Counter(row["object_name"] for row in selected)
    if len(selected) != args.expected_count or len({r["episode"] for r in selected}) != len(selected):
        raise ValueError("Selection does not contain the expected number of unique clips")
    if args.expected_count == 30 and counts != Counter(black_box=10, blue_sphere=10, yellow_mustard_tin=10):
        raise ValueError("The 30-clip collection must contain ten clips per object")
    directory = args.out.resolve()
    if (not directory.is_relative_to(ROOT) or directory.is_relative_to(ROOT / "data")
            or (directory.exists() and any(directory.iterdir()))):
        raise ValueError("Choose a fresh repository review output directory outside source data")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "selection.json", selection)
    rows = []
    for index, selected_row in enumerate(selected, start=1):
        episode = (ROOT / selected_row["episode"]).resolve()
        source_task = ROOT / selected_row["sam2_trajectory"]
        source_labels = ROOT / selected_row["sam2_position_label_candidates"]
        metadata = read_json(episode / "metadata.json")
        if capture_issues(episode, metadata):
            raise ValueError(f"Incomplete capture: {episode}")
        task, labels = read_json(source_task), read_json(source_labels)
        if task["object_name"] != selected_row["object_name"] or Path(task["source_episode"]).resolve() != episode:
            raise ValueError("Trajectory identity/source mismatch")
        raw_path = source_task.with_name("raw_detections.json")
        raw = read_json(raw_path)
        mask_path = Path(raw["metadata"]["mask_manifest"])
        mask_manifest = read_json(mask_path)
        if mask_manifest.get("completed") is not True or mask_manifest["object_name"] != task["object_name"]:
            raise ValueError("Mask sequence is incomplete or belongs to another object")
        if (file_sha256(raw_path) != labels["raw_detections_sha256"]
                or file_sha256(mask_path) != labels["mask_manifest_sha256"]):
            raise ValueError("Candidate positions are not bound to the current tracks/masks")
        frames = list_rgb_frames(episode)
        if len(frames) != len(labels["frames"]) or len(frames) != len(mask_manifest["frames"]):
            raise ValueError("RGB/mask/label frame count mismatch")
        fingerprints = {}

        def bind(path: Path, expected: str | None = None) -> str:
            path = path.resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError(f"Dataset reference outside repository: {path}")
            actual = file_sha256(path)
            if expected is not None and actual != expected:
                raise ValueError(f"Source fingerprint mismatch: {path}")
            relative = str(path.relative_to(ROOT))
            fingerprints[relative] = actual
            return relative

        for frame_index, (rgb, entry, diagnostic, label) in enumerate(zip(
                frames, mask_manifest["frames"], raw["metadata"]["frame_diagnostics"], labels["frames"])):
            if entry["frame_idx"] != frame_index or entry["rgb_name"] != rgb.name:
                raise ValueError("Source/mask frame order mismatch")
            if not np.isclose(label["capture_t_s"], metadata["timestamps_relative_s"][frame_index], atol=1e-6):
                raise ValueError("Candidate label timestamp is not the capture timestamp")
            bind(rgb, entry["rgb_sha256"])
            bind(episode / "depth" / f"{rgb.stem}.png", diagnostic["depth_sha256"])
            bind(mask_path.parent / f"{frame_index:06d}.png", entry["mask_sha256"])
        checked = checked_replay_task(task, labels)
        build_physical_plan(checked)
        task_path = directory / "tasks" / f"{index:03d}_{selected_row['object_name']}.json"
        write_json(task_path, checked)
        quality_path = source_task.parent.parent / "preparation_quality.json"
        quality = read_json(quality_path)
        preview_path = source_task.with_name("detections.mp4")
        cap = cv2.VideoCapture(str(preview_path))
        try:
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames):
                raise ValueError("Annotated video is missing or has an incorrect frame count")
            for frame_index in (0, len(frames) // 2, len(frames) - 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = cap.read()
                if not ok or np.std(image) < 10:
                    raise ValueError("Annotated video has an undecodable/blank review sample")
        finally:
            cap.release()
        calibration_path = ROOT / selected_row.get("sam2_workspace_config", "configs/workspace.real.yaml")
        calibration = read_yaml(calibration_path)
        centres = calibration["workspace"]["calibration_reference"]["diagnostics"]["marker_centres_desk_m"]
        distance = float(np.linalg.norm(np.asarray(centres[0]) - centres[1]) * 1000)
        bind(calibration_path, raw["metadata"]["workspace_config_sha256"])
        for path in (episode / "metadata.json", raw_path, source_task, source_labels, mask_path,
                     quality_path, source_task.with_name("detections.json"), preview_path,
                     source_task.parent.parent / "review.jpg"):
            bind(path)
        rows.append({"index": index, "episode": selected_row["episode"], "object_name": selected_row["object_name"],
                     "capture_session": metadata.get("batch_id") or "earlier_singles_setup_unconfirmed",
                     "trajectory": bind(task_path), "candidate_labels": str(source_labels.relative_to(ROOT)),
                     "annotated_video": str(preview_path.relative_to(ROOT)),
                     "frames": len(frames), "candidate_frames": quality["position_label_candidate_frames"],
                     "gated_frames": quality["gated_frames"], "training_ready": False,
                     "calibration": {"config": str(calibration_path.relative_to(ROOT)), "marker_0_1_mm": distance,
                                     "inside_ruler_interval": args.distance_mm[0] <= distance <= args.distance_mm[1]},
                     "file_sha256": fingerprints})
        print(f"CHECKED {index}/{len(selected)} {selected_row['episode']}", flush=True)
    manifest = {"schema_version": "hmnd_prepared_review_dataset_v1", "path_base": "repository_root",
                "ready_for_isaac_review": True, "training_ready": False, "clip_count": len(rows),
                "object_counts": dict(counts), "selection_snapshot": str((directory / "selection.json").relative_to(ROOT)),
                "ruler_check": {"marker_ids": [0, 1], "distance_interval_mm": args.distance_mm,
                                "setup_unchanged_confirmed": False,
                                "notes": "Coarse operator measurement, not an exact scale correction. Saved calibrations underestimate this interval; absolute accuracy remains unverified. No rescale applied."},
                "split_status": "not_assigned; split whole clips and group correlated capture setups before perception training",
                "notes": "Frozen review tasks gate unsupported measured guidance. Control interpolation is not teacher ground truth. Only the selected object is simulated, with fixed orientation and assumed mass/friction.",
                "clips": rows}
    path = directory / "manifest.json"
    write_json(path, manifest)
    load_review_manifest(path, ROOT, verify=False)
    lines = ["# Prepared 30-Clip Review Dataset", "", f"Ready for Isaac review: {len(rows)} unique demonstrations. Training-ready: false.",
             f"Object counts: {dict(counts)}. Source frames: {sum(r['frames'] for r in rows)}.", "",
             "Source RGB/depth/mask hashes and sampled annotated videos checked. Unsupported guidance frames are excluded from the frozen replay tasks, not relabelled through interpolation.", "",
             "Ruler interval: 570-580 mm between marker centres 0 and 1. Saved models estimate approximately 560-566 mm. This does not validate millimetre-level accuracy; no forced rescale or original calibration overwrite was performed.", "",
             "No final perception split or teacher-label approval has been assigned. All 30 episodes can be used for controller review, but reserve grouped held-out clips before fitting a perception model.", "",
             "| Index | Clip | Object | Candidate / Total | Gated |", "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {r['index']} | {r['episode']} | {r['object_name']} | {r['candidate_frames']} / {r['frames']} | {r['gated_frames']} |" for r in rows)
    (directory / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Freeze a fingerprint-checked dataset for physical Isaac/RGB review.")
    parser.add_argument("--selection", type=Path, default=ROOT / "data/real/rl_pilot/dataset_selection.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument("--distance-mm", type=float, nargs=2, default=[570.0, 580.0])
    args = parser.parse_args()
    if args.expected_count < 1 or not 0 < args.distance_mm[0] <= args.distance_mm[1] or not np.isfinite(args.distance_mm).all():
        parser.error("Use a positive count and finite positive distance interval")
    try:
        print(f"Manifest: {prepare(args)}")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Dataset not ready: {error}\n")


if __name__ == "__main__":
    main()
