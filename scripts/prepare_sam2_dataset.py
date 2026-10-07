#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.calibration import calibration_from_config
from humanoid_retarget.io import list_rgb_frames, read_depth_m, read_episode_metadata, read_json, read_yaml, write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.sam2_dataset import (
    capture_issues, image_boundary_frames, label_candidates, propose_prompt, review_sheet,
)
from humanoid_retarget.tracking import load_object_specs


def validate_plan_locations(plan: dict, path: Path) -> None:
    directory, source = Path(plan["output_dir"]).resolve(), Path(plan["source"]).resolve()
    if path.parent.resolve() != directory or directory.is_relative_to(source):
        raise ValueError("Plan must remain in its output directory, outside the source recordings")
    for row in plan["clips"]:
        episode, output = Path(row["episode"]).resolve(), Path(row["output_dir"]).resolve()
        if not episode.is_relative_to(source) or not output.is_relative_to(directory) or output == directory:
            raise ValueError("Episode/output paths escaped their planned roots")
        if output.is_relative_to(episode):
            raise ValueError("Derived outputs must not modify source episodes")


def make_plan(args) -> Path:
    directory = args.out.resolve()
    if directory.is_relative_to(args.source.resolve()) or (directory.exists() and any(directory.iterdir())):
        raise ValueError("Choose a fresh output directory outside the source recordings")
    selection_path = args.selection or args.source / "dataset_selection.json"
    selection = read_json(selection_path)
    retained = {Path(r["episode"]).resolve() for r in selection["retained_clips"]}
    prior = {Path(r["episode"]).resolve(): r for r in selection["excluded_clips"]}
    specs = {s.name: s for s in load_object_specs(args.objects_config)}
    calibration = calibration_from_config(read_yaml(args.workspace_config))
    batch_status = {}
    for path in args.source.glob("batch_*/batch.json"):
        for entry in read_json(path)["clips"]:
            batch_status[(path.parent / entry["episode"]).resolve()] = entry["status"]
    directory.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for metadata_path in sorted(args.source.rglob("metadata.json")):
        episode = metadata_path.parent.resolve()
        relative = episode.relative_to(args.source.resolve())
        metadata = read_episode_metadata(episode)
        previous = prior.get(episode, {})
        row = {"episode": str(episode), "relative_episode": str(relative), "training_ready": False,
               "prior_retained": episode in retained, "prior_reason": previous.get("reason"),
               "source_metadata_sha256": file_sha256(metadata_path), "status": "pending",
               "output_dir": str(directory / relative), "capture_issues": capture_issues(episode, metadata)}
        name = metadata.get("action_object")
        if name is None:
            historical = ROOT / "outputs/rl_pilot" / relative / "trajectory.json"
            if historical.is_file():
                name = read_json(historical)["object_name"]
                row["label_source"] = str(historical)
        row["object_name"] = name
        skip = ("Failed first capture" in previous.get("reason", "")
                or "Static setup check" in previous.get("reason", "")
                or batch_status.get(episode) == "rejected" or bool(row["capture_issues"]))
        if skip or name not in specs:
            row.update(status="skipped", skip_reason=previous.get("reason") or "; ".join(row["capture_issues"]) or "No supported object label")
            rows.append(row)
            continue
        frames = list_rgb_frames(episode)
        image = cv2.imread(str(frames[0]))
        depth = read_depth_m(episode, frames[0].stem, metadata.get("depth_unit_m", 0.001))
        if image is None or depth is None:
            row.update(status="needs_prompt", error="Cannot read initial RGB-D")
            rows.append(row)
            continue
        existing = ROOT / "outputs/sam2" / relative / "masks"
        manifest_path = existing / "manifest.json"
        if manifest_path.is_file() and read_json(manifest_path).get("completed") is True:
            manifest = read_json(manifest_path)
            if manifest["object_name"] != name or Path(manifest["source_episode"]).resolve() != episode:
                raise ValueError(f"Existing mask identity mismatch for {episode}")
            row["existing_masks"] = str(existing)
            box = manifest["prompt"]["box_xyxy_px"]
            row["prompt_frame"] = manifest["prompt"]["frame_idx"]
            image = cv2.imread(str(frames[row["prompt_frame"]]))
        else:
            try:
                box = propose_prompt(image, depth, specs[name], calibration)
            except ValueError as error:
                row.update(status="needs_prompt", error=str(error))
                rows.append(row)
                continue
            row["prompt_frame"] = 0
        row.update(box_xyxy_px=box, prompt_rgb_sha256=file_sha256(frames[row["prompt_frame"]]), num_frames=len(frames))
        left, top, right, bottom = [int(round(v)) for v in box]
        preview = image.copy()
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 180, 255), 3)
        cx, cy = (left + right) // 2, (top + bottom) // 2
        x, y = max(0, min(cx - 200, image.shape[1] - 400)), max(0, min(cy - 120, image.shape[0] - 240))
        tile = cv2.resize(preview[y:y + 240, x:x + 400], (360, 216))
        tile = cv2.copyMakeBorder(tile, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(tile, str(relative), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(tile, name, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (245, 245, 245), 1, cv2.LINE_AA)
        tiles.append(tile)
        rows.append(row)
    for page, offset in enumerate(range(0, len(tiles), 12), start=1):
        group = tiles[offset:offset + 12]
        while len(group) % 4:
            group.append(np.zeros_like(group[0]))
        sheet = np.vstack([np.hstack(group[i:i + 4]) for i in range(0, len(group), 4)])
        if not cv2.imwrite(str(directory / f"prompt_review_{page:02d}.jpg"), sheet):
            raise OSError("Cannot write prompt review")
    payload = {"schema_version": "hmnd_sam2_dataset_preparation_v1", "training_ready": False,
               "source": str(args.source.resolve()), "output_dir": str(directory),
               "workspace_config": str(args.workspace_config.resolve()), "objects_config": str(args.objects_config.resolve()),
               "workspace_config_sha256": file_sha256(args.workspace_config),
               "objects_config_sha256": file_sha256(args.objects_config),
               "selection_path": str(selection_path.resolve()), "selection_sha256": file_sha256(selection_path),
               "source_count": len(rows), "clips": rows, "notes": "Prompts require visual review; processing is not training approval. Original calibration retained for diagnostic comparison."}
    path = directory / "plan.json"
    write_json(path, payload)
    print(f"Inventoried {len(rows)}: {dict(Counter(r['status'] for r in rows))}. Review prompt_review_*.jpg then execute {path}")
    return path


def process_clip(row: dict, plan: dict, python: Path, preview_width: int) -> None:
    episode, directory = Path(row["episode"]), Path(row["output_dir"])
    if file_sha256(episode / "metadata.json") != row["source_metadata_sha256"]:
        raise ValueError("Source metadata changed after planning")
    frames = list_rgb_frames(episode)
    if file_sha256(frames[row["prompt_frame"]]) != row["prompt_rgb_sha256"]:
        raise ValueError("Prompt source changed after planning")
    directory.mkdir(parents=True, exist_ok=True)
    masks = Path(row["existing_masks"]) if row.get("existing_masks") else directory / "masks"
    manifest = masks / "manifest.json"
    if not manifest.is_file() or read_json(manifest).get("completed") is not True:
        command = [str(python), str(ROOT / "scripts/segment_sam2_episode.py"), "--episode", str(episode),
                   "--object-name", row["object_name"], "--frame", str(row["prompt_frame"]),
                   "--box", *[str(v) for v in row["box_xyxy_px"]], "--out", str(masks)]
        with (directory / "segmentation.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    tracking = directory / "tracking"
    quality_path = tracking / "tracking_quality.json"
    if not quality_path.is_file():
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/build_trajectory.py"), "--episode", str(episode),
                   "--object-name", row["object_name"], "--mask-dir", str(masks), "--workspace-config", plan["workspace_config"],
                   "--objects-config", plan["objects_config"], "--preview-width", str(preview_width), "--out", str(tracking / "trajectory.json")]
        with (directory / "tracking.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode not in (0, 2) or not quality_path.is_file():
            raise RuntimeError(f"Tracking failed; inspect {directory / 'tracking.log'}")
    raw, filtered, quality = (read_json(tracking / name) for name in ("raw_detections.json", "detections.json", "tracking_quality.json"))
    metadata = raw["metadata"]
    if (metadata["mask_manifest_sha256"] != file_sha256(manifest)
            or metadata["workspace_config_sha256"] != plan["workspace_config_sha256"]
            or metadata["objects_config_sha256"] != plan["objects_config_sha256"]):
        raise ValueError("Cached tracking fingerprints changed; use a separate tracking run")
    entries = read_json(manifest)["frames"]
    if len(entries) != len(frames):
        raise ValueError("Mask/source frame count changed")
    for index, (source, entry) in enumerate(zip(frames, entries)):
        if (entry["frame_idx"] != index or entry["rgb_name"] != source.name
                or file_sha256(source) != entry["rgb_sha256"]
                or file_sha256(masks / f"{index:06d}.png") != entry["mask_sha256"]):
            raise ValueError(f"Cached mask/RGB fingerprint mismatch at frame {index}")
        depth = episode / "depth" / f"{source.stem}.png"
        if file_sha256(depth) != metadata["frame_diagnostics"][index]["depth_sha256"]:
            raise ValueError(f"Cached depth fingerprint mismatch at frame {index}")
    clipped = image_boundary_frames(masks, quality["frames"])
    labels, summary = label_candidates(raw, filtered, quality, clipped)
    write_json(directory / "position_label_candidates.json", {"training_ready": False, "episode": str(episode),
               "raw_detections_sha256": file_sha256(tracking / "raw_detections.json"),
               "mask_manifest_sha256": metadata["mask_manifest_sha256"],
               "notes": "Quality-gated candidate positions, not approved ground truth or robot policy actions. Never use predicted/interpolated frames as observed labels.", "frames": labels})
    write_json(directory / "preparation_quality.json", summary)
    review_sheet(episode, masks, raw, summary, directory / "review.jpg")
    row.update(status="prepared", masks=str(masks), quality=summary, training_ready=False)


def write_report(plan: dict) -> None:
    rows = plan["clips"]
    prepared = [r for r in rows if r["status"] == "prepared"]
    continuous = [r for r in prepared if r["quality"]["continuous_supported_track"] and r["quality"]["task_generated"]]
    lines = ["# SAM Dataset Preparation", "", "Diagnostic preparation only. Training approval and calibration validation remain pending.", "",
             f"Inventoried: {len(rows)}. Prepared: {len(prepared)}. Continuous supported task tracks: {len(continuous)}.",
             f"Previously retained: {sum(r['prior_retained'] for r in rows)}. Training-ready labels: 0.", "",
             "| Clip | Object | State | Candidate Frames | Gated | Initial / Final p95 mm | Previous Review (Before SAM) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        q = row.get("quality", {})
        holds = [q.get(key) for key in ("initial_hold", "final_hold")]
        h = " / ".join(f"{v['p95_deviation_mm']:.2f}" if v else "-" for v in holds)
        link = f"[{row['relative_episode']}]({row['relative_episode']}/review.jpg)" if row["status"] == "prepared" else row["relative_episode"]
        note = row.get("error") or row.get("skip_reason") or row.get("prior_reason") or "Previously retained; re-review new labels"
        lines.append(f"| {link} | {row['object_name']} | {row['status']} | {q.get('position_label_candidate_frames', '-')} / {q.get('frames', '-')} | {q.get('gated_frames', '-')} | {h} | {note} |")
    lines.extend(["", "Full-resolution masks and source RGB/depth are unchanged by preview scaling. Annotated videos are under each clip's tracking/ folder.",
                  "A continuous supported track is a continuity check, not proof of correct object identity, hand release or absolute accuracy.",
                  "Retain original operator rejections, cropped/occluded demonstrations and late-release warnings until explicit review. Do not count this report as dataset approval.",
                  "Position candidates gate missing/weak/predicted frames, image-edge masks, adjacent steps >30 mm, and estimator switches with height changes >6 mm. These are review heuristics, not measured error.", ""])
    if (Path(plan["output_dir"]) / "quality.md").is_file():
        lines.extend(["See the separate [collection review](quality.md) for current retention decisions. The table's previous-review notes are historical, not judgments of the new SAM tracks.", ""])
    (Path(plan["output_dir"]) / "preparation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan, review and batch-prepare saved pilot episodes using SAM masks.")
    parser.add_argument("--source", type=Path, default=ROOT / "data/real/rl_pilot")
    parser.add_argument("--selection", type=Path, help="Selection manifest outside --source, for preparing one batch")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workspace-config", type=Path, default=ROOT / "configs/workspace.real.yaml")
    parser.add_argument("--objects-config", type=Path, default=ROOT / "configs/objects.rl_pilot.yaml")
    parser.add_argument("--execute", type=Path, help="Existing plan.json; completed clips are skipped on resume")
    parser.add_argument("--reviewed-prompts", action="store_true", help="Confirm the prompt contact sheets were visually reviewed")
    parser.add_argument("--sam-python", type=Path, default=ROOT / ".venv-sam2/bin/python")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--refresh-reviews", action="store_true", help="Recompute derived candidate labels/reviews from completed masks/tracks")
    args = parser.parse_args()
    if args.execute is None:
        if args.out is None:
            parser.error("Planning requires a fresh --out")
        make_plan(args)
        return
    if not args.reviewed_prompts or args.preview_width <= 0:
        parser.error("Review prompts first and use a positive preview width")
    plan = read_json(args.execute)
    if plan.get("schema_version") != "hmnd_sam2_dataset_preparation_v1":
        parser.error("Unknown dataset plan schema")
    try:
        validate_plan_locations(plan, args.execute)
    except ValueError as error:
        parser.error(str(error))
    for path_key, hash_key in (("workspace_config", "workspace_config_sha256"), ("objects_config", "objects_config_sha256"),
                               ("selection_path", "selection_sha256")):
        if file_sha256(plan[path_key]) != plan[hash_key]:
            parser.error(f"{path_key} changed after planning; preserve this run and create a new plan")
    pending = [r for r in plan["clips"] if r["status"] in ("pending", "failed", "processing")
               or (args.refresh_reviews and r["status"] == "prepared")]
    # Resolving a venv's Python symlink would silently select the base environment.
    sam_python = args.sam_python.absolute()
    if any(not row.get("existing_masks") and not (Path(row["output_dir"]) / "masks/manifest.json").is_file() for row in pending):
        subprocess.run([str(sam_python), "-c", "import torch, sam2; assert torch.cuda.is_available(), 'CUDA unavailable'"],
                       cwd=ROOT, check=True)
    for index, row in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] START {row['relative_episode']}", flush=True)
        if row.get("error"):
            row.setdefault("previous_errors", []).append(row.pop("error"))
        row["status"] = "processing"
        write_json(args.execute, plan)
        try:
            process_clip(row, plan, sam_python, args.preview_width)
            q = row["quality"]
            print(f"DONE {row['relative_episode']} | {q['position_label_candidate_frames']}/{q['frames']} candidate frames | task={q['task_generated']}", flush=True)
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
            row.update(status="failed", error=str(error))
            print(f"FAILED {row['relative_episode']} | {error}", flush=True)
        write_json(args.execute, plan)
        write_report(plan)
    write_report(plan)
    print(f"Report: {Path(plan['output_dir']) / 'preparation_report.md'}", flush=True)


if __name__ == "__main__":
    main()
