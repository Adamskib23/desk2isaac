#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.demo import bundled_path, load_demo
from humanoid_retarget.io import read_json, write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.rgb_data import load_training

RUNTIME_ROOTS = {".git", ".venv", ".venv-ml", ".pytest_cache", ".ruff_cache", "outputs",
                 "data", "models", "third_party", "development", "submission", "Log", "logs"}


def check_links(path: Path):
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        relative = target.split("#", 1)[0].strip("<>")
        if relative and not (path.parent / relative).exists():
            raise ValueError(f"Broken documentation link in {path.name}: {target}")


def included_files(root: Path):
    return {str(p.relative_to(root)): p for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).parts[0] not in RUNTIME_ROOTS
            and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".egg-info"))
            and not any(part.endswith(".egg-info") for part in p.relative_to(root).parts)}


def check_inventory(root: Path, *, strict: bool):
    path = root / "SUBMISSION_MANIFEST.json"
    if not path.exists():
        if strict:
            raise ValueError("Strict checking requires SUBMISSION_MANIFEST.json from a clean export")
        return
    inventory = read_json(path)
    if inventory.get("schema_version") != "hmnd_submission_files_v1":
        raise ValueError("Unsupported submission inventory")
    for relative, row in inventory["files"].items():
        source = bundled_path(root, relative)
        if file_sha256(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise ValueError(f"Exported file changed: {relative}")
        if row["bytes"] >= 50 * 1024 * 1024:
            raise ValueError(f"Large file in submission: {relative}")
    if strict:
        actual = set(included_files(root)) - {"SUBMISSION_MANIFEST.json"}
        if actual != set(inventory["files"]):
            raise ValueError(f"Export inventory differs: {sorted(actual.symmetric_difference(inventory['files']))}")


def check_media(root: Path):
    for path in (root / "docs/assets").rglob("*.png"):
        image = cv2.imread(str(path))
        if image is None or min(image.shape[:2]) < 100 or image.std() < 5:
            raise ValueError(f"Unreadable/blank evidence image: {path.name}")
    for path in (root / "docs/assets/videos").glob("*.mp4"):
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        try:
            for index in (0, count // 2, count - 1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame.std() < 5:
                    raise ValueError(f"Unreadable/blank evidence video: {path.name}")
                frames.append(cv2.resize(frame, (160, 90)).astype(float))
        finally:
            capture.release()
        if count < 10 or fps <= 0 or max(np.abs(frames[0] - f).mean() for f in frames[1:]) < 0.1:
            raise ValueError(f"Missing motion in evidence video: {path.name}")


def check(root: Path, *, strict=False, media=True, verify_inventory=True):
    demo = load_demo(root / "examples/retargeting/manifest.json")
    if {r["object_name"] for r in demo["clips"]} != {"black_box", "blue_sphere", "yellow_mustard_tin"}:
        raise ValueError("Expected one demo per evaluated object")
    manifest, inputs, labels = load_training(root / "examples/rgb_supervision/training_manifest.json")
    if Counter(r["split"] for r in inputs["samples"]) != {"train": 2202, "val": 805}:
        raise ValueError("RGB sample counts changed")
    if Counter(r["split"] for r in inputs["clips"]) != {"train": 22, "val": 8}:
        raise ValueError("RGB clip split changed")
    if len(labels) != 3007 or manifest["training_ready"] is not False:
        raise ValueError("Pseudo-label acknowledgement changed")
    summary = read_json(root / "evidence/robot/summary.json")
    if (summary["evaluated_clip_count"], summary["pick_place_passed"], summary["within_5_mm"],
            summary["pick_place_with_yaw_passed"]) != (30, 29, 20, 20):
        raise ValueError("Robot review summary changed")
    passed = 0
    for index in range(1, 31):
        report = read_json(root / f"evidence/robot/clips/{index:03d}.json")
        if not report["completed"] or report["object_pose_commands_during_run"] != 0:
            raise ValueError("Robot evidence is not completed free-object dynamics")
        passed += report["pick_place_passed"]
    if passed != summary["pick_place_passed"]:
        raise ValueError("Robot reports disagree with summary")
    for name in ("rgb_short", "rgb_long"):
        report = read_json(root / "evidence" / name / "report.json")
        if report["held_out_test_completed"] or report["sample_counts"] != {"train": 2202, "val": 805}:
            raise ValueError("RGB validation/test status changed")
    for path in [root / "README.md", root / "THIRD_PARTY.md", root / "DATA_TERMS.md", *sorted((root / "docs").glob("*.md"))]:
        check_links(path)
    for folder in ("docs", "examples", "evidence", "configs"):
        for path in (root / folder).rglob("*"):
            if path.suffix in {".md", ".json", ".yaml"}:
                if re.search(r"/home/[^/\s]+/", path.read_text(encoding="utf-8")):
                    raise ValueError(f"Machine-specific path remains: {path.relative_to(root)}")
    if verify_inventory:
        check_inventory(root, strict=strict)
    if media:
        check_media(root)
    return {"robot_examples": len(demo["clips"]), "rgb_frames": len(labels), "robot_review": 30}


def main():
    parser = argparse.ArgumentParser(description="Check portable submission files without Torch, a camera or Isaac Lab")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--strict-files", action="store_true")
    parser.add_argument("--skip-media", action="store_true")
    parser.add_argument("--refresh-inventory", action="store_true", help="Rebuild an existing export inventory after intentional edits")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        if args.refresh_inventory and not (root / "SUBMISSION_MANIFEST.json").exists():
            raise ValueError("Refresh the inventory from a clean export with an existing file manifest")
        counts = check(root, strict=args.strict_files, media=not args.skip_media,
                       verify_inventory=not args.refresh_inventory)
        if args.refresh_inventory:
            files = included_files(root)
            files.pop("SUBMISSION_MANIFEST.json", None)
            if any(p.stat().st_size >= 50 * 1024 * 1024 for p in files.values()):
                raise ValueError("File over 50 MiB in export")
            write_json(root / "SUBMISSION_MANIFEST.json", {"schema_version": "hmnd_submission_files_v1",
                       "files": {name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                                 for name, path in sorted(files.items())}})
            check_inventory(root, strict=True)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Submission check failed: {error}\n")
    print(f"Submission checks passed: {counts['robot_examples']} robot examples, "
          f"{counts['rgb_frames']} RGB frames, {counts['robot_review']} robot outcomes")


if __name__ == "__main__":
    main()
