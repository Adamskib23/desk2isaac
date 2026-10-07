#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.calibration import calibration_from_config
from humanoid_retarget.dataset_review import load_review_manifest
from humanoid_retarget.io import read_json, read_yaml, write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.rgb_data import (GEOMETRY_NAMES, calibration_snapshot, load_inputs,
                                       recording_group, rgb_observation, teacher_target, validate_split_groups)


def prepare(manifest_path: Path, directory: Path, validation_indices: list[int], stride: int) -> Path:
    directory = directory.resolve()
    if (not directory.is_relative_to(ROOT) or directory.is_relative_to(ROOT / "data")
            or directory.is_relative_to(manifest_path.parent.resolve()) or directory.exists()):
        raise ValueError("Choose a fresh repository output outside recordings/the frozen dataset")
    if stride < 1:
        raise ValueError("Sampling stride must be positive")
    manifest = load_review_manifest(manifest_path, ROOT)
    if not set(validation_indices).issubset({r["index"] for r in manifest["clips"]}):
        raise ValueError("Validation indices must exist in the frozen dataset")
    clips, samples, labels, calibrations = [], [], [], {}
    for row in manifest["clips"]:
        metadata = read_json(ROOT / row["episode"] / "metadata.json")
        clips.append({"index": row["index"], "episode": row["episode"], "object_name": row["object_name"],
                      "recording_group": recording_group(metadata, row["episode"]),
                      "split": "val" if row["index"] in validation_indices else "train"})
    validate_split_groups(clips)
    directory.mkdir(parents=True)
    for clip, source in zip(clips, manifest["clips"]):
        task = read_json(ROOT / source["trajectory"])
        if task.get("orientation", {}).get("schema_version") != "hmnd_upright_yaw_v1":
            raise ValueError("RGB training expects the separately frozen yaw-derived tasks")
        positions = read_json(ROOT / source["candidate_labels"])["frames"]
        yaw = {r["frame_idx"]: r for r in task["orientation"]["frames"]}
        calibration = calibration_from_config(read_yaml(ROOT / source["calibration"]["config"]))
        key = str(source["index"])
        calibrations[key] = calibration_snapshot(calibration)
        selected = set(range(0, len(positions), stride)) | {len(positions) - 1}
        count = 0
        for label in positions:
            index = label["frame_idx"]
            if index not in selected or not label["position_label_candidate"]:
                continue
            if not label["observed"] or label["exclusion_reasons"]:
                raise ValueError("Unsupported/predicted positions cannot supervise a student")
            rgb_path = ROOT / source["episode"] / "rgb" / f"{index:06d}.jpg"
            mask_path = Path(task["events"]["mask_manifest"]).parent / f"{index:06d}.png"
            image, mask = cv2.imread(str(rgb_path)), cv2.imread(str(mask_path), 0)
            crop, observation = rgb_observation(image, mask, task["object_name"], task["object"]["size_m"], calibration)
            sample_id = f"{source['index']:03d}_{index:06d}"
            relative = f"rgb/{sample_id}.jpg"
            output = directory / relative
            output.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(output), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError("Could not write RGB student crop")
            measured_yaw = yaw.get(index, {})
            has_yaw = (measured_yaw.get("status") == "observed" and not measured_yaw.get("control_rejected")
                       and task["object"]["shape"] == "cuboid")
            angle = measured_yaw["desk_yaw_rad"] if has_yaw else None
            xyz = label["raw_desk_xyz_m"]
            phase = ("initial" if index < task["events"]["pick_frame"] else
                     "final" if index > task["events"]["place_frame"] else "carry")
            samples.append({"sample_id": sample_id, "clip_index": source["index"], "frame_idx": index,
                            "split": clip["split"], "rgb_crop": relative, "rgb_crop_sha256": file_sha256(output),
                            "calibration_key": key, **observation})
            labels.append({"sample_id": sample_id, "target": teacher_target(xyz, angle, observation, calibration),
                           "yaw_observed": has_yaw, "desk_xyz_m": xyz, "desk_yaw_rad": angle,
                           "phase": phase, "confidence": label["confidence"]})
            count += 1
        print(f"{source['index']:02d} {clip['split']} {task['object_name']}: {count} observed samples", flush=True)
    inputs = directory / "inputs.json"
    write_json(inputs, {"schema_version": "hmnd_rgb_pose_inputs_v1", "clips": clips, "samples": samples,
                        "calibrations": calibrations, "geometry_feature_names": list(GEOMETRY_NAMES),
                        "crop_size": 224, "runtime_depth_required": False,
                        "notes": "Only RGB crops, RGB-derived SAM geometry, known objects and saved calibration. "
                                 "No pose labels, depth, capture time or clip identity are model features. "
                                 "SAM inference needs a manual RGB-only initial prompt; original prompt proposals could use depth."})
    teacher_path = directory / "teacher_labels.json"
    write_json(teacher_path, {"schema_version": "hmnd_depth_teacher_candidates_v1", "samples": labels,
                             "independently_validated_ground_truth": False,
                             "notes": "Quality-gated observed pseudo-labels, not predicted/interpolated guidance. "
                                      "Human video review does not prove absolute depth/calibration accuracy."})
    path = directory / "training_manifest.json"
    write_json(path, {"schema_version": "hmnd_rgb_student_training_v1", "inputs": "inputs.json",
                      "teacher_labels": "teacher_labels.json", "inputs_sha256": file_sha256(inputs),
                      "teacher_labels_sha256": file_sha256(teacher_path),
                      "source_review_manifest": str(manifest_path.resolve().relative_to(ROOT)),
                      "source_review_manifest_sha256": file_sha256(manifest_path),
                      "source_code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in
                                             (Path(__file__).resolve(), ROOT / "src/humanoid_retarget/rgb_data.py")},
                      "sample_counts": dict(Counter(r["split"] for r in samples)),
                      "clip_counts": dict(Counter(r["split"] for r in clips)), "stride": stride,
                      "test_status": "No held-out test yet; fresh recordings will be added after fitting/selection.",
                      "augmentation_policy": "Train-only photometric variants. No geometric flips/rotations. "
                                             "Reversing per-frame videos adds no new examples; derivatives cannot cross splits.",
                      "training_supervision": "approximate_depth_teacher_pseudo_labels", "training_ready": False})
    load_inputs(inputs)
    return path


def main():
    parser = argparse.ArgumentParser(description="Export RGB-only inputs and separate approximate depth teacher labels")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/datasets/pilot_030_yaw/manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-indices", type=int, nargs="+", default=list(range(23, 31)))
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    try:
        print(f"Training manifest: {prepare(args.manifest, args.out, args.validation_indices, args.stride)}")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Cannot prepare RGB supervision: {error}. Partial outputs preserved.\n")


if __name__ == "__main__":
    main()
