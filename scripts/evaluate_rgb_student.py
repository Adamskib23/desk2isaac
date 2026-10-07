#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.io import read_json, write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.rgb_data import load_training, pose_metrics


def plot_clips(inputs, labels, predictions, root, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = {r["sample_id"]: r for r in predictions}
    for clip in inputs["clips"]:
        rows = sorted([r for r in inputs["samples"] if r["clip_index"] == clip["index"]
                       and r["sample_id"] in predicted], key=lambda r: r["frame_idx"])
        if not rows:
            continue
        target = np.asarray([labels[r["sample_id"]]["desk_xyz_m"] for r in rows]) * 1000
        poses = np.asarray([predicted[r["sample_id"]]["desk_xyz_m"] for r in rows]) * 1000
        frame = [r["frame_idx"] for r in rows]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        colours = ("#137c68", "#245ab1")
        for xyz, name, colour in zip((target, poses), ("Approx. depth teacher", "RGB student"), colours):
            axes[0, 0].plot(xyz[:, 0], xyz[:, 1], color=colour, label=name, linewidth=1.5)
            axes[0, 1].plot(frame, xyz[:, 2], color=colour, label=name, linewidth=1.5)
        axes[0, 0].set(xlabel="Desk X (mm)", ylabel="Desk Y (mm)", title="Object-centre path", aspect="equal")
        axes[0, 1].set(xlabel="Recorded frame", ylabel="Centre Z (mm)", title="Height")
        yaw_rows = [r for r in rows if labels[r["sample_id"]]["yaw_observed"]
                    and predicted[r["sample_id"]]["desk_yaw_rad"] is not None]
        if yaw_rows:
            delta = np.asarray([predicted[r["sample_id"]]["desk_yaw_rad"] - labels[r["sample_id"]]["desk_yaw_rad"]
                                for r in yaw_rows])
            error = np.rad2deg(np.abs(np.arctan2(np.sin(2 * delta), np.cos(2 * delta))) / 2)
            axes[0, 2].plot([r["frame_idx"] for r in yaw_rows], error, color="#b63744")
            axes[0, 2].set(xlabel="Recorded frame", ylabel="Yaw error (degrees)", title="Yaw teacher disagreement", ylim=(0, 90))
        else:
            axes[0, 2].text(0.5, 0.5, "Sphere: yaw not defined", ha="center", va="center")
            axes[0, 2].axis("off")
        axes[0, 0].legend(fontsize=8)
        for axis in axes[0]:
            axis.grid(alpha=0.2)
        for axis, phase in zip(axes[1], ("initial", "carry", "final")):
            matching = [r for r in rows if labels[r["sample_id"]]["phase"] == phase]
            row = matching[len(matching) // 2] if matching else rows[len(rows) // 2]
            image = cv2.imread(str(root / row["rgb_crop"]))
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            error = np.linalg.norm(np.asarray(predicted[row["sample_id"]]["desk_xyz_m"]) - labels[row["sample_id"]]["desk_xyz_m"]) * 1000
            axis.set_title(f"{phase.title()}, frame {row['frame_idx']} | XYZ error {error:.1f} mm", fontsize=10)
            axis.axis("off")
        fig.suptitle(f"Validation clip {clip['index']:02d}: {clip['object_name']} | Teacher agreement, not ground-truth accuracy", fontsize=12)
        fig.savefig(directory / f"{clip['index']:03d}_{clip['object_name']}.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare RGB predictions to separate approximate teacher labels")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh evaluation directory")
    _, inputs, labels = load_training(args.manifest)
    payload = read_json(args.predictions)
    if payload.get("schema_version") != "hmnd_rgb_pose_predictions_v1":
        parser.error("Unsupported prediction file")
    if payload["inputs_sha256"] != file_sha256(args.manifest.parent / "inputs.json"):
        parser.error("Predictions do not match the evaluation RGB inputs")
    expected = {r["sample_id"] for r in inputs["samples"] if payload["split"] == "all" or r["split"] == payload["split"]}
    if {r["sample_id"] for r in payload["samples"]} != expected:
        parser.error("Predictions must cover every sample in the selected split")
    args.out.mkdir(parents=True)
    metrics = pose_metrics(payload["samples"], inputs, labels)
    write_json(args.out / "metrics.json", {"split": payload["split"], "metrics": metrics,
                                          "teacher_agreement_not_ground_truth_accuracy": True,
                                          "predictions_sha256": file_sha256(args.predictions),
                                          "training_manifest_sha256": file_sha256(args.manifest)})
    if args.plots:
        plot_clips(inputs, labels, payload["samples"], args.manifest.parent, args.out)
    print(f"Teacher agreement metrics saved: {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
