#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.rgb_data import GEOMETRY_NAMES, decode_pose, load_training, pose_metrics
from humanoid_retarget.rgb_student import RGBPoseStudent, image_tensor, pose_loss
from humanoid_retarget.training_control import TrainingBudget


def encode_features(model, indices, samples, root, batch_size, device, rng=None, budget=None):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            if budget is not None and budget.time_expired:
                return None
            rows = [samples[i] for i in indices[start:start + batch_size]]
            images = torch.stack([image_tensor(root / r["rgb_crop"], rng) for r in rows])
            chunks.append(model.encoder(images.to(device)).cpu())
    return torch.cat(chunks)


def save_progress(directory, checkpoint, history, progress):
    # Replace complete files atomically so interruption cannot corrupt the last save.
    temporary = directory / ".best.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(directory / "best.pt")
    for name, payload in (("history.json", {"epochs": history}), ("progress.json", progress)):
        temporary = directory / ("." + name + ".tmp")
        write_json(temporary, payload)
        temporary.replace(directory / name)


def train(args):
    if not args.allow_pseudo_labels:
        raise ValueError("Acknowledge approximate teacher supervision with --allow-pseudo-labels")
    limits = TrainingBudget(args.epochs, args.time_budget_minutes)
    if args.batch_size < 1 or args.augment_views < 0 or args.refresh_augment_every < 0:
        raise ValueError("Batch size must be positive and augmentation counts nonnegative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if args.refresh_augment_every and not args.augment_views:
        raise ValueError("Refreshing augmentation requires at least one augmented view")
    directory = args.out.resolve()
    if not directory.is_relative_to(ROOT / "outputs") or directory.exists():
        raise ValueError("Choose a fresh directory under outputs")
    manifest, inputs, labels = load_training(args.manifest)
    directory.mkdir(parents=True)
    source_fingerprints = {str(p.relative_to(ROOT)): file_sha256(p) for p in
                           (Path(__file__).resolve(), ROOT / "src/humanoid_retarget/rgb_student.py",
                            ROOT / "src/humanoid_retarget/rgb_data.py",
                            ROOT / "src/humanoid_retarget/training_control.py")}
    manifest_fingerprint = file_sha256(args.manifest)
    write_json(directory / "run_config.json", {"training_manifest_sha256": manifest_fingerprint,
                                               "source_code_sha256": source_fingerprints,
                                               "seed": args.seed, "epoch_limit": limits.max_epochs,
                                               "time_budget_minutes": args.time_budget_minutes,
                                               "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                                               "augment_views": args.augment_views,
                                               "refresh_augment_every_epochs": args.refresh_augment_every,
                                               "encoder_frozen": True, "device": args.device,
                                               "supervision": "approximate_depth_teacher_pseudo_labels"})
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(ROOT / "models/rgb_student/cache"))
    model = RGBPoseStudent(pretrained=True).to(args.device).eval()
    rng = np.random.default_rng(args.seed)
    samples = inputs["samples"]
    train_indices = [i for i, row in enumerate(samples) if row["split"] == "train"]
    val_indices = [i for i, row in enumerate(samples) if row["split"] == "val"]
    started = time.perf_counter()
    features = []
    for view in range(args.augment_views + 1):
        indices = list(range(len(samples))) if view == 0 else train_indices
        features.append(encode_features(model, indices, samples, args.manifest.parent,
                                        args.batch_size, args.device, rng if view else None))
        print(f"Encoded view {view}: {len(indices)} RGB crops", flush=True)
    base_features = features[0].to(args.device)
    geometry = torch.tensor([r["geometry"] for r in samples], dtype=torch.float32, device=args.device)
    targets = torch.tensor([labels[r["sample_id"]]["target"] for r in samples], dtype=torch.float32, device=args.device)
    yaw_mask = torch.tensor([labels[r["sample_id"]]["yaw_observed"] for r in samples], dtype=torch.float32, device=args.device)
    with torch.no_grad():
        model.feature_mean.copy_(base_features[train_indices].mean(0))
        model.feature_std.copy_(base_features[train_indices].std(0).clamp_min(0.1))
        model.geometry_mean.copy_(geometry[train_indices].mean(0))
        model.geometry_std.copy_(geometry[train_indices].std(0).clamp_min(0.01))
    train_features = torch.cat([base_features[train_indices], *[f.to(args.device) for f in features[1:]]])
    train_geometry = geometry[train_indices].repeat(args.augment_views + 1, 1)
    train_targets = targets[train_indices].repeat(args.augment_views + 1, 1)
    train_yaw = yaw_mask[train_indices].repeat(args.augment_views + 1)
    strata = [(samples[i]["object_name"], labels[samples[i]["sample_id"]]["phase"]) for i in train_indices]
    counts = Counter(strata)
    weights = torch.tensor([1 / counts[s] for s in strata], dtype=torch.float32, device=args.device).repeat(args.augment_views + 1)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=0.001)
    history, best_state, best_loss, best_epoch = [], None, float("inf"), None
    budget = TrainingBudget(args.epochs, args.time_budget_minutes)
    epoch, refreshes, next_save, next_log = 0, 0, 0.0, 0.0

    def checkpoint_payload():
        return {"schema_version": "hmnd_rgb_pose_student_v1", "state_dict": best_state,
                "geometry_feature_names": list(GEOMETRY_NAMES), "seed": args.seed,
                "training_inputs_sha256": manifest["inputs_sha256"], "best_epoch": best_epoch,
                "training_manifest_sha256": manifest_fingerprint, "source_code_sha256": source_fingerprints,
                "supervision": "approximate_depth_teacher_pseudo_labels",
                "encoder": "torchvision_resnet18_IMAGENET1K_V1_frozen"}

    def persist(reason=None):
        if best_state is not None:
            save_progress(directory, checkpoint_payload(), history,
                          {"completed_epochs": epoch, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                           "fitting_elapsed_s": budget.elapsed_s, "fitting_time_budget_minutes": args.time_budget_minutes,
                           "augmentation_refreshes": refreshes, "running": reason is None, "stop_reason": reason})

    print(f"Fitting budget: {args.time_budget_minutes} minutes; epoch limit: {budget.max_epochs}. "
          "Best checkpoint/progress saved about every 60 seconds and on exit.", flush=True)
    stop_reason = None
    try:
        while (stop_reason := budget.stop_reason(epoch)) is None:
            if epoch and args.refresh_augment_every and epoch % args.refresh_augment_every == 0:
                model.eval()
                variants = []
                for _ in range(args.augment_views):
                    features = encode_features(model, train_indices, samples, args.manifest.parent,
                                               args.batch_size, args.device, rng, budget)
                    if features is None:
                        break
                    variants.append(features.to(args.device))
                if len(variants) != args.augment_views or budget.time_expired:
                    stop_reason = "time_budget"
                    break
                train_features = torch.cat([base_features[train_indices], *variants])
                refreshes += 1
            model.train()
            selected = torch.multinomial(weights, len(train_features), replacement=True)
            losses = []
            for batch in selected.split(args.batch_size):
                if budget.time_expired:
                    break
                prediction = model.from_features(train_features[batch], train_geometry[batch])
                loss = pose_loss(prediction, train_targets[batch], train_yaw[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            if not losses:
                stop_reason = "time_budget"
                break
            model.eval()
            with torch.no_grad():
                prediction = model.from_features(base_features[val_indices], geometry[val_indices])
                validation = float(pose_loss(prediction, targets[val_indices], yaw_mask[val_indices]))
            if not np.isfinite(validation):
                raise ValueError("Nonfinite validation loss; last periodic checkpoint preserved")
            epoch += 1
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": validation,
                            "fitting_elapsed_s": budget.elapsed_s, "optimizer_batches": len(losses),
                            "partial_epoch": len(losses) < len(selected.split(args.batch_size))})
            if validation < best_loss:
                best_loss, best_epoch = validation, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if budget.elapsed_s >= next_save:
                persist()
                next_save = budget.elapsed_s + 60
            if epoch == 1 or (args.time_budget_minutes is None and epoch % 10 == 0) or budget.elapsed_s >= next_log:
                print(f"Epoch {epoch:03d} | fitting {budget.elapsed_s / 60:.1f} min | train {np.mean(losses):.5f}, "
                      f"val {validation:.5f}; best {best_epoch}", flush=True)
                next_log = budget.elapsed_s + 30
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Interrupted; saving the best validated checkpoint and final report.", flush=True)
    if best_state is None:
        raise ValueError("Stopped before any validated epoch; no trained checkpoint exists")
    fitting_elapsed = budget.elapsed_s
    persist(stop_reason)
    model.load_state_dict(best_state)
    model.eval()
    calibrations = {k: WorkspaceCalibration(**v) for k, v in inputs["calibrations"].items()}
    metrics, predictions = {}, {}
    with torch.no_grad():
        outputs = model.from_features(base_features, geometry).cpu().numpy()
    for split in ("train", "val"):
        student, baseline = [], []
        for row, output in zip(samples, outputs):
            if row["split"] != split:
                continue
            calibration = calibrations[row["calibration_key"]]
            student.append({"sample_id": row["sample_id"], **decode_pose(output, row, calibration)})
            baseline_output = [row["size_m"][2] / 0.2, 0, 0, 1, 0]
            pose = decode_pose(baseline_output, row, calibration)
            pose["desk_yaw_rad"] = None
            baseline.append({"sample_id": row["sample_id"], **pose})
        predictions[split] = student
        metrics[split] = {"student": pose_metrics(student, inputs, labels),
                          "rgb_centroid_rest_height_baseline": pose_metrics(baseline, inputs, labels)}
    checkpoint = directory / "best.pt"
    write_json(directory / "fit_predictions.json", predictions)
    report = {"schema_version": "hmnd_rgb_student_report_v1", "metrics": metrics,
              "training_manifest_sha256": manifest_fingerprint, "checkpoint_sha256": file_sha256(checkpoint),
              "source_code_sha256": source_fingerprints,
              "seed": args.seed, "epochs": epoch, "requested_epoch_limit": budget.max_epochs,
              "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "stop_reason": stop_reason, "fitting_time_budget_minutes": args.time_budget_minutes,
              "fitting_elapsed_s": fitting_elapsed, "learning_rate": args.learning_rate,
              "refresh_augment_every_epochs": args.refresh_augment_every, "augmentation_refreshes": refreshes,
              "clip_counts": manifest["clip_counts"], "sample_counts": manifest["sample_counts"],
              "photometric_train_views": args.augment_views, "elapsed_s": time.perf_counter() - started,
              "checkpoint_selection": "Minimum validation Huber(z/0.1m, pixel offsets/bbox) + 0.05 masked doubled-yaw MSE",
              "sampling": "Balanced object/initial-carry-final strata, replacement sampling; labels not model inputs",
              "normalisation": "Training base views only; validation images never augmented",
              "teacher_agreement_not_ground_truth_accuracy": True, "held_out_test_completed": False,
              "notes": "Fixed camera, saved calibration, known objects and RGB-derived SAM masks. "
                       "Model selection uses batch 003, not an independent test. No robot evaluation performed.",
              "versions": {"torch": str(torch.__version__), "python": sys.version.split()[0]}}
    write_json(directory / "report.json", report)
    metric = metrics["val"]["student"]["overall"]
    print(f"Best epoch {best_epoch}; validation teacher agreement: median XYZ {metric['xyz_mm']['median']:.2f} mm, "
          f"p95 {metric['xyz_mm']['p95']:.2f} mm. Report: {directory / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fit an RGB-only student to approximate depth teacher candidates")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-pseudo-labels", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch cap; default 80 unless a time budget is supplied")
    parser.add_argument("--time-budget-minutes", type=float, help="Fitting wall-clock budget; setup and final reporting are additional")
    parser.add_argument("--refresh-augment-every", type=int, default=0, help="Refresh train-only photometric views every N epochs (0 disables)")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augment-views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"RGB training failed: {error}. Partial outputs preserved.\n")


if __name__ == "__main__":
    main()
