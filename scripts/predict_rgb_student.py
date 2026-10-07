#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.io import write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.rgb_data import load_inputs
from humanoid_retarget.rgb_student import load_student, predict_inputs


def main():
    parser = argparse.ArgumentParser(description="Predict poses using RGB inputs only; no teacher/depth files opened")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-cache", type=Path, help="Torch Hub cache for the compact checkpoint encoder")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh prediction output")
    torch.set_num_threads(4)
    inputs = load_inputs(args.inputs)
    model, checkpoint = load_student(args.checkpoint, args.device, encoder_cache=args.encoder_cache)
    started = time.perf_counter()
    predictions = predict_inputs(model, inputs, args.inputs.parent, split=args.split, device=args.device)
    elapsed = time.perf_counter() - started
    write_json(args.out, {"schema_version": "hmnd_rgb_pose_predictions_v1", "samples": predictions,
                          "runtime_depth_required": False, "teacher_files_loaded": False,
                          "inputs_sha256": file_sha256(args.inputs), "checkpoint_sha256": file_sha256(args.checkpoint),
                          "training_inputs_sha256": checkpoint["training_inputs_sha256"],
                          "split": args.split, "elapsed_s": elapsed,
                          "crops_per_second_including_io": len(predictions) / elapsed,
                          "approved_for_robot_execution": False,
                          "notes": "Approximate learned pose candidates; not independently checked physical measurements. "
                                   "Timing excludes SAM masking and model loading."})
    print(f"Predicted {len(predictions)} RGB poses without teacher/depth files: {args.out}")


if __name__ == "__main__":
    main()
