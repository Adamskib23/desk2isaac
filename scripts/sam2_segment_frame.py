#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.io import list_rgb_frames, write_json


def validate_box(box, width: int, height: int) -> np.ndarray:
    values = np.asarray(box, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Box must contain four finite coordinates: x0 y0 x1 y1")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Box must have positive area and lie inside the source image")
    return values


def select_box(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, 1280 / width, 800 / height)
    preview = cv2.resize(image, (round(width * scale), round(height * scale)))
    title = "SAM 2 - select object, Enter confirms, Esc cancels"
    print("Drag a tight rectangle around the object; press Enter to confirm.", flush=True)
    try:
        x, y, w, h = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyAllWindows()
    # Map the prompt back to the original image, not the resized display.
    sx, sy = width / preview.shape[1], height / preview.shape[0]
    return validate_box([x * sx, y * sy, (x + w) * sx, (y + h) * sy], width, height)


def mask_preview(image: np.ndarray, mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask must match the original RGB frame dimensions")
    result = image.copy()
    result[mask] = np.round(0.55 * image[mask] + 0.45 * np.asarray([70, 220, 80])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (70, 220, 80), 2)
    x0, y0, x1, y1 = np.round(box).astype(int)
    cv2.rectangle(result, (x0, y0), (x1, y1), (0, 190, 255), 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SAM 2.1 Tiny on one saved RGB frame.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Original-image pixel coordinates; otherwise select a box in a window")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/sam2/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", type=Path, default=Path("outputs/sam2/frame_check"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Tiny checkpoint missing; follow docs/sam2_setup.md")
    if args.out.resolve().is_relative_to(args.episode.resolve()):
        parser.error("Write previews outside the original episode directory")
    frames = list_rgb_frames(args.episode)
    if not 0 <= args.frame < len(frames):
        parser.error(f"--frame must be between 0 and {len(frames) - 1}")
    source = frames[args.frame]
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Could not decode {source}")
    try:
        box = (select_box(image) if args.box is None
               else validate_box(args.box, image.shape[1], image.shape[0]))
    except (ValueError, cv2.error) as error:
        parser.error(f"{error}. For headless use, supply --box X0 Y0 X1 Y1")

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        parser.error(f"Run with .venv-sam2/bin/python; SAM dependencies unavailable: {error}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this session. Check GPU access or explicitly use --device cpu")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    predictor = SAM2ImagePredictor(build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(args.checkpoint), device=args.device,
    ))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
    if args.device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), autocast:
        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    if args.device == "cuda":
        torch.cuda.synchronize()
    inference_s = time.perf_counter() - started
    mask = masks[0].astype(bool)
    if not mask.any():
        parser.error("SAM produced an empty mask; try a clearer frame or a tighter object prompt")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, output in (("mask.png", mask.astype(np.uint8) * 255),
                         ("preview.jpg", mask_preview(image, mask, box))):
        if not cv2.imwrite(str(args.out / name), output):
            raise OSError(f"Could not write {args.out / name}")
    write_json(args.out / "prompt.json", {
        "source_image": str(source.resolve()), "frame_idx": args.frame,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "box_xyxy_px": box.tolist(), "checkpoint": str(args.checkpoint.resolve()),
        "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "device": args.device, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "mask_pixels": int(mask.sum()), "model_mask_score": float(scores[0]),
        "inference_s": inference_s,
        "peak_allocated_vram_mb": (torch.cuda.max_memory_allocated() / 1024 ** 2
                                   if args.device == "cuda" else None),
        "scope": "Single RGB frame segmentation test only; no depth, pose, video tracking or robot execution.",
        "notes": "Score is not calibrated tracking confidence. Single-frame latency is not a video FPS benchmark. No workspace/marker clipping is applied.",
    })
    print(f"SAM 2.1 Tiny: {int(mask.sum())} mask pixels, score {float(scores[0]):.3f}")
    print(f"Single-frame inference: {inference_s:.3f} s (model loading excluded)")
    print(f"Saved {args.out / 'preview.jpg'} and {args.out / 'mask.png'}")


if __name__ == "__main__":
    main()
