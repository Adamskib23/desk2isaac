#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.io import list_rgb_frames, read_episode_metadata, read_json, write_json
from humanoid_retarget.mask_tracking import file_sha256
from sam2_segment_frame import select_box, validate_box


def saved_prompt(path: Path, frames: list[Path]) -> tuple[int, np.ndarray]:
    prompt = read_json(path)
    index = prompt["frame_idx"]
    if not isinstance(index, int) or not 0 <= index < len(frames):
        raise ValueError("Saved prompt frame does not exist in this episode")
    source = frames[index]
    if (Path(prompt["source_image"]).resolve() != source.resolve()
            or prompt["source_sha256"] != file_sha256(source)):
        raise ValueError("Saved prompt belongs to another frame/episode or the source changed")
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not decode {source}")
    return index, validate_box(prompt["box_xyxy_px"], image.shape[1], image.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagate a SAM 2.1 Tiny prompt into auditable saved video masks.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--object-name", help="Object label; defaults to the episode's declared action_object")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt-json", type=Path, help="Reuse the saved-frame tool's fingerprinted prompt")
    prompts.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--frame", type=int, default=0, help="Prompt frame for --box or interactive selection")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/sam2/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="New directory for masks and manifest")
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.episode.resolve()):
        parser.error("Write masks outside the original episode")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; choose a fresh --out to preserve previous results")
    if not args.checkpoint.is_file():
        parser.error("Tiny checkpoint missing; follow docs/sam2_setup.md")
    frames = list_rgb_frames(args.episode)
    if any(path.suffix.lower() != ".jpg" for path in frames):
        parser.error("This video adapter requires recorded JPEG RGB frames")
    metadata = read_episode_metadata(args.episode)
    name = args.object_name or metadata.get("action_object")
    if not isinstance(name, str) or not name:
        parser.error("Declare --object-name or record an action_object label")
    if args.prompt_json is not None and args.frame != 0:
        parser.error("--prompt-json already identifies its frame; do not also set --frame")
    try:
        if args.prompt_json is not None:
            frame_index, box = saved_prompt(args.prompt_json, frames)
        else:
            frame_index = args.frame
            if not 0 <= frame_index < len(frames):
                raise ValueError("Prompt frame is outside the episode")
            image = cv2.imread(str(frames[frame_index]))
            if image is None:
                raise ValueError(f"Could not decode {frames[frame_index]}")
            box = (select_box(image) if args.box is None
                   else validate_box(args.box, image.shape[1], image.shape[0]))
    except (ValueError, cv2.error) as error:
        parser.error(str(error))
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; run in a GPU-enabled terminal or explicitly use --device cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "hmnd_sam2_masks_v1", "completed": False,
        "source_episode": str(args.episode.resolve()), "object_name": name,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml", "torch_version": torch.__version__,
        "source_metadata_sha256": file_sha256(args.episode / "metadata.json"),
        "prompt": {"frame_idx": frame_index, "box_xyxy_px": box.tolist(),
                   "rgb_sha256": file_sha256(frames[frame_index])},
        "prompt_json": str(args.prompt_json.resolve()) if args.prompt_json is not None else None,
        "device": args.device, "offload_video_to_cpu": True, "offload_state_to_cpu": True,
        "training_ready": False, "frames": [],
        "notes": "One prompted instance, not automatic semantic recognition. Masks may drift or be occluded. No workspace/marker image clipping. Original RGB/depth untouched.",
    }
    write_json(args.out / "manifest.json", manifest)
    predictor = build_sam2_video_predictor(manifest["model_config"], str(args.checkpoint),
                                           device=args.device, apply_postprocessing=False)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
    entries = {}
    started = time.perf_counter()
    # Numeric symlink names keep SAM's frame ordering identical to list_rgb_frames.
    with TemporaryDirectory(prefix="sam2_frames_", dir=args.out) as temp:
        for index, path in enumerate(frames):
            (Path(temp) / f"{index:06d}.jpg").symlink_to(path.resolve())
        with torch.inference_mode(), autocast:
            state = predictor.init_state(temp, offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_points_or_box(state, frame_idx=frame_index, obj_id=1, box=box)
            for reverse in ([False, True] if frame_index else [False]):
                for index, object_ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=frame_index, reverse=reverse):
                    if index in entries:
                        continue
                    if object_ids != [1]:
                        raise ValueError("Unexpected SAM object identity")
                    mask = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                    source = frames[index]
                    image = cv2.imread(str(source))
                    if image is None or mask.shape != image.shape[:2]:
                        raise ValueError(f"SAM mask/image dimensions differ at frame {index}")
                    output = args.out / f"{index:06d}.png"
                    if not cv2.imwrite(str(output), mask):
                        raise OSError(f"Could not save {output}")
                    entries[index] = {"frame_idx": index, "rgb_name": source.name,
                                      "rgb_sha256": file_sha256(source), "mask_sha256": file_sha256(output),
                                      "mask_pixels": int(np.count_nonzero(mask))}
    if sorted(entries) != list(range(len(frames))):
        raise ValueError("SAM did not return every episode frame; manifest remains incomplete")
    manifest.update(completed=True, frames=[entries[i] for i in range(len(frames))],
                    processing_s=time.perf_counter() - started)
    write_json(args.out / "manifest.json", manifest)
    print(f"Saved {len(entries)} masks for {name}: {args.out}")
    print("Next: scripts/build_trajectory.py --mask-dir with the matching episode and calibration.")


if __name__ == "__main__":
    main()
