#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from humanoid_retarget.recording import CaptureTimer


def _frame_to_bgr(color_frame) -> np.ndarray | None:
    from pyorbbecsdk import OBFormat

    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    frame_format = color_frame.get_format()
    if frame_format == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if frame_format == OBFormat.BGR:
        return data.reshape(height, width, 3).copy()
    if frame_format in (OBFormat.YUYV, OBFormat.UYVY):
        conversion = cv2.COLOR_YUV2BGR_YUYV if frame_format == OBFormat.YUYV else cv2.COLOR_YUV2BGR_UYVY
        return cv2.cvtColor(data.reshape(height, width, 2), conversion)
    if frame_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    raise RuntimeError(f"Unsupported camera colour format: {frame_format}")


def record_opencv(out: Path, seconds: float, fps: float, camera_index: int, *, timer: CaptureTimer | None = None) -> None:
    timer = timer or CaptureTimer(seconds)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open OpenCV camera index {camera_index}")
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps and actual_fps > 1:
        fps = float(actual_fps)

    rgb_dir = out / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frame_period = 1.0 / fps
    idx = 0
    timestamps: list[float] = []
    relative_times: list[float] = []
    width = height = 0
    startup_deadline = time.monotonic() + 20 + timer.countdown_seconds
    interrupted = False
    print("Opening RGB camera. Timer starts when frames are ready. Press Ctrl+C to stop early.", flush=True)
    try:
        while not timer.finished:
            start = time.monotonic()
            timer.update(start)
            if timer.finished:
                break
            ok, frame = cap.read()
            if not ok:
                if timer.started_at is None and time.monotonic() >= startup_deadline:
                    raise RuntimeError("Camera did not return valid RGB frames within the startup timeout")
                time.sleep(min(frame_period, 0.1))
                continue
            now = time.monotonic()
            if timer.update(now, frame_ready=True):
                if not cv2.imwrite(str(rgb_dir / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError(f"Could not save RGB frame {idx} to {out}")
                height, width = frame.shape[:2]
                timestamps.append(time.time())
                relative_times.append(now - timer.started_at)
                idx += 1
            sleep_s = frame_period - (time.monotonic() - start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        cap.release()
    if not timestamps and not interrupted:
        raise RuntimeError("Camera returned no RGB frames during the recording window")
    recorded_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else fps
    metadata = {
        "schema_version": "hmnd_rgbd_episode_v1",
        "fps": recorded_fps,
        "requested_fps": fps,
        "width": width,
        "height": height,
        "depth_unit_m": 1.0,
        "camera": f"opencv_index_{camera_index}",
        "timestamps_unix_s": timestamps,
        "timestamps_relative_s": relative_times,
        "recording_guidance": timer.metadata(),
        "interrupted": interrupted,
        "notes": "RGB-only capture; z estimates fall back to configured object dimensions.",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Wrote {idx} frames to {out}", flush=True)
    if interrupted:
        print("Stopped early; saved partial episode metadata (completed: false).", flush=True)
        raise KeyboardInterrupt


def record_orbbec(out: Path, seconds: float, fps: float, *, timer: CaptureTimer | None = None) -> None:
    timer = timer or CaptureTimer(seconds)
    try:
        from pyorbbecsdk import (
            AlignFilter, Config, OBFormat, OBFrameAggregateOutputMode,
            OBSensorType, OBStreamType, Pipeline,
        )
    except Exception as exc:
        raise RuntimeError(
            "pyorbbecsdk is not installed. Install pyorbbecsdk2, or use --backend opencv "
            "for a dry run with a standard webcam."
        ) from exc

    pipeline = Pipeline()
    config = Config()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = color_profiles.get_video_stream_profile(0, 0, OBFormat.RGB, int(fps))
    except Exception:
        color_profile = color_profiles.get_default_video_stream_profile()
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    pipeline.enable_frame_sync()
    pipeline.start(config)
    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    frame_period = 1.0 / fps
    idx = 0
    timestamps: list[float] = []
    relative_times: list[float] = []
    width = height = 0
    camera_intrinsics = None
    camera_distortion = None
    warmup_remaining = 15
    startup_deadline = time.monotonic() + 20 + timer.countdown_seconds
    interrupted = False
    print("Warming up Orbbec RGB-D. Timer starts after warm-up and valid aligned frames. Press Ctrl+C to stop early.", flush=True)
    try:
        while not timer.finished:
            start = time.monotonic()
            timer.update(start)
            if timer.finished:
                break
            if timer.started_at is None and start >= startup_deadline:
                raise RuntimeError("Camera did not return valid aligned RGB-D frames within the startup timeout")
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue
            if warmup_remaining:
                warmup_remaining -= 1
                continue
            frames = align_filter.process(frames)
            if frames is None:
                continue
            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            bgr = _frame_to_bgr(color_frame)
            if bgr is None:
                continue
            width, height = int(bgr.shape[1]), int(bgr.shape[0])
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_raw = depth_raw.reshape((depth_frame.get_height(), depth_frame.get_width()))
            if depth_raw.shape != bgr.shape[:2]:
                raise RuntimeError("Aligned depth dimensions do not match colour; cannot use these frames for calibration")
            if camera_intrinsics is None:
                camera_param = pipeline.get_camera_param()
                intrinsic = camera_param.rgb_intrinsic
                camera_intrinsics = {
                    key: float(getattr(intrinsic, key)) for key in ("fx", "fy", "cx", "cy")
                }
                camera_intrinsics.update(width=int(intrinsic.width), height=int(intrinsic.height))
                distortion = camera_param.rgb_distortion
                camera_distortion = [float(getattr(distortion, key))
                                     for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]
            scale_to_mm = float(depth_frame.get_depth_scale())
            depth_mm = np.clip(
                np.round(depth_raw.astype(np.float32) * scale_to_mm),
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            now = time.monotonic()
            if timer.update(now, frame_ready=True):
                rgb_saved = cv2.imwrite(str(rgb_dir / f"{idx:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                depth_saved = cv2.imwrite(str(depth_dir / f"{idx:06d}.png"), depth_mm)
                if not rgb_saved or not depth_saved:
                    raise OSError(f"Could not save RGB-D frame {idx} to {out}")
                timestamps.append(time.time())
                relative_times.append(now - timer.started_at)
                idx += 1
            sleep_s = frame_period - (time.monotonic() - start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        pipeline.stop()

    if not timestamps and not interrupted:
        raise RuntimeError("Camera opened but returned no aligned RGB-D pairs; retry with --seconds 10")
    recorded_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else fps
    metadata = {
        "schema_version": "hmnd_rgbd_episode_v1",
        "fps": recorded_fps,
        "requested_fps": fps,
        "width": width,
        "height": height,
        "depth_unit_m": 0.001,
        "depth_aligned_to": "color",
        "color_intrinsics": camera_intrinsics,
        "color_distortion": camera_distortion,
        "camera": "orbbec_gemini_2",
        "timestamps_unix_s": timestamps,
        "timestamps_relative_s": relative_times,
        "recording_guidance": timer.metadata(),
        "interrupted": interrupted,
        "notes": "Depth is aligned to colour and stored as 16-bit millimetres in depth/*.png.",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Wrote {idx} RGB-D frames to {out}", flush=True)
    if interrupted:
        print("Stopped early; saved partial episode metadata (completed: false).", flush=True)
        raise KeyboardInterrupt


def preview_timer(timer: CaptureTimer) -> None:
    print("Timer preview only; no camera is opened and no episode is written.", flush=True)
    while not timer.finished:
        timer.update(time.monotonic(), frame_ready=True)
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an RGB-D manipulation episode.")
    parser.add_argument("--out", help="Output episode directory (required unless previewing the timer).")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--backend", choices=["orbbec", "opencv"], default="orbbec")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--guided", action="store_true", help="Print countdown and hold/move/hold cues during capture.")
    parser.add_argument("--countdown", type=int, default=3, help="Preparation countdown in guided mode (seconds).")
    parser.add_argument("--initial-hold-seconds", type=float, default=2.0)
    parser.add_argument("--final-hold-seconds", type=float, default=2.0)
    parser.add_argument("--beep", action="store_true", help="Ring the terminal bell at recording phase changes.")
    parser.add_argument("--preview-timer", action="store_true", help="Preview the timer without using a camera or saving frames.")
    args = parser.parse_args()

    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    try:
        timer = CaptureTimer(args.seconds, guided=args.guided, countdown_seconds=args.countdown,
                             initial_hold_seconds=args.initial_hold_seconds, final_hold_seconds=args.final_hold_seconds,
                             beep=args.beep)
    except ValueError as exc:
        parser.error(str(exc))
    if args.preview_timer:
        preview_timer(timer)
        return
    if not args.out:
        parser.error("--out is required when recording")
    out = Path(args.out)
    if any((out / folder).exists() and any((out / folder).iterdir()) for folder in ("rgb", "depth")):
        parser.error(f"Capture folder {out} already contains frames; choose a new --out directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.backend == "orbbec":
        record_orbbec(out, args.seconds, args.fps, timer=timer)
    else:
        record_opencv(out, args.seconds, args.fps, args.camera_index, timer=timer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
