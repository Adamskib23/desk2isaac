from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from humanoid_retarget.io import read_json
from humanoid_retarget.mask_tracking import file_sha256


def checked_replay_task(task: dict, labels: dict) -> dict:
    """Gate measured guidance without treating robot interpolation as ground truth."""
    if task.get("schema_version") != "hmnd_rgbd_retarget_v1":
        raise ValueError("Unsupported trajectory schema")
    rows = labels["frames"]
    indices = [row["frame_idx"] for row in rows]
    if indices != list(range(len(rows))) or len(rows) < 3:
        raise ValueError("Candidate frames must cover the entire ordered capture")
    accepted = {row["frame_idx"] for row in rows if row["position_label_candidate"]}
    for row in rows:
        if row["position_label_candidate"] and (not row["observed"] or row["exclusion_reasons"]):
            raise ValueError("Excluded/unobserved frame cannot be accepted")
    width = max(3, int(round(float(task["fps"]) * 0.4)))
    for boundary in (range(min(width, len(rows))), range(max(0, len(rows) - width), len(rows))):
        if sum(i in accepted for i in boundary) < np.ceil(len(boundary) * 0.75):
            raise ValueError("Insufficient supported initial/final endpoint observations")
    result = deepcopy(task)
    result["object_trace"] = [row for row in task["object_trace"]
                              if row["frame_idx"] in accepted and row["visible"]]
    carry = [row for row in result["object_trace"]
             if task["events"]["pick_frame"] <= row["frame_idx"] <= task["events"]["place_frame"]]
    if len(carry) < 2:
        raise ValueError("Insufficient supported carry guidance")
    result["events"].update(training_ready=False, guidance_quality_gated=True,
                            guidance_excluded_frames=sorted(set(indices) - accepted),
                            interpolation_policy="Robot control only; not measured perception labels")
    return result


def load_review_manifest(path: Path, root: Path, *, verify: bool = True) -> dict:
    manifest = read_json(path)
    if manifest.get("schema_version") != "hmnd_prepared_review_dataset_v1":
        raise ValueError("Unsupported review dataset schema")
    if manifest.get("ready_for_isaac_review") is not True:
        raise ValueError("Dataset has not passed review preparation")
    rows = manifest["clips"]
    if len(rows) != manifest["clip_count"] or len({r["episode"] for r in rows}) != len(rows):
        raise ValueError("Duplicate clips or inconsistent manifest count")
    for index, row in enumerate(rows, start=1):
        if row["index"] != index:
            raise ValueError("Review indices must be consecutive and one-based")
        required = [row["trajectory"], row["candidate_labels"], row["annotated_video"],
                    str(Path(row["episode"]) / "metadata.json")]
        if any(relative not in row["file_sha256"] for relative in required):
            raise ValueError("Replay/reference paths must be fingerprint-bound")
        for relative, expected in row["file_sha256"].items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                raise ValueError(f"Missing or escaped dataset path: {relative}")
            if verify and file_sha256(target) != expected:
                raise ValueError(f"Dataset fingerprint changed: {relative}")
    return manifest


def reference_knots(task: dict, capture_times: list[float]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(capture_times, dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Reference timestamps must increase and be finite")
    waypoints = task["waypoints"]
    guidance = task.get("task_input", {}).get("guidance_trajectory", [])
    carry = [wp for wp in waypoints if wp["phase"] == "carry_guidance"]
    if not carry:
        return np.asarray([0.0, waypoints[-1]["t"]]), np.asarray([times[0], times[-1]])
    if len(carry) != len(guidance) or any(row.get("source_frame_idx") is None for row in guidance):
        raise ValueError("Reference playback needs frame-indexed physical guidance")
    knots = [(0.0, times[0])]
    by_time = {wp["t"]: row["source_frame_idx"] for wp, row in zip(carry, guidance)}
    last_source = times[0]
    for wp in waypoints:
        if wp["t"] in by_time:
            frame = by_time[wp["t"]]
            if not 0 <= frame < len(times):
                raise ValueError("Guidance source frame is outside the reference capture")
            last_source = times[frame]
            knots.append((wp["t"], last_source))
        elif wp["phase"] == "airborne_hold":
            knots.append((wp["t"], last_source))
    knots.append((waypoints[-1]["t"], times[-1]))
    knots = np.asarray(knots, dtype=float)
    if np.any(np.diff(knots[:, 0]) <= 0) or np.any(np.diff(knots[:, 1]) < 0):
        raise ValueError("Reference alignment must be monotonic")
    return knots[:, 0], knots[:, 1]


def reference_frame(command_time: float, knots: tuple[np.ndarray, np.ndarray], capture_times: list[float]) -> int:
    source_time = np.interp(command_time, *knots)
    return int(np.clip(np.searchsorted(capture_times, source_time, side="right") - 1, 0, len(capture_times) - 1))


def comparison_frame(rgb_bgr: np.ndarray, sim_bgr: np.ndarray, *, title: str,
                     capture_t: float, sim_t: float) -> np.ndarray:
    height, width = sim_bgr.shape[:2]
    reference = np.zeros_like(sim_bgr)
    scale = min(width / rgb_bgr.shape[1], (height - 44) / rgb_bgr.shape[0])
    resized = cv2.resize(rgb_bgr, (int(round(rgb_bgr.shape[1] * scale)), int(round(rgb_bgr.shape[0] * scale))))
    y = 44 + (height - 44 - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    reference[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(reference, f"RGB | capture {capture_t:.2f}s | {title}", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    result = np.hstack((reference, sim_bgr))
    cv2.putText(result, f"sim {sim_t:.2f}s", (width + 12, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return result
