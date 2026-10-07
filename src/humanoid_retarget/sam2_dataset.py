from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import list_rgb_frames
from humanoid_retarget.mask_tracking import _inside_workspace
from humanoid_retarget.tracking import hsv_mask
from humanoid_retarget.types import ObjectSpec


def propose_prompt(image: np.ndarray, depth: np.ndarray, spec: ObjectSpec,
                   calibration: WorkspaceCalibration) -> list[int]:
    """Propose only an initial prompt; its identity still needs visual review."""
    if depth.shape != image.shape[:2] or image.shape[:2] != calibration.image_size_px[::-1]:
        raise ValueError("Prompt proposal requires calibrated, colour-aligned depth")
    mask = hsv_mask(image, spec)
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.05)
    py, px = np.nonzero(valid)
    if len(px) < 20:
        raise ValueError("No initial object prompt found; supply a checked box manually")
    points = calibration.pixels_depth_to_desk_xyz(np.column_stack((px, py)), depth[py, px])
    keep = (np.isfinite(points).all(axis=1) & _inside_workspace(points[:, :2], calibration)
            & (points[:, 2] > calibration.table_z_m + 0.008)
            & (points[:, 2] < calibration.table_z_m + spec.size_m[2] + 0.015))
    foreground = np.zeros(mask.shape, np.uint8)
    foreground[py[keep], px[keep]] = 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= spec.min_area_px]
    if not contours:
        raise ValueError("No initial object prompt found; supply a checked box manually")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return [max(0, x - 6), max(0, y - 6), min(image.shape[1], x + w + 6), min(image.shape[0], y + h + 6)]


def capture_issues(episode: Path, metadata: dict[str, Any]) -> list[str]:
    rgb = {p.stem for p in (episode / "rgb").glob("*.jpg")}
    depth = {p.stem for p in (episode / "depth").glob("*.png")}
    issues = []
    if len(rgb) < 3 or rgb != depth:
        issues.append("Missing or unmatched RGB/depth frames")
    if metadata.get("depth_aligned_to") != "color":
        issues.append("Depth is not explicitly colour-aligned")
    if metadata.get("interrupted") is not False or metadata.get("recording_guidance", {}).get("completed") is not True:
        issues.append("Capture did not complete normally")
    times = np.asarray(metadata.get("timestamps_relative_s", []), dtype=float)
    if times.shape != (len(rgb),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        issues.append("Missing or invalid capture timestamps")
    return issues


def image_boundary_frames(mask_dir: Path, count: int) -> list[int]:
    clipped = []
    for index in range(count):
        mask = cv2.imread(str(mask_dir / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Missing mask at frame {index}")
        if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            clipped.append(index)
    return clipped


def label_candidates(raw_payload: dict, filtered_payload: dict, quality: dict,
                     clipped_frames: tuple[int, ...] | list[int] = ()) -> tuple[list[dict], dict]:
    raw = {d["frame_idx"]: d for d in raw_payload["detections"]}
    filtered = {d["frame_idx"]: d for d in filtered_payload["detections"]}
    diagnostics = raw_payload["metadata"]["frame_diagnostics"]
    clipped = set(clipped_frames)
    step_flags, switch_flags = set(), set()
    for index in range(1, len(diagnostics)):
        previous, current = raw.get(index - 1), raw.get(index)
        if previous is None or current is None:
            continue
        delta = np.asarray(current["desk_xyz_m"]) - previous["desk_xyz_m"]
        if np.linalg.norm(delta) > 0.03:
            step_flags.update((index - 1, index))
        if previous["position_source"] != current["position_source"] and abs(delta[2]) > 0.006:
            switch_flags.update((index - 1, index))
    rows = []
    for diagnostic in diagnostics:
        index = diagnostic["frame_idx"]
        r, f = raw.get(index), filtered.get(index)
        reasons = []
        if r is None:
            reasons.append(diagnostic["status"])
        elif r["confidence"] < 0.6:
            reasons.append("weak_depth_area_support")
        if f is not None and not f["visible"]:
            reasons.append("kalman_prediction")
        if index in step_flags:
            reasons.append("large_adjacent_step_review")
        if index in switch_flags:
            reasons.append("height_estimator_switch_review")
        if index in clipped:
            reasons.append("mask_touches_image_boundary")
        rows.append({"frame_idx": index, "capture_t_s": diagnostic["t"],
                     "observed": r is not None, "position_label_candidate": r is not None and not reasons,
                     "raw_desk_xyz_m": r["desk_xyz_m"] if r else None,
                     "filtered_desk_xyz_m": f["desk_xyz_m"] if f and f["visible"] else None,
                     "position_source": r["position_source"] if r else None,
                     "confidence": r["confidence"] if r else 0.0, "exclusion_reasons": reasons})
    count = sum(r["position_label_candidate"] for r in rows)
    def hold_stats(indices):
        points = [filtered[i]["desk_xyz_m"] for i in indices
                  if i in filtered and filtered[i]["visible"] and filtered[i]["confidence"] >= 0.6]
        if len(points) < 3:
            return None
        points = np.asarray(points)
        return {"frames": len(points), "p95_deviation_mm": float(np.percentile(np.linalg.norm(
            points - np.median(points, axis=0), axis=1), 95) * 1000)}
    fps = float(raw_payload["metadata"]["fps"])
    summary = {**quality, "position_label_candidate_frames": count, "gated_frames": len(rows) - count,
               "large_step_frames": sorted(step_flags), "height_switch_frames": sorted(switch_flags),
               "image_boundary_frames": sorted(clipped),
               "continuous_supported_track": count == len(rows),
               "initial_hold": hold_stats(range(min(len(rows), max(3, int(0.4 * fps))))),
               "final_hold": hold_stats(range(max(0, len(rows) - max(3, int(fps))), len(rows))),
               "training_ready": False}
    return rows, summary


def review_sheet(episode: Path, masks: Path, raw_payload: dict, summary: dict, out: Path) -> None:
    frames = list_rgb_frames(episode)
    raw = {d["frame_idx"]: d for d in raw_payload["detections"]}
    count = len(frames)
    indices = set(np.linspace(0, count - 1, 8).astype(int).tolist())
    indices.update([min(count - 1, int(f * raw_payload["metadata"]["fps"])) for f in (8.0, 8.6, 9.0)])
    flagged = (summary["missing_frames"] + summary["weak_frames"] + summary["height_switch_frames"]
               + summary["image_boundary_frames"])
    if flagged:
        indices.add(flagged[len(flagged) // 2])
    indices = sorted(indices)[:12]
    tiles = []
    for index in indices:
        image = cv2.imread(str(frames[index]))
        mask = cv2.imread(str(masks / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Cannot decode review frame {index}")
        ys, xs = np.nonzero(mask)
        d = raw.get(index)
        centre = d["pixel_xy"] if d else ((np.median(xs), np.median(ys)) if len(xs) else (image.shape[1] / 2, image.shape[0] / 2))
        image[mask > 0] = (image[mask > 0] * 0.65 + np.array([70, 220, 80]) * 0.35).astype(np.uint8)
        x, y = (int(round(v)) for v in centre)
        cv2.drawMarker(image, (x, y), (0, 190, 255), cv2.MARKER_CROSS, 20, 2)
        left, top = max(0, min(x - 240, image.shape[1] - 480)), max(0, min(y - 145, image.shape[0] - 290))
        tile = cv2.resize(image[top:top + 290, left:left + 480], (320, 194))
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        label = f"{index} | {index / raw_payload['metadata']['fps']:.2f}s | " + (f"support {d['confidence']:.2f}" if d else "MISSING")
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    image = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    if not cv2.imwrite(str(out), image):
        raise OSError(f"Cannot write {out}")
