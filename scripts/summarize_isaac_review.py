#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.io import read_json, write_json
from humanoid_retarget.mask_tracking import file_sha256


def summarize(run_path: Path, manifest_path: Path) -> dict:
    state, manifest = read_json(run_path), read_json(manifest_path)
    if state["mode"] != "dynamic" or state["identity"]["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Review is not a dynamic run of this frozen manifest")
    sources = {row["index"]: row for row in manifest["clips"]}
    if len({r["index"] for r in state["clips"]}) != len(state["clips"]):
        raise ValueError("Duplicate review results")
    rows = []
    for result in sorted(state["clips"], key=lambda r: r["index"]):
        if not result["completed"]:
            continue
        source = sources[result["index"]]
        directory = Path(result["directory"])
        metrics = read_json(directory / "metrics.json")
        plan = read_json(directory / "metrics_plan.json")
        if (metrics["mode"] != "dynamic_contact_grasp_with_guidance_ik" or not metrics["completed"]
                or metrics["object_pose_commands_during_run"] != 0
                or Path(metrics["source_trajectory"]).resolve() != (ROOT / source["trajectory"]).resolve()):
            raise ValueError("Metrics do not describe completed, uncommanded-object dynamics for this task")
        physical = metrics["physical_results"][0]
        carry = next(wp for wp in plan["waypoints"] if wp["phase"] == "carry_guidance")
        offset = np.asarray(carry["ee_xyz_m"]) - carry["object_xyz_m"]
        samples = [s for s in metrics["samples"] if s["phase"] in ("carry_guidance", "airborne_hold")]
        errors = [np.linalg.norm(np.asarray(s["environments"][0]["object_xyz_m"])
                                 - (np.asarray(s["target_ee_xyz_m"]) - offset)) for s in samples]
        if not errors or not np.isfinite(errors).all():
            raise ValueError("Missing/nonfinite carry telemetry")
        placed = [s for s in metrics["samples"] if s["phase"] in ("descend_to_place", "place_hold")]
        before_release, release_shift = None, None
        if placed:
            before = np.asarray(placed[-1]["environments"][0]["object_xyz_m"][:2])
            tail = [s["environments"][0]["object_xyz_m"][:2] for s in metrics["samples"]
                    if s["t"] >= metrics["samples"][-1]["t"] - 0.5]
            before_release = float(np.linalg.norm(before - np.asarray(plan["task_input"]["goal_xyz_m"][:2])) * 1000)
            release_shift = float(np.linalg.norm(np.median(tail, axis=0) - before) * 1000)
        rows.append({"index": result["index"], "episode": source["episode"], "object_name": source["object_name"],
                     "pick_place_passed": metrics["pick_place_passed"], "tracking_passed": metrics["tracking_passed"],
                     "placement_error_mm": physical["final_goal_xy_error_m"] * 1000,
                     "pre_release_placement_error_mm": before_release, "post_release_shift_mm": release_shift,
                     "grasp_slip_mm": physical["grasp_slip_max_m"] * 1000 if physical["grasp_slip_max_m"] is not None else None,
                     "bilateral_airborne_hold_s": physical["bilateral_airborne_hold_s"],
                     "object_controller_path_error_p95_mm": float(np.percentile(errors, 95) * 1000),
                     "arm_position_error_p95_mm": metrics["position_error_p95_m"] * 1000,
                     "sim_completion_time_s": metrics["samples"][-1]["t"],
                     "metrics": str(directory / "metrics.json")})
        if "goal_yaw_rad" in plan["task_input"]:
            rows[-1].update(goal_yaw_error_deg=physical["final_goal_yaw_error_deg"],
                            upright_tilt_deg=physical["final_upright_tilt_deg"],
                            yaw_goal_passed=physical["yaw_goal_passed"],
                            pick_place_with_yaw_passed=physical["pick_place_with_yaw_passed"])
            if "carry_yaw_error_p95_deg" in physical:
                rows[-1]["carry_yaw_error_p95_deg"] = physical["carry_yaw_error_p95_deg"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["object_name"]].append(row)
    statistics = {}
    for name, group in grouped.items():
        errors = [r["placement_error_mm"] for r in group]
        statistics[name] = {"count": len(group), "pick_place_passed": sum(r["pick_place_passed"] for r in group),
                            "within_5_mm": sum(e <= 5 for e in errors),
                            "placement_error_median_mm": float(np.median(errors)),
                            "placement_error_p95_mm": float(np.percentile(errors, 95)),
                            "placement_error_max_mm": float(np.max(errors)),
                            "object_controller_path_p95_median_mm": float(np.median([r["object_controller_path_error_p95_mm"] for r in group]))}
        yaw_rows = [r for r in group if "goal_yaw_error_deg" in r]
        if yaw_rows:
            statistics[name].update(yaw_evaluated=len(yaw_rows), yaw_goal_passed=sum(r["yaw_goal_passed"] for r in yaw_rows),
                                    pick_place_with_yaw_passed=sum(r["pick_place_with_yaw_passed"] for r in yaw_rows),
                                    goal_yaw_error_median_deg=float(np.median([r["goal_yaw_error_deg"] for r in yaw_rows])),
                                    goal_yaw_error_max_deg=float(np.max([r["goal_yaw_error_deg"] for r in yaw_rows])))
    expected = set(state["identity"]["indices"])
    evaluated = {r["index"] for r in rows}
    return {"schema_version": "hmnd_dynamic_review_summary_v1", "dataset_clip_count": manifest["clip_count"],
            "requested_clip_count": len(expected), "evaluated_clip_count": len(rows),
            "full_dataset_evaluated": bool(state["completed"] and evaluated == expected == set(sources)),
            "pick_place_passed": sum(r["pick_place_passed"] for r in rows),
            "yaw_evaluated_clip_count": sum("goal_yaw_error_deg" in r for r in rows),
            "pick_place_with_yaw_passed": sum(r.get("pick_place_with_yaw_passed", False) for r in rows),
            "within_5_mm": sum(r["placement_error_mm"] <= 5 for r in rows), "by_object": statistics, "clips": rows,
            "notes": "Existing pass threshold is 25 mm final XY error plus contact/lift/slip/release/settling checks. The 5 mm count is an additional diagnostic, not a changed pass criterion. Object-path error is relative to the clearance-adjusted controller reference, not the original human path. Simulation metrics do not validate absolute perception accuracy. No RL decision is automatic."}


def main():
    parser = argparse.ArgumentParser(description="Summarize dynamic Isaac review, including actual errors rather than only pass flags.")
    parser.add_argument("--run", type=Path, required=True, help="Playlist run.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/datasets/pilot_030/manifest.json")
    args = parser.parse_args()
    try:
        summary = summarize(args.run, args.manifest)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Cannot summarize review: {error}\n")
    directory = args.run.parent
    write_json(directory / "summary.json", summary)
    lines = ["# Dynamic Controller Review", "", f"Evaluated: {summary['evaluated_clip_count']} / {summary['requested_clip_count']}. Full dataset evaluated: {summary['full_dataset_evaluated']}.",
             f"Existing pick/place checks passed: {summary['pick_place_passed']}. Placements within 5 mm: {summary['within_5_mm']}.", "",
             "| Object | Count | Passed | Within 5 mm | Median / p95 / max Placement mm |", "| --- | --- | --- | --- | --- |"]
    for name, stats in summary["by_object"].items():
        lines.append(f"| {name} | {stats['count']} | {stats['pick_place_passed']} | {stats['within_5_mm']} | {stats['placement_error_median_mm']:.2f} / {stats['placement_error_p95_mm']:.2f} / {stats['placement_error_max_mm']:.2f} |")
    lines.extend(["", summary["notes"], "", "Review all paired videos before deciding whether the controller is sufficient. Investigate perception/geometry/controller causes before adding residual RL. Keep teacher labels and dataset splits separate from this simulation review.", "",
                  "| Index | Object | Passed | Placement mm | Pre-Release mm | Release Shift mm | Object Reference Path p95 mm |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in summary["clips"]:
        before = f"{row['pre_release_placement_error_mm']:.2f}" if row['pre_release_placement_error_mm'] is not None else "-"
        shift = f"{row['post_release_shift_mm']:.2f}" if row['post_release_shift_mm'] is not None else "-"
        lines.append(f"| {row['index']} | {row['object_name']} | {row['pick_place_passed']} | {row['placement_error_mm']:.2f} | {before} | {shift} | {row['object_controller_path_error_p95_mm']:.2f} |")
    if summary["yaw_evaluated_clip_count"]:
        lines.extend(["", "## Upright Yaw", "", "Shape-axis error modulo 180 degrees, measured from actual object physics quaternions. "
                      "Yaw tolerance 10 degrees; final tilt tolerance 15 degrees. Position-only clips are not counted as yaw passes.", "",
                      f"Pick/place plus yaw passed: {summary['pick_place_with_yaw_passed']} / {summary['yaw_evaluated_clip_count']} yaw-enabled clips.", "",
                      "| Index | Goal Yaw Error deg (Tail p95) | Upright Tilt deg (Tail p95) | Pick/Place + Yaw |",
                      "| --- | --- | --- | --- |"])
        for row in summary["clips"]:
            if "goal_yaw_error_deg" in row:
                lines.append(f"| {row['index']} | {row['goal_yaw_error_deg']:.2f} | {row['upright_tilt_deg']:.2f} | {row['pick_place_with_yaw_passed']} |")
    (directory / "review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluated {summary['evaluated_clip_count']}; pick/place passed {summary['pick_place_passed']}; within 5 mm {summary['within_5_mm']}. Report: {directory / 'review_report.md'}")


if __name__ == "__main__":
    main()
