import importlib.util
from pathlib import Path

import pytest

from humanoid_retarget.io import read_json, write_json
from humanoid_retarget.mask_tracking import file_sha256


@pytest.fixture
def review(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_isaac_review.py"
    spec = importlib.util.spec_from_file_location("isaac_review_summary_tool", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    clips, results = [], []
    for index, (name, error) in enumerate((("black_box", 0.002), ("blue_sphere", 0.022)), start=1):
        trajectory = f"tasks/{index}.json"
        clips.append({"index": index, "episode": f"episode_{index}", "object_name": name, "trajectory": trajectory})
        directory = tmp_path / str(index)
        samples = [{"t": float(i), "phase": phase, "target_ee_xyz_m": [0.5, 0, 1.1714],
                    "environments": [{"object_xyz_m": [0.5 + (error if i == 3 else 0), 0, 1.068]}]}
                   for i, phase in enumerate(("carry_guidance", "descend_to_place", "place_hold", "retreat"))]
        metrics = {"mode": "dynamic_contact_grasp_with_guidance_ik", "completed": True,
                   "object_pose_commands_during_run": 0, "source_trajectory": str(tmp_path / trajectory),
                   "pick_place_passed": True, "tracking_passed": True, "position_error_p95_m": 0.004,
                   "samples": samples,
                   "physical_results": [{"final_goal_xy_error_m": error, "grasp_slip_max_m": 0.001,
                                         "bilateral_airborne_hold_s": 1.5}]}
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "metrics_plan.json", {"task_input": {"goal_xyz_m": [0.5, 0, 1.068]},
                   "waypoints": [{"phase": "carry_guidance", "ee_xyz_m": [0.5, 0, 1.1714], "object_xyz_m": [0.5, 0, 1.068]}]})
        results.append({"index": index, "completed": True, "directory": str(directory)})
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {"clip_count": 2, "clips": clips})
    run_path = tmp_path / "run.json"
    write_json(run_path, {"mode": "dynamic", "completed": True, "clips": results,
                         "identity": {"manifest_sha256": file_sha256(manifest_path), "indices": [1, 2]}})
    return module, run_path, manifest_path


def test_summary_distinguishes_baseline_pass_from_precise_placement(review):
    module, run, manifest = review
    summary = module.summarize(run, manifest)
    assert summary["full_dataset_evaluated"]
    assert summary["pick_place_passed"] == 2 and summary["within_5_mm"] == 1
    sphere = summary["clips"][1]
    assert sphere["placement_error_mm"] == 22
    assert sphere["pre_release_placement_error_mm"] == pytest.approx(0)
    assert sphere["post_release_shift_mm"] == pytest.approx(22)
    assert sphere["object_controller_path_error_p95_mm"] == pytest.approx(0)


@pytest.mark.parametrize("change", ["subset", "partial"])
def test_subset_or_partial_review_is_not_full_dataset_evaluation(review, change):
    module, run, manifest = review
    state = read_json(run)
    if change == "subset":
        state["identity"]["indices"] = [1]
        state["clips"] = state["clips"][:1]
    else:
        state["completed"] = False
    write_json(run, state)
    assert not module.summarize(run, manifest)["full_dataset_evaluated"]


@pytest.mark.parametrize("change", ["kinematic", "pose_commands", "wrong_source", "manifest", "duplicates"])
def test_invalid_physical_review_cannot_be_reported_as_success(review, change):
    module, run, manifest = review
    state = read_json(run)
    if change in ("manifest", "duplicates"):
        if change == "manifest":
            state["identity"]["manifest_sha256"] = "wrong"
        else:
            state["clips"].append(state["clips"][0])
        write_json(run, state)
    else:
        path = Path(state["clips"][0]["directory"]) / "metrics.json"
        metrics = read_json(path)
        if change == "kinematic":
            metrics["mode"] = "kinematic_object_pose_replay_with_pose_ik"
        elif change == "pose_commands":
            metrics["object_pose_commands_during_run"] = 1
        else:
            metrics["source_trajectory"] = "wrong.json"
        write_json(path, metrics)
    with pytest.raises(ValueError):
        module.summarize(run, manifest)
