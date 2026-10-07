from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from humanoid_retarget.dataset_review import (
    checked_replay_task, comparison_frame, load_review_manifest, reference_frame, reference_knots,
)
from humanoid_retarget.io import write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.physical import build_physical_plan


def task_and_labels():
    task = {"schema_version": "hmnd_rgbd_retarget_v1", "fps": 30,
            "object": {"size_m": [0.05555, 0.045, 0.031]},
            "workspace": {"retarget": {"sim_table_z_m": 1.05}},
            "events": {"pick_frame": 15, "place_frame": 75},
            "waypoints": [{"t": 0, "phase": "ready", "object_xyz_m": [0.5, 0.1, 1.0655], "ee_xyz_m": [0.5, 0.1, 1.31]},
                          {"t": 3, "phase": "retreat", "object_xyz_m": [0.6, 0, 1.0655]}],
            "object_trace": [{"frame_idx": i, "t": i / 30, "visible": True,
                              "sim_xyz_m": [0.5 + i * 0.001, 0.1 - i * 0.001, 1.1455 if i == 40 else 1.0655]}
                             for i in range(90)]}
    labels = {"frames": [{"frame_idx": i, "observed": True, "position_label_candidate": True, "exclusion_reasons": []}
                          for i in range(90)]}
    return task, labels


def test_replay_guidance_gates_exclusions_without_changing_source():
    task, labels = task_and_labels()
    original = deepcopy(task)
    labels["frames"][40].update(position_label_candidate=False, exclusion_reasons=["kalman_prediction"])
    result = checked_replay_task(task, labels)
    assert task == original
    assert 40 not in [r["frame_idx"] for r in result["object_trace"]]
    assert result["events"]["guidance_excluded_frames"] == [40]
    assert result["events"]["training_ready"] is False
    physical = build_physical_plan(result)
    assert 40 not in [r["source_frame_idx"] for r in physical["task_input"]["guidance_trajectory"]]


@pytest.mark.parametrize("invalid", ["missing_endpoint", "duplicate_frame", "accept_excluded", "accept_unobserved", "no_carry"])
def test_review_task_cannot_accept_invalid_labels(invalid):
    task, labels = task_and_labels()
    if invalid == "missing_endpoint":
        for row in labels["frames"][-12:]:
            row["position_label_candidate"] = False
    elif invalid == "duplicate_frame":
        labels["frames"][20]["frame_idx"] = 19
    elif invalid == "accept_excluded":
        labels["frames"][20]["exclusion_reasons"] = ["weak_depth_area_support"]
    elif invalid == "accept_unobserved":
        labels["frames"][20]["observed"] = False
    else:
        for row in labels["frames"][15:76]:
            row["position_label_candidate"] = False
    with pytest.raises(ValueError):
        checked_replay_task(task, labels)


def test_reference_alignment_uses_capture_times_and_pauses_airborne_hold():
    task, labels = task_and_labels()
    physical = build_physical_plan(checked_replay_task(task, labels))
    capture_times = [i / 29.7 for i in range(90)]
    knots = reference_knots(physical, capture_times)
    assert reference_frame(-1, knots, capture_times) == 0
    assert reference_frame(100, knots, capture_times) == 89
    hold = next(wp for wp in physical["waypoints"] if wp["phase"] == "airborne_hold")
    assert reference_frame(hold["t"] - 0.5, knots, capture_times) == 40
    assert reference_frame(hold["t"], knots, capture_times) == 40
    mapped = [reference_frame(t, knots, capture_times) for t in np.linspace(0, 20, 300)]
    assert all(b >= a for a, b in zip(mapped, mapped[1:]))


@pytest.mark.parametrize("invalid", ["timestamp_order", "missing_frame", "outside_frame"])
def test_bad_reference_alignment_is_not_silently_clamped(invalid):
    task, labels = task_and_labels()
    physical = build_physical_plan(checked_replay_task(task, labels))
    capture_times = [i / 30 for i in range(90)]
    if invalid == "timestamp_order":
        capture_times[10] = capture_times[9]
    elif invalid == "missing_frame":
        physical["task_input"]["guidance_trajectory"][0]["source_frame_idx"] = None
    else:
        physical["task_input"]["guidance_trajectory"][0]["source_frame_idx"] = 900
    with pytest.raises(ValueError):
        reference_knots(physical, capture_times)


def test_comparison_preserves_both_panels_and_stable_dimensions():
    source = np.full((1080, 1920, 3), (20, 90, 200), np.uint8)
    simulation = np.full((720, 960, 3), (150, 30, 60), np.uint8)
    frame = comparison_frame(source, simulation, title="01/30 box", capture_t=1.2, sim_t=3.4)
    assert frame.shape == (720, 1920, 3)
    assert np.array_equal(frame[360, 400], source[540, 800])
    assert np.array_equal(frame[360, 1400], simulation[360, 440])


@pytest.mark.parametrize("invalid", [None, "hash", "escaped", "unbound_task", "duplicate", "not_ready"])
def test_manifest_requires_unique_fingerprint_bound_paths(tmp_path, invalid):
    metadata = tmp_path / "episode/metadata.json"
    write_json(metadata, {})
    task = tmp_path / "task.json"
    write_json(task, {})
    row = {"index": 1, "episode": "episode", "trajectory": "task.json", "candidate_labels": "task.json",
           "annotated_video": "task.json", "file_sha256": {"episode/metadata.json": file_sha256(metadata), "task.json": file_sha256(task)}}
    manifest = {"schema_version": "hmnd_prepared_review_dataset_v1", "ready_for_isaac_review": True, "clip_count": 1, "clips": [row]}
    if invalid == "hash":
        row["file_sha256"]["task.json"] = "wrong"
    elif invalid == "escaped":
        row["file_sha256"]["../outside.json"] = "wrong"
    elif invalid == "unbound_task":
        row["trajectory"] = "unbound.json"
    elif invalid == "duplicate":
        manifest["clips"].append(deepcopy(row))
        manifest["clip_count"] = 2
    elif invalid == "not_ready":
        manifest["ready_for_isaac_review"] = False
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    if invalid:
        with pytest.raises(ValueError):
            load_review_manifest(path, tmp_path)
    else:
        assert load_review_manifest(path, tmp_path)["clip_count"] == 1
