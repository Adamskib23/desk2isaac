from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.io import write_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.sam2_dataset import capture_issues, image_boundary_frames, label_candidates, propose_prompt
from humanoid_retarget.types import Detection, ObjectSpec


def _calibration():
    return WorkspaceCalibration(
        (160, 120), ((0, 0), (159, 0), (159, 119), (0, 119)),
        ((-0.4, 0.3), (0.4, 0.3), (0.4, -0.3), (-0.4, -0.3)),
        workspace_outline_m=((-0.15, 0.15), (0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)),
        camera_matrix_px=((200, 0, 80), (0, 200, 60), (0, 0, 1)),
        camera_to_desk_m=((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0.8), (0, 0, 0, 1)),
    )


def test_initial_prompt_proposal_uses_foreground_not_larger_colour_shadow():
    image = np.full((120, 160, 3), 150, np.uint8)
    image[45:70, 70:95] = (0, 0, 255)
    image[10:40, 10:60] = (0, 0, 255)
    depth = np.full((120, 160), 0.8)
    depth[45:70, 70:95] = 0.77
    spec = ObjectSpec("box", "cuboid", (0.04, 0.04, 0.03), [((0, 100, 100), (10, 255, 255))], min_area_px=20)
    box = propose_prompt(image, depth, spec, _calibration())
    assert box[0] < 70 and box[1] < 45 and box[2] > 95 and box[3] > 70
    assert box[0] > 60 and box[1] > 35


def test_prompt_proposal_refuses_unaligned_depth_or_no_instance():
    spec = ObjectSpec("box", "cuboid", (0.04, 0.04, 0.03), [], min_area_px=20)
    image = np.zeros((120, 160, 3), np.uint8)
    with pytest.raises(ValueError, match="aligned"):
        propose_prompt(image, np.ones((60, 80)), spec, _calibration())
    with pytest.raises(ValueError, match="No initial"):
        propose_prompt(image, np.full((120, 160), 0.8), spec, _calibration())


def test_position_candidates_never_include_missing_weak_predicted_or_switch_frames():
    base = Detection(0, 0, "box", (80, 60), (0, 0, 0.05), 1000, 1.0, position_source="fit")
    rows = [base, replace(base, frame_idx=1, t=0.04, desk_xyz_m=(0, 0, 0.058), position_source="surface"),
            replace(base, frame_idx=2, t=0.08, position_source="surface", desk_xyz_m=(0, 0, 0.058)),
            replace(base, frame_idx=3, t=0.12, confidence=0.5, position_source="surface", desk_xyz_m=(0, 0, 0.058))]
    raw = {"detections": [asdict(d) for d in rows], "metadata": {"fps": 25, "frame_diagnostics": [
        {"frame_idx": i, "t": i * 0.04, "status": "observed" if i < 4 else "missing_depth"} for i in range(5)]}}
    filtered = {"detections": [asdict(replace(d, visible=False) if d.frame_idx == 2 else d) for d in rows]}
    candidates, summary = label_candidates(raw, filtered, {"training_ready": True, "frames": 5})
    assert not any(r["position_label_candidate"] for r in candidates)
    assert candidates[2]["filtered_desk_xyz_m"] is None
    assert candidates[4]["raw_desk_xyz_m"] is None
    assert summary["height_switch_frames"] == [0, 1] and not summary["training_ready"]
    assert candidates[1]["capture_t_s"] == 0.04


def test_large_adjacent_steps_are_review_flags_not_hidden_labels():
    rows = [Detection(i, i / 30, "box", (80, 60), (0.1 if i == 1 else 0, 0, 0.05), 1000, 1)
            for i in range(3)]
    raw = {"detections": [asdict(d) for d in rows], "metadata": {"fps": 30, "frame_diagnostics": [
        {"frame_idx": i, "t": i / 30, "status": "observed"} for i in range(3)]}}
    labels, summary = label_candidates(raw, raw, {"frames": 3})
    assert summary["large_step_frames"] == [0, 1, 2]
    assert all("large_adjacent_step_review" in r["exclusion_reasons"] for r in labels)


def test_capture_integrity_does_not_accept_complete_flag_without_pairs(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    for i in range(3):
        cv2.imwrite(str(tmp_path / "rgb" / f"{i:06d}.jpg"), np.zeros((10, 10, 3), np.uint8))
    metadata = {"interrupted": False, "recording_guidance": {"completed": True},
                "timestamps_relative_s": [0, 0.1, 0.2], "depth_aligned_to": "color"}
    assert "Missing or unmatched RGB/depth frames" in capture_issues(tmp_path, metadata)


def test_clipped_masks_never_supply_candidate_positions(tmp_path):
    mask = np.zeros((40, 60), np.uint8)
    mask[10:20, 20:30] = 255
    cv2.imwrite(str(tmp_path / "000000.png"), mask)
    mask[:20, 20:30] = 255
    cv2.imwrite(str(tmp_path / "000001.png"), mask)
    clipped = image_boundary_frames(tmp_path, 2)
    assert clipped == [1]
    rows = [Detection(i, i / 30, "box", (20, 20), (0, 0, 0.05), 1000, 1) for i in range(2)]
    payload = {"detections": [asdict(d) for d in rows], "metadata": {"fps": 30, "frame_diagnostics": [
        {"frame_idx": i, "t": i / 30, "status": "observed"} for i in range(2)]}}
    labels, summary = label_candidates(payload, payload, {"frames": 2}, clipped)
    assert labels[0]["position_label_candidate"] and not labels[1]["position_label_candidate"]
    assert summary["image_boundary_frames"] == [1]


@pytest.mark.parametrize("change", ["source_output", "escaped_output", "escaped_episode", "moved_plan"])
def test_execute_plan_cannot_write_sources_or_escape_output_root(tmp_path, change):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_sam2_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_sam2_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "source", tmp_path / "derived"
    plan = {"source": str(source), "output_dir": str(output),
            "clips": [{"episode": str(source / "clip"), "output_dir": str(output / "clip")}]}
    plan_path = output / "plan.json"
    if change == "source_output":
        plan["output_dir"] = str(source / "derived")
        plan_path = source / "derived/plan.json"
    elif change == "escaped_output":
        plan["clips"][0]["output_dir"] = str(tmp_path / "outside")
    elif change == "escaped_episode":
        plan["clips"][0]["episode"] = str(tmp_path / "outside")
    else:
        plan_path = tmp_path / "moved/plan.json"
    with pytest.raises(ValueError):
        module.validate_plan_locations(plan, plan_path)


def test_batch_runner_preserves_venv_python_symlink(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_sam2_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_sam2_environment_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    source, output = tmp_path / "source", tmp_path / "derived"
    config = tmp_path / "config.json"
    write_json(config, {})
    plan = {"schema_version": "hmnd_sam2_dataset_preparation_v1", "source": str(source), "output_dir": str(output),
            "clips": [{"episode": str(source / "clip"), "relative_episode": "clip", "prior_retained": False,
                       "output_dir": str(output / "clip"), "status": "pending"}]}
    for key in ("workspace_config", "objects_config", "selection_path"):
        plan[key] = str(config)
        plan[{"selection_path": "selection_sha256"}.get(key, key + "_sha256")] = file_sha256(config)
    plan_path = output / "plan.json"
    write_json(plan_path, plan)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append(command))
    def fake_process(row, plan, executable, width):
        assert executable == python.absolute() and executable != python.resolve()
        row.update(status="prepared", quality={"position_label_candidate_frames": 3, "frames": 3, "task_generated": True})
    monkeypatch.setattr(module, "process_clip", fake_process)
    monkeypatch.setattr(module, "write_report", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["prepare", "--execute", str(plan_path), "--reviewed-prompts", "--sam-python", str(python)])
    module.main()
    assert calls[0][0] == str(python.absolute())


@pytest.mark.parametrize("changed", ["rgb", "mask", "depth", "manifest", "workspace", "objects"])
def test_cached_preparation_refuses_changed_tracking_inputs(tmp_path, changed):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_sam2_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_sam2_cache_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episode, output = tmp_path / "source", tmp_path / "derived"
    masks, tracking = output / "masks", output / "tracking"
    for folder in (episode / "rgb", episode / "depth", masks, tracking):
        folder.mkdir(parents=True, exist_ok=True)
    write_json(episode / "metadata.json", {})
    entries, diagnostics = [], []
    for index in range(3):
        name = f"{index:06d}"
        rgb, depth, mask = episode / "rgb" / f"{name}.jpg", episode / "depth" / f"{name}.png", masks / f"{name}.png"
        cv2.imwrite(str(rgb), np.zeros((40, 60, 3), np.uint8))
        cv2.imwrite(str(depth), np.full((40, 60), 800, np.uint16))
        cv2.imwrite(str(mask), np.zeros((40, 60), np.uint8))
        entries.append({"frame_idx": index, "rgb_name": rgb.name, "rgb_sha256": file_sha256(rgb), "mask_sha256": file_sha256(mask)})
        diagnostics.append({"frame_idx": index, "depth_sha256": file_sha256(depth)})
    write_json(masks / "manifest.json", {"completed": True, "frames": entries})
    metadata = {"mask_manifest_sha256": file_sha256(masks / "manifest.json"), "workspace_config_sha256": "workspace",
                "objects_config_sha256": "objects", "frame_diagnostics": diagnostics}
    row = {"episode": str(episode), "output_dir": str(output), "source_metadata_sha256": file_sha256(episode / "metadata.json"),
           "prompt_frame": 0, "prompt_rgb_sha256": file_sha256(episode / "rgb/000000.jpg")}
    plan = {"workspace_config_sha256": "workspace", "objects_config_sha256": "objects"}
    if changed in ("rgb", "mask", "depth"):
        target = {"rgb": episode / "rgb/000001.jpg", "mask": masks / "000001.png", "depth": episode / "depth/000001.png"}[changed]
        cv2.imwrite(str(target), np.ones((40, 60), np.uint8))
    else:
        key = {"manifest": "mask_manifest_sha256", "workspace": "workspace_config_sha256", "objects": "objects_config_sha256"}[changed]
        metadata[key] = "changed"
    write_json(tracking / "raw_detections.json", {"metadata": metadata})
    write_json(tracking / "detections.json", {})
    write_json(tracking / "tracking_quality.json", {"frames": 3})
    with pytest.raises(ValueError, match="fingerprint"):
        module.process_clip(row, plan, Path(sys.executable), 1280)
    assert not (output / "position_label_candidates.json").exists()
