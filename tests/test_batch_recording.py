from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch():
    spec = importlib.util.spec_from_file_location("test_batch_script", ROOT / "scripts/record_pilot_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path, **overrides):
    values = dict(
        out=str(tmp_path / "batch_001"), count=3, seconds=10.0, fps=30.0,
        objects_config=str(ROOT / "configs/objects.rl_pilot.yaml"), backend="orbbec",
        camera_index=0, countdown=3, initial_hold_seconds=2.0, final_hold_seconds=2.0,
        beep=False, resume=False, retake_clip=None, min_free_gb=0.0,
    )
    return argparse.Namespace(**{**values, **overrides})


def feed(monkeypatch, *choices):
    responses = iter(choices)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))


def write_capture(batch, episode, *, interrupted=False, aligned=True, completed=True):
    (episode / "rgb").mkdir(exist_ok=True)
    for i in range(3):
        (episode / "rgb" / f"{i:06d}.jpg").write_bytes(b"rgb frame")
    if aligned:
        (episode / "depth").mkdir(exist_ok=True)
        for i in range(3):
            (episode / "depth" / f"{i:06d}.png").write_bytes(b"depth frame")
    metadata = {
        "schema_version": "hmnd_rgbd_episode_v1", "fps": 30, "width": 32, "height": 24,
        "depth_aligned_to": "color" if aligned else None,
        "timestamps_relative_s": [0.0, 5.0, 9.99],
        "recording_guidance": {"completed": completed, "requested_seconds": 10.0},
        "interrupted": interrupted, "color_intrinsics": {"fx": 100, "fy": 100},
    }
    batch.write_json(episode / "metadata.json", metadata)
    return metadata


def successful_camera(batch, calls):
    def capture(episode, seconds, fps, *, timer):
        assert timer.guided and timer.countdown_seconds == 3
        calls.append(episode)
        write_capture(batch, episode)
    return capture


def manifest(args):
    return json.loads((Path(args.out) / "batch.json").read_text())


def test_batch_labels_mixed_objects_and_keeps_camera_metadata(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path)
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "3", "blue_sphere", "2")
    assert batch.run_batch(args) == 0
    data = manifest(args)
    assert [c["object_name"] for c in data["clips"]] == ["yellow_mustard_tin", "blue_sphere", "black_box"]
    assert all(c["status"] == "completed" for c in data["clips"])
    assert data["objects"][-1]["size_m"] == [0.051, 0.035, 0.085]
    for entry, episode in zip(data["clips"], calls):
        metadata = batch.read_json(episode / "metadata.json")
        assert metadata["action_object"] == entry["object_name"]
        assert metadata["recording_batch"]["batch_id"] == data["batch_id"]
        assert metadata["recording_batch"]["clip_index"] == entry["clip_index"]
        assert metadata["color_intrinsics"] == {"fx": 100, "fy": 100}
        assert metadata["depth_aligned_to"] == "color"
    assert not list(Path(args.out).glob("*.tmp"))


def test_invalid_choices_do_not_start_recording(tmp_path, monkeypatch, batch, capsys):
    args = arguments(tmp_path, count=1)
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "", "0", "4", "glass", "1")
    assert batch.run_batch(args) == 0
    assert len(calls) == 1
    assert capsys.readouterr().out.count("Recording has not started") == 4


def test_quit_and_resume_skip_completed_captures_without_rewriting(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=2)
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "1", "q")
    assert batch.run_batch(args) == 0
    first = calls[0]
    metadata_stat = (first / "metadata.json").stat().st_mtime_ns
    old_rgb = (first / "rgb/000000.jpg").read_bytes()
    args.resume = True
    feed(monkeypatch, "3")
    assert batch.run_batch(args) == 0
    assert len(calls) == 2
    assert (first / "metadata.json").stat().st_mtime_ns == metadata_stat
    assert (first / "rgb/000000.jpg").read_bytes() == old_rgb
    assert [c["clip_index"] for c in manifest(args)["clips"]] == [1, 2]


def test_ctrl_c_preserves_partial_clip_and_resume_uses_new_folder(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=1)
    calls = []

    def interrupted(episode, *_, **__):
        calls.append(episode)
        write_capture(batch, episode, interrupted=True, completed=False)
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "record_orbbec", interrupted)
    feed(monkeypatch, "3")
    assert batch.run_batch(args) == 130
    first = calls[0]
    assert manifest(args)["clips"][0]["status"] == "interrupted"
    assert batch.read_json(first / "metadata.json")["action_object"] == "yellow_mustard_tin"
    before = (first / "metadata.json").read_bytes()
    args.resume = True
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "1")
    assert batch.run_batch(args) == 0
    assert len(calls) == 2 and first != calls[1]
    assert (first / "metadata.json").read_bytes() == before
    assert [c["status"] for c in manifest(args)["clips"]] == ["interrupted", "completed"]


def test_retake_excludes_only_selected_clip_and_keeps_original_files(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=2)
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "1", "2")
    assert batch.run_batch(args) == 0
    originals = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for episode in calls for path in episode.rglob("*") if path.is_file()
    }
    other_entry = manifest(args)["clips"][1].copy()
    args.resume = True
    args.retake_clip = 1
    feed(monkeypatch, "1")
    assert batch.run_batch(args) == 0
    entries = manifest(args)["clips"]
    assert [entry["status"] for entry in entries] == ["rejected", "completed", "completed"]
    assert entries[0]["rejection_reason"] == "operator_requested_retake"
    assert entries[0]["rejected_at_utc"]
    assert entries[1] == other_entry
    assert calls[-1].name == "clip_003_blue_sphere"
    for path, (data, mtime) in originals.items():
        assert path.read_bytes() == data
        assert path.stat().st_mtime_ns == mtime
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Batch is already full"))
    assert batch.run_batch(args) == 0
    assert manifest(args)["clips"] == entries
    assert len(calls) == 3


def test_retake_can_stop_before_recording_and_resume_without_retake_flag(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=1)
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "1")
    assert batch.run_batch(args) == 0
    args.resume = True
    args.retake_clip = 1
    feed(monkeypatch, "q")
    assert batch.run_batch(args) == 0
    rejected = manifest(args)["clips"][0].copy()
    assert rejected["status"] == "rejected" and len(calls) == 1
    args.retake_clip = None
    feed(monkeypatch, "1")
    assert batch.run_batch(args) == 0
    assert manifest(args)["clips"][0] == rejected
    assert calls[-1].name == "clip_002_blue_sphere"


@pytest.mark.parametrize("retake_clip", [0, -1, 1])
def test_retake_requires_resume_and_positive_index(tmp_path, batch, retake_clip):
    args = arguments(tmp_path, retake_clip=retake_clip)
    with pytest.raises(ValueError, match="requires --resume"):
        batch.run_batch(args)
    assert not Path(args.out).exists()


def test_unknown_retake_does_not_change_manifest_or_open_camera(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=1)
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "1")
    assert batch.run_batch(args) == 0
    before = (Path(args.out) / "batch.json").read_bytes()
    args.resume = True
    args.retake_clip = 2
    with pytest.raises(ValueError, match="completed or already rejected"):
        batch.run_batch(args)
    assert (Path(args.out) / "batch.json").read_bytes() == before
    assert len(calls) == 1


def test_capture_error_is_saved_and_no_later_capture_runs(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path)

    def failed(episode, *_, **__):
        (episode / "rgb").mkdir()
        (episode / "rgb/000000.jpg").write_bytes(b"partial")
        raise RuntimeError("camera disconnected")

    monkeypatch.setattr(batch, "record_orbbec", failed)
    feed(monkeypatch, "2")
    with pytest.raises(RuntimeError, match="camera disconnected"):
        batch.run_batch(args)
    data = manifest(args)
    assert len(data["clips"]) == 1 and data["clips"][0]["status"] == "failed"
    assert data["clips"][0]["object_name"] == "black_box"
    assert (Path(args.out) / data["clips"][0]["episode"] / "rgb/000000.jpg").read_bytes() == b"partial"


@pytest.mark.parametrize("complete", [True, False])
def test_resume_recovers_crashed_manifest_without_overwriting_frames(tmp_path, monkeypatch, batch, complete):
    args = arguments(tmp_path, count=1)
    feed(monkeypatch, "q")
    assert batch.run_batch(args) == 0
    data = manifest(args)
    old = Path(args.out) / "clip_001_yellow_mustard_tin"
    old.mkdir()
    write_capture(batch, old, completed=complete)
    data["clips"].append({"clip_index": 1, "object_name": "yellow_mustard_tin", "episode": old.name, "status": "recording"})
    batch.write_atomic_json(Path(args.out) / "batch.json", data)
    args.resume = True
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "2")
    assert batch.run_batch(args) == 0
    assert len(calls) == (0 if complete else 1)
    assert manifest(args)["clips"][0]["status"] == ("completed" if complete else "incomplete")
    assert (old / "rgb/000000.jpg").read_bytes() == b"rgb frame"


def test_existing_batch_requires_resume_and_original_settings(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path)
    feed(monkeypatch, "q")
    assert batch.run_batch(args) == 0
    before = (Path(args.out) / "batch.json").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        batch.run_batch(args)
    args.resume = True
    args.seconds = 11
    with pytest.raises(ValueError, match="original recording settings"):
        batch.run_batch(args)
    assert (Path(args.out) / "batch.json").read_bytes() == before


def test_resume_does_not_overwrite_a_corrected_episode_label(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=1)
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "3")
    assert batch.run_batch(args) == 0
    metadata = batch.read_json(calls[0] / "metadata.json")
    metadata["action_object"] = "blue_sphere"
    batch.write_json(calls[0] / "metadata.json", metadata)
    args.resume = True
    with pytest.raises(ValueError, match="label disagrees"):
        batch.run_batch(args)
    assert batch.read_json(calls[0] / "metadata.json")["action_object"] == "blue_sphere"


def test_resume_rejects_unsafe_manifest_paths(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path)
    feed(monkeypatch, "q")
    assert batch.run_batch(args) == 0
    data = manifest(args)
    data["clips"].append({"clip_index": 1, "object_name": "blue_sphere", "episode": "../outside", "status": "recording"})
    batch.write_atomic_json(Path(args.out) / "batch.json", data)
    args.resume = True
    with pytest.raises(ValueError, match="Invalid or duplicated"):
        batch.run_batch(args)


def test_low_disk_space_stops_before_camera_or_prompt(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, min_free_gb=1)
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Should not prompt when disk is full"))
    with pytest.raises(ValueError, match="GiB free"):
        batch.run_batch(args)
    assert manifest(args)["clips"] == []


def test_simultaneous_writer_is_rejected(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path)
    Path(args.out).mkdir()
    with (Path(args.out) / ".batch.lock").open("a") as lock:
        batch.fcntl.flock(lock, batch.fcntl.LOCK_EX | batch.fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another recorder"):
            batch.run_batch(args)


@pytest.mark.parametrize("overrides", [
    {"count": 0}, {"fps": float("nan")}, {"seconds": 4},
    {"min_free_gb": -1}, {"countdown": -1},
])
def test_invalid_batch_settings_do_not_create_outputs(tmp_path, batch, overrides):
    args = arguments(tmp_path, **overrides)
    with pytest.raises(ValueError):
        batch.run_batch(args)
    assert not Path(args.out).exists()


@pytest.mark.parametrize("failure", ["missing_depth", "non_increasing_time", "stopped_frames"])
def test_incomplete_capture_is_not_counted(tmp_path, monkeypatch, batch, failure):
    args = arguments(tmp_path, count=1)

    def incomplete(episode, *_, **__):
        metadata = write_capture(batch, episode)
        if failure == "missing_depth":
            (episode / "depth/000002.png").unlink()
        elif failure == "non_increasing_time":
            metadata["timestamps_relative_s"] = [0, 0, 9.99]
        else:
            metadata["timestamps_relative_s"] = [0, 1, 2]
        batch.write_json(episode / "metadata.json", metadata)

    monkeypatch.setattr(batch, "record_orbbec", incomplete)
    feed(monkeypatch, "1")
    with pytest.raises(RuntimeError, match="incomplete"):
        batch.run_batch(args)
    assert manifest(args)["clips"][0]["status"] == "failed"


def test_opencv_batch_can_capture_without_depth(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=1, backend="opencv")

    def capture(episode, seconds, fps, camera_index, *, timer):
        assert camera_index == 0 and timer.guided
        write_capture(batch, episode, aligned=False)

    monkeypatch.setattr(batch, "record_opencv", capture)
    feed(monkeypatch, "2")
    assert batch.run_batch(args) == 0
    assert manifest(args)["clips"][0]["status"] == "completed"


def test_eof_at_prompt_preserves_empty_batch(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path)

    def eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert batch.run_batch(args) == 0
    assert manifest(args)["clips"] == []


def test_existing_unlisted_clip_is_not_modified(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=1)
    feed(monkeypatch, "q")
    assert batch.run_batch(args) == 0
    old = Path(args.out) / "clip_001_blue_sphere"
    old.mkdir()
    write_capture(batch, old)
    before = (old / "metadata.json").read_bytes()
    args.resume = True
    feed(monkeypatch, "1")
    with pytest.raises(FileExistsError):
        batch.run_batch(args)
    assert (old / "metadata.json").read_bytes() == before
    assert manifest(args)["clips"][0]["status"] == "failed"


def test_ctrl_c_during_folder_creation_reserves_a_fresh_retry_index(tmp_path, monkeypatch, batch):
    args = arguments(tmp_path, count=1)
    mkdir = Path.mkdir

    def interrupt_folder(self, *args, **kwargs):
        if self.name == "clip_001_blue_sphere":
            raise KeyboardInterrupt
        return mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", interrupt_folder)
    feed(monkeypatch, "1")
    assert batch.run_batch(args) == 130
    assert manifest(args)["clips"][0]["status"] == "interrupted"
    args.resume = True
    calls = []
    monkeypatch.setattr(batch, "record_orbbec", successful_camera(batch, calls))
    feed(monkeypatch, "1")
    assert batch.run_batch(args) == 0
    assert calls[0].name == "clip_002_blue_sphere"
