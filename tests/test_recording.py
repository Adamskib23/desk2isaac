from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_retarget.recording import CaptureTimer


def test_guided_timer_counts_down_before_starting_and_saving(capsys):
    timer = CaptureTimer(10, guided=True)
    assert not timer.update(100)
    assert timer.ready_at is None
    assert not timer.update(101, frame_ready=True)
    assert not timer.update(102, frame_ready=True)
    assert not timer.update(103, frame_ready=True)
    assert timer.started_at is None
    assert timer.update(104, frame_ready=True)
    assert timer.started_at == 104
    output = capsys.readouterr().out
    assert "starts in 3 s" in output
    assert "starts in 2 s" in output
    assert "starts in 1 s" in output
    assert "HOLD STILL - initial position" in output
    assert "initial position; hands clear" in output


def test_timer_phase_boundaries_ticks_and_completion(capsys):
    timer = CaptureTimer(10, guided=True, countdown_seconds=0)
    assert timer.update(100, frame_ready=True)
    assert timer.update(100.1, frame_ready=True)
    assert timer.update(101, frame_ready=True)
    assert timer.update(102, frame_ready=True)
    assert timer.update(107.99, frame_ready=True)
    assert timer.update(108, frame_ready=True)
    assert not timer.update(110, frame_ready=True)
    assert not timer.update(111, frame_ready=True)
    assert timer.finished
    output = capsys.readouterr().out
    assert output.count("[00.0") == 1
    assert "MOVE NOW" in output
    assert "HOLD STILL - final position" in output
    assert "pick, move, place, and RELEASE" in output
    assert "final position; object released, hands clear" in output
    assert output.count("DONE") == 1
    assert timer.metadata()["movement_seconds"] == 6
    assert [cue["phase"] for cue in timer.cue_events] == ["hold_start", "move", "hold_end", "done"]
    assert [cue["recording_time_s"] for cue in timer.cue_events] == [0, 2, 8, 10]


def test_timer_waits_for_a_valid_first_frame_after_countdown(capsys):
    timer = CaptureTimer(10, guided=True)
    assert not timer.update(10, frame_ready=True)
    assert not timer.update(13)
    assert timer.started_at is None
    assert timer.update(15, frame_ready=True)
    assert timer.started_at == 15
    assert "waiting for a valid camera frame" in capsys.readouterr().out
    assert not timer.update(25)
    assert timer.finished


def test_timer_still_finishes_when_camera_frames_stop():
    timer = CaptureTimer(1)
    assert timer.update(10, frame_ready=True)
    assert not timer.update(10.5)
    assert not timer.update(11)
    assert timer.finished


def test_timer_preserves_short_unguided_calibration_recordings(capsys):
    timer = CaptureTimer(2)
    assert timer.update(100, frame_ready=True)
    assert not timer.update(102)
    assert "GET READY" not in capsys.readouterr().out
    assert timer.metadata()["movement_seconds"] is None


@pytest.mark.parametrize("kwargs", [
    {"seconds": 0}, {"seconds": float("nan")}, {"seconds": float("inf")},
    {"seconds": 4, "guided": True}, {"seconds": 10, "countdown_seconds": -1},
    {"seconds": 10, "initial_hold_seconds": -1}, {"seconds": 10, "final_hold_seconds": float("nan")},
])
def test_timer_rejects_invalid_schedules(kwargs):
    with pytest.raises(ValueError):
        CaptureTimer(**kwargs)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def time(self):
        return 1_700_000_000 + self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def recorder():
    path = Path(__file__).resolve().parents[1] / "scripts/record_orbbec_episode.py"
    spec = importlib.util.spec_from_file_location("test_recorder_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_opencv_capture_uses_actual_frame_size_and_guidance_metadata(tmp_path, monkeypatch, recorder):
    clock = FakeClock()

    class Camera:
        released = False

        def isOpened(self):
            return True

        def get(self, _):
            assert not self.released
            return 20

        def read(self):
            clock.sleep(0.05)
            return True, np.full((24, 32, 3), 180, dtype=np.uint8)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(recorder, "time", clock)
    monkeypatch.setattr(recorder.cv2, "VideoCapture", lambda _: camera)
    timer = CaptureTimer(0.6, guided=True, countdown_seconds=1, initial_hold_seconds=0.1, final_hold_seconds=0.1)
    recorder.record_opencv(tmp_path, 0.6, 20, 0, timer=timer)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert camera.released
    assert (metadata["width"], metadata["height"]) == (32, 24)
    assert len(list((tmp_path / "rgb").glob("*.jpg"))) == len(metadata["timestamps_unix_s"]) > 5
    assert metadata["recording_guidance"]["completed"]
    assert metadata["timestamps_relative_s"][0] == 0
    assert max(metadata["timestamps_relative_s"]) < 0.6
    assert not metadata["interrupted"]


def test_ctrl_c_saves_partial_episode_metadata_and_releases_camera(tmp_path, monkeypatch, recorder):
    clock = FakeClock()

    class Camera:
        count = 0
        released = False

        def isOpened(self):
            return True

        def get(self, _):
            return 20

        def read(self):
            self.count += 1
            if self.count == 4:
                raise KeyboardInterrupt
            clock.sleep(0.05)
            return True, np.zeros((24, 32, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(recorder, "time", clock)
    monkeypatch.setattr(recorder.cv2, "VideoCapture", lambda _: camera)
    with pytest.raises(KeyboardInterrupt):
        recorder.record_opencv(tmp_path, 10, 20, 0)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert camera.released
    assert metadata["interrupted"]
    assert not metadata["recording_guidance"]["completed"]
    assert len(metadata["timestamps_unix_s"]) == 3


def test_orbbec_warms_up_before_countdown_and_saves_aligned_pairs(tmp_path, monkeypatch, recorder):
    clock = FakeClock()

    class Color:
        def get_width(self):
            return 32

        def get_height(self):
            return 24

        def get_format(self):
            return "rgb"

        def get_data(self):
            return np.full((24, 32, 3), 180, dtype=np.uint8).tobytes()

    class Depth(Color):
        def get_data(self):
            return np.full((24, 32), 500, dtype=np.uint16).tobytes()

        def get_depth_scale(self):
            return 1.0

    class Frames:
        def as_frame_set(self):
            return self

        def get_color_frame(self):
            return Color()

        def get_depth_frame(self):
            return Depth()

    class Pipeline:
        stopped = False
        count = 0

        def get_stream_profile_list(self, _):
            return SimpleNamespace(get_video_stream_profile=lambda *_: "profile", get_default_video_stream_profile=lambda: "profile")

        def enable_frame_sync(self):
            pass

        def start(self, _):
            pass

        def wait_for_frames(self, _):
            self.count += 1
            clock.sleep(0.05)
            return Frames()

        def get_camera_param(self):
            return SimpleNamespace(
                rgb_intrinsic=SimpleNamespace(fx=100, fy=100, cx=16, cy=12, width=32, height=24),
                rgb_distortion=SimpleNamespace(**{key: 0 for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")}),
            )

        def stop(self):
            self.stopped = True

    pipeline = Pipeline()
    sdk = SimpleNamespace(
        Pipeline=lambda: pipeline,
        Config=lambda: SimpleNamespace(enable_stream=lambda _: None, set_frame_aggregate_output_mode=lambda _: None),
        AlignFilter=lambda **_: SimpleNamespace(process=lambda frames: frames),
        OBFormat=SimpleNamespace(RGB="rgb", BGR="bgr", YUYV="yuyv", UYVY="uyvy", MJPG="mjpg"),
        OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE="full"),
        OBSensorType=SimpleNamespace(COLOR_SENSOR="color", DEPTH_SENSOR="depth"),
        OBStreamType=SimpleNamespace(COLOR_STREAM="color"),
    )
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
    monkeypatch.setattr(recorder, "time", clock)
    timer = CaptureTimer(0.6, guided=True, countdown_seconds=1, initial_hold_seconds=0.1, final_hold_seconds=0.1)
    recorder.record_orbbec(tmp_path, 0.6, 20, timer=timer)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert pipeline.stopped
    assert timer.ready_at >= 100 + 15 / 20
    assert timer.started_at >= timer.ready_at + 1
    rgb = sorted((tmp_path / "rgb").glob("*.jpg"))
    depth = sorted((tmp_path / "depth").glob("*.png"))
    assert [p.stem for p in rgb] == [p.stem for p in depth]
    assert len(rgb) == len(metadata["timestamps_relative_s"]) > 5
    assert metadata["depth_aligned_to"] == "color"
    assert metadata["depth_unit_m"] == 0.001
    assert metadata["recording_guidance"]["completed"]
    assert len(metadata["color_distortion"]) == 8
