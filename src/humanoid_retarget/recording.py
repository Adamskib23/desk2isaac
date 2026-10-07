from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class CaptureTimer:
    seconds: float
    guided: bool = False
    countdown_seconds: int = 3
    initial_hold_seconds: float = 2.0
    final_hold_seconds: float = 2.0
    beep: bool = False
    ready_at: float | None = field(default=None, init=False)
    started_at: float | None = field(default=None, init=False)
    finished: bool = field(default=False, init=False)
    cue_events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _last_display: tuple[str, int] | None = field(default=None, init=False)
    _last_phase: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("--seconds must be finite and positive")
        if self.countdown_seconds < 0:
            raise ValueError("--countdown must be non-negative")
        if not all(math.isfinite(v) and v >= 0 for v in (self.initial_hold_seconds, self.final_hold_seconds)):
            raise ValueError("Hold durations must be finite and non-negative")
        if self.guided and self.initial_hold_seconds + self.final_hold_seconds >= self.seconds:
            raise ValueError("Guided capture must be longer than the initial and final holds combined")

    def update(self, now: float, *, frame_ready: bool = False) -> bool:
        """Print cues on a monotonic clock; return whether this frame belongs in the episode."""
        if self.finished:
            return False
        if self.ready_at is None:
            if not frame_ready:
                return False
            self.ready_at = now
        countdown = self.countdown_seconds if self.guided else 0
        remaining = countdown - (now - self.ready_at)
        if self.started_at is None and remaining > 0:
            display = ("countdown", math.ceil(remaining))
            if display != self._last_display:
                print(f"GET READY | recording starts in {display[1]} s", flush=True)
                self._last_display = display
            return False
        if self.started_at is None:
            if not frame_ready:
                if self._last_display != ("waiting", 0):
                    print("READY | waiting for a valid camera frame...", flush=True)
                    self._last_display = ("waiting", 0)
                return False
            self.started_at = now
        elapsed = max(0.0, now - self.started_at)
        if elapsed >= self.seconds:
            self.finished = True
            self._print(f"DONE | {self.seconds:g} s recording complete", alert=True)
            self.cue_events.append({"phase": "done", "recording_time_s": elapsed})
            return False
        phase, label, end = self._phase(elapsed)
        display = (phase, math.floor(elapsed))
        if display != self._last_display:
            changed = phase != self._last_phase
            self._print(f"[{elapsed:04.1f} / {self.seconds:g} s] {label} | {max(0.0, end - elapsed):.1f} s remaining",
                        alert=changed)
            self._last_display = display
            if changed:
                self.cue_events.append({"phase": phase, "recording_time_s": elapsed})
                self._last_phase = phase
        return frame_ready

    def _phase(self, elapsed: float) -> tuple[str, str, float]:
        if not self.guided:
            return "record", "RECORDING", self.seconds
        if elapsed < self.initial_hold_seconds:
            return "hold_start", "HOLD STILL - initial position; hands clear", self.initial_hold_seconds
        if elapsed < self.seconds - self.final_hold_seconds:
            return "move", "MOVE NOW - pick, move, place, and RELEASE", self.seconds - self.final_hold_seconds
        return "hold_end", "HOLD STILL - final position; object released, hands clear", self.seconds

    def _print(self, message: str, *, alert: bool = False) -> None:
        print(("\a" if self.beep and alert else "") + message, flush=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "guided": self.guided, "requested_seconds": self.seconds,
            "countdown_seconds": self.countdown_seconds if self.guided else 0,
            "initial_hold_seconds": self.initial_hold_seconds if self.guided else None,
            "movement_seconds": self.seconds - self.initial_hold_seconds - self.final_hold_seconds if self.guided else None,
            "final_hold_seconds": self.final_hold_seconds if self.guided else None,
            "completed": self.finished, "cue_events": self.cue_events,
            "notes": "Terminal cues are instructions, not annotations of observed object movement.",
        }
