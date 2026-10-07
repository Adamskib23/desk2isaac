from __future__ import annotations

import math
import time


class TrainingBudget:
    """Bound fitting by epochs, wall-clock time, or whichever comes first."""

    def __init__(self, epochs: int | None, minutes: float | None, *, clock=time.monotonic):
        if epochs is not None and epochs < 1:
            raise ValueError("Epoch limit must be positive")
        if minutes is not None and (not math.isfinite(minutes) or minutes <= 0):
            raise ValueError("Time budget must be finite and positive")
        self.max_epochs = 80 if epochs is None and minutes is None else epochs
        self.seconds = None if minutes is None else minutes * 60
        self.clock = clock
        self.started = clock()

    @property
    def elapsed_s(self) -> float:
        return self.clock() - self.started

    @property
    def time_expired(self) -> bool:
        return self.seconds is not None and self.elapsed_s >= self.seconds

    def stop_reason(self, completed_epochs: int) -> str | None:
        if self.time_expired:
            return "time_budget"
        if self.max_epochs is not None and completed_epochs >= self.max_epochs:
            return "epoch_limit"
        return None
