import pytest

from humanoid_retarget.training_control import TrainingBudget


def test_default_epochs_and_timed_mode_does_not_stop_after_80_epochs():
    default = TrainingBudget(None, None, clock=lambda: 0)
    assert default.stop_reason(79) is None
    assert default.stop_reason(80) == "epoch_limit"
    timed = TrainingBudget(None, 120, clock=lambda: 0)
    assert timed.max_epochs is None
    assert timed.stop_reason(100000) is None


def test_two_hour_budget_stops_at_deadline():
    now = [100.0]
    budget = TrainingBudget(None, 120, clock=lambda: now[0])
    now[0] += 7199.99
    assert budget.stop_reason(10) is None
    now[0] += 0.01
    assert budget.time_expired
    assert budget.stop_reason(10) == "time_budget"


def test_combined_limits_stop_on_either():
    now = [0.0]
    budget = TrainingBudget(3, 1, clock=lambda: now[0])
    assert budget.stop_reason(2) is None
    assert budget.stop_reason(3) == "epoch_limit"
    now[0] = 60
    assert budget.stop_reason(0) == "time_budget"


@pytest.mark.parametrize("epochs,minutes", [(0, None), (-1, None), (None, 0), (None, -1),
                                          (None, float("nan")), (None, float("inf"))])
def test_invalid_limits_rejected(epochs, minutes):
    with pytest.raises(ValueError):
        TrainingBudget(epochs, minutes)
