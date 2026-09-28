import numpy as np
import pytest

from hoops_bot.tracker import HoopTracker

PERIOD = 4.0
OMEGA = 2 * np.pi / PERIOD


def hoop_x(t, centre=600.0, amp=150.0, phase=0.7):
    return centre + amp * np.sin(OMEGA * t + phase)


def feed(tracker, t_end, dt=0.25):
    for t in np.arange(0, t_end, dt):
        tracker.update(t, hoop_x(t))


def test_predicts_future_position_of_moving_hoop():
    tracker = HoopTracker(period=PERIOD)
    feed(tracker, 5.0)
    for horizon in (0.3, 0.5, 1.0):
        t = 5.0 + horizon
        assert tracker.predict(t) == pytest.approx(hoop_x(t), abs=1.0)


def test_static_hoop_predicts_its_position():
    tracker = HoopTracker(period=PERIOD)
    for t in np.arange(0, 3, 0.25):
        tracker.update(t, 812.0)
    assert tracker.predict(3.5) == pytest.approx(812.0, abs=0.5)


def test_falls_back_to_last_x_before_enough_samples():
    tracker = HoopTracker(period=PERIOD)
    tracker.update(0.0, 500.0)
    tracker.update(0.25, 510.0)
    assert tracker.predict(1.0) == 510.0


def test_falls_back_to_last_x_after_respawn():
    tracker = HoopTracker(period=PERIOD, tolerance_px=12.0)
    for t in np.arange(0, 2, 0.25):
        tracker.update(t, 300.0)
    for t in np.arange(2, 3, 0.25):  # hoop jumps to a new spot: window now mixes two positions
        tracker.update(t, 900.0)
    assert tracker.predict(3.5) == 900.0


def test_no_samples_raises():
    with pytest.raises(RuntimeError):
        HoopTracker().predict(1.0)
