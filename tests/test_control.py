import threading
import time

import pytest

from hoops_bot.control import StopRequested, TrainingControl


def test_wait_returns_false_when_not_paused():
    assert TrainingControl().wait_if_paused(poll=0.01) is False


def test_pause_blocks_until_resumed():
    control = TrainingControl()
    control.toggle_pause()
    assert control.paused
    threading.Timer(0.1, control.toggle_pause).start()
    start = time.monotonic()
    assert control.wait_if_paused(poll=0.01) is True
    assert time.monotonic() - start >= 0.09
    assert not control.paused


def test_stop_wakes_a_paused_wait():
    control = TrainingControl()
    control.toggle_pause()
    threading.Timer(0.05, control.request_stop).start()
    with pytest.raises(StopRequested):
        control.wait_if_paused(poll=0.01)


def test_check_raises_after_stop():
    control = TrainingControl()
    control.check()
    control.request_stop()
    with pytest.raises(StopRequested):
        control.check()
