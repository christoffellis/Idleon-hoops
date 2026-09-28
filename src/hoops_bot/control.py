"""Manual pause / resume / stop for a long training session.

Between games the bot waits on its own (see env.py), so the controls are for when *you* need a
break: F8 toggles pause, F9 saves and stops. Ctrl+C in the terminal also saves and stops.
Global hotkeys use pynput, which works while the game window has focus.
"""
from __future__ import annotations

import threading
import time


class StopRequested(Exception):
    """Raised inside the environment when the user asks to stop; the trainer saves and exits."""


class TrainingControl:
    def __init__(self, pause_key: str = "<f8>", stop_key: str = "<f9>"):
        self.pause_key = pause_key
        self.stop_key = stop_key
        self._running = threading.Event()
        self._running.set()
        self._stop = threading.Event()
        self._listener = None

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def toggle_pause(self) -> None:
        if self._running.is_set():
            self._running.clear()
            print(f"[paused] Training holds after the current shot. Press {self.pause_key} to resume.")
        else:
            self._running.set()
            print("[resumed]")

    def request_stop(self) -> None:
        self._stop.set()
        self._running.set()  # wake anything waiting in a pause
        print("[stopping] Saving the model...")

    def check(self) -> None:
        if self._stop.is_set():
            raise StopRequested

    def wait_if_paused(self, poll: float = 0.1) -> bool:
        """Block while paused. Returns True if it had to wait, so the caller can refresh its state."""
        waited = False
        while not self._running.is_set():
            waited = True
            time.sleep(poll)
        self.check()
        return waited

    def start_hotkeys(self) -> None:
        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys(
                {self.pause_key: self.toggle_pause, self.stop_key: self.request_stop}
            )
            self._listener.start()
        except Exception as exc:  # no display, missing permissions, pynput not installed
            print(f"Global hotkeys unavailable ({exc}). Ctrl+C in this terminal still saves and stops.")

    def stop_hotkeys(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
