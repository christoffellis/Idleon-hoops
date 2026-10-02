"""Fast screen capture of the game viewport, with a timestamp taken at the moment of the grab."""
from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import mss
import numpy as np

from .geometry import Viewport


class ScreenGrabber:
    def __init__(
        self,
        monitor: int = 1,
        region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._sct = mss.mss()
        mon = self._sct.monitors[monitor]
        left, top, width, height = region
        self._box = {
            "left": int(mon["left"] + left * mon["width"]),
            "top": int(mon["top"] + top * mon["height"]),
            "width": max(1, int(width * mon["width"])),
            "height": max(1, int(height * mon["height"])),
        }
        self._clock = clock
        self.viewport = Viewport(self._box["width"], self._box["height"])

    def grab_timed(self) -> tuple[np.ndarray, float]:
        shot = np.asarray(self._sct.grab(self._box))
        stamp = self._clock()
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR), stamp

    def grab(self) -> np.ndarray:
        return self.grab_timed()[0]
