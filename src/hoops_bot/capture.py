"""Fast screen capture of the game region (script section 7)."""
from __future__ import annotations

import cv2
import mss
import numpy as np


class ScreenGrabber:
    def __init__(self, region: tuple[int, int, int, int]):
        left, top, width, height = region
        self._monitor = {"left": left, "top": top, "width": width, "height": height}
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        """Return the region as a BGR image."""
        shot = np.asarray(self._sct.grab(self._monitor))
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
