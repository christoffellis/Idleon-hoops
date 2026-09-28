"""Find the ball and the hoop back panel with OpenCV template matching (script sections 5 and 7)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: float  # centre, in region pixels
    y: float
    score: float  # match confidence, 0..1
    w: int
    h: int


class TemplateDetector:
    def __init__(self, template: str | np.ndarray, threshold: float = 0.75):
        if isinstance(template, str):
            image = cv2.imread(template, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(
                    f"Template image not found: {template}. Put your cropped image in assets/."
                )
            template = image
        self.template = template
        self.threshold = threshold
        self.h, self.w = template.shape[:2]

    def find(self, frame: np.ndarray) -> Detection | None:
        if frame.shape[0] < self.h or frame.shape[1] < self.w:
            return None
        result = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, best, _, top_left = cv2.minMaxLoc(result)
        if best < self.threshold:
            return None
        return Detection(
            x=top_left[0] + self.w / 2,
            y=top_left[1] + self.h / 2,
            score=float(best),
            w=self.w,
            h=self.h,
        )
