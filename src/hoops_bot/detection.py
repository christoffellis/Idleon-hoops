"""Find the ball, the hoop back panel and the remaining lives with OpenCV template matching
(script sections 5 and 7)."""
from __future__ import annotations

import math
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


def _load_template(template: str | np.ndarray) -> np.ndarray:
    if not isinstance(template, str):
        return template
    image = cv2.imread(template, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Template image not found: {template}. Put your cropped image in assets/."
        )
    return image


class TemplateDetector:
    def __init__(self, template: str | np.ndarray, threshold: float = 0.75):
        self.template = _load_template(template)
        self.threshold = threshold
        self.h, self.w = self.template.shape[:2]

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


class LivesCounter:
    """Counts how many copies of the life icon are visible. A lost life means the icon is gone
    (or replaced by an empty slot that does not match the template), so the count drops."""

    def __init__(
        self,
        template: str | np.ndarray,
        threshold: float = 0.8,
        region: tuple[int, int, int, int] | None = None,
    ):
        self.template = _load_template(template)
        self.threshold = threshold
        self.region = region
        self.h, self.w = self.template.shape[:2]

    def count(self, frame: np.ndarray) -> int:
        if self.region is not None:
            x, y, w, h = self.region
            frame = frame[y : y + h, x : x + w]
        if frame.shape[0] < self.h or frame.shape[1] < self.w:
            return 0
        result = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= self.threshold)
        order = np.argsort(-result[ys, xs])[:500]  # strongest matches first
        min_dist = min(self.w, self.h) / 2  # matches closer than this are the same icon
        kept: list[tuple[int, int]] = []
        for i in order:
            point = (int(xs[i]), int(ys[i]))
            if all(math.dist(point, other) >= min_dist for other in kept):
                kept.append(point)
        return len(kept)
