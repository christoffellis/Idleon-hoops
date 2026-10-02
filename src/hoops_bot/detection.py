"""Find the ball, the hoop back panel and the remaining lives with scale-aware template matching.

Template matching is not scale invariant, so templates are resized from the viewport width they
were cropped at (``reference_width``) to the width of the current frame. Results are in
normalised units (u, v) and refined to sub-pixel accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    u: float       # centre, fraction of viewport width
    v: float       # centre, fraction of viewport height
    score: float   # match confidence, 0..1
    w: float       # size, fraction of viewport width
    h: float       # size, fraction of viewport height


def load_template(template: str | np.ndarray) -> np.ndarray:
    if not isinstance(template, str):
        return template
    image = cv2.imread(template, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Template image not found: {template}. Put your cropped image in assets/.")
    return image


def resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 0.01:
        return template
    h, w = template.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(template, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


class ScaledTemplate:
    """A template cropped at ``reference_width``, resized on demand to suit each frame width."""

    def __init__(self, template: str | np.ndarray, reference_width: float | None = None):
        self.original = load_template(template)
        self.reference_width = reference_width
        self._cache: dict[int, np.ndarray] = {}

    def for_frame(self, frame: np.ndarray) -> np.ndarray:
        width = frame.shape[1]
        if width not in self._cache:
            scale = width / self.reference_width if self.reference_width else 1.0
            self._cache[width] = resize_template(self.original, scale)
        return self._cache[width]


def _peak_offset(left: float, centre: float, right: float) -> float:
    """Sub-pixel offset of a peak from three neighbouring match scores (parabola fit)."""
    denom = left - 2 * centre + right
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


class TemplateDetector:
    def __init__(self, template: str | np.ndarray, threshold: float = 0.75, reference_width: float | None = None):
        self.template = ScaledTemplate(template, reference_width)
        self.threshold = threshold

    def find(self, frame: np.ndarray) -> Detection | None:
        tpl = self.template.for_frame(frame)
        h, w = tpl.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        if frame_h < h or frame_w < w:
            return None
        result = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
        _, best, _, (x, y) = cv2.minMaxLoc(result)
        if not np.isfinite(best) or best < self.threshold:
            return None
        dx = dy = 0.0
        if 0 < x < result.shape[1] - 1:
            dx = _peak_offset(result[y, x - 1], result[y, x], result[y, x + 1])
        if 0 < y < result.shape[0] - 1:
            dy = _peak_offset(result[y - 1, x], result[y, x], result[y + 1, x])
        return Detection((x + dx + w / 2) / frame_w, (y + dy + h / 2) / frame_h, float(best), w / frame_w, h / frame_h)


def crop_fraction(frame: np.ndarray, region: tuple[float, float, float, float] | None) -> np.ndarray:
    """Crop a (left, top, width, height) region given as fractions of the frame."""
    if region is None:
        return frame
    height, width = frame.shape[:2]
    left, top, w, h = region
    x0, y0 = int(left * width), int(top * height)
    return frame[y0 : y0 + max(1, int(h * height)), x0 : x0 + max(1, int(w * width))]


class LivesCounter:
    """Counts the life icons on screen. A lost life makes the icon disappear (or change), so the
    count drops."""

    def __init__(
        self,
        template: str | np.ndarray,
        threshold: float = 0.8,
        region: tuple[float, float, float, float] | None = None,
        reference_width: float | None = None,
    ):
        self.template = ScaledTemplate(template, reference_width)
        self.threshold = threshold
        self.region = region

    def count(self, frame: np.ndarray) -> int:
        tpl = self.template.for_frame(frame)  # scale from the full frame, before cropping
        crop = crop_fraction(frame, self.region)
        h, w = tpl.shape[:2]
        if crop.shape[0] < h or crop.shape[1] < w:
            return 0
        result = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= self.threshold)
        order = np.argsort(-result[ys, xs])[:500]
        min_dist = min(w, h) / 2
        kept: list[tuple[int, int]] = []
        for i in order:
            point = (int(xs[i]), int(ys[i]))
            if all(math.dist(point, other) >= min_dist for other in kept):
                kept.append(point)
        return len(kept)


def find_best_scale(frame: np.ndarray, template: np.ndarray, low: float = 0.3, high: float = 3.0) -> tuple[float, float]:
    """Search for the template scale that matches this frame best. Returns (scale, match score).

    With the viewport width W, the width the template was cropped at is W / scale.
    """
    def score_at(scale: float) -> float:
        tpl = resize_template(template, scale)
        if frame.shape[0] < tpl.shape[0] or frame.shape[1] < tpl.shape[1]:
            return -1.0
        _, best, _, _ = cv2.minMaxLoc(cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED))
        return float(best) if np.isfinite(best) else -1.0

    coarse = np.geomspace(low, high, 40)
    best_scale = max(coarse, key=score_at)
    fine = np.geomspace(best_scale / 1.08, best_scale * 1.08, 25)
    best_scale = max(fine, key=score_at)
    return float(best_scale), score_at(best_scale)
