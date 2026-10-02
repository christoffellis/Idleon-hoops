"""Resolution independence: everything is stored in normalised units.

u = x / width and v = y / height of the game viewport, so data and models stay valid when the
screen size changes. Time is always in seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Viewport:
    width: int
    height: int

    @classmethod
    def from_frame(cls, frame: np.ndarray) -> "Viewport":
        return cls(width=int(frame.shape[1]), height=int(frame.shape[0]))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def to_norm(self, x: float, y: float) -> tuple[float, float]:
        return x / self.width, y / self.height

    def to_px(self, u: float, v: float) -> tuple[float, float]:
        return u * self.width, v * self.height

    def dist(self, du: float, dv: float) -> float:
        """Distance in height units. Horizontal differences are scaled by the aspect ratio so
        that a circle stays a circle."""
        return math.hypot(du * self.aspect, dv)

    def template_scale(self, reference_width: float | None) -> float:
        """How much to resize templates cropped at ``reference_width`` for this viewport."""
        if not reference_width:
            return 1.0
        return self.width / reference_width


def same_aspect(a: float, b: float, tolerance: float = 0.01) -> bool:
    return abs(a - b) <= tolerance * max(a, b)
