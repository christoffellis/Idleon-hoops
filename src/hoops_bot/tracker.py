"""Predict where the hoop will be (script section 8).

The hoop completes one horizontal cycle every ``period`` seconds, so its frequency is known
(0.25 Hz). Sampling at 4 Hz is 16x that, far above the Nyquist rate (2x), so a handful of
samples pins down the whole motion. With the frequency fixed, fitting

    x(t) = a + b*sin(wt) + c*cos(wt)

is a plain linear least-squares problem, so it is fast and needs no iteration.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class HoopTracker:
    def __init__(self, period: float = 4.0, tolerance_px: float = 12.0, min_samples: int = 6):
        self.omega = 2 * np.pi / period
        self.window = period  # keep one full cycle of samples
        self.tolerance = tolerance_px
        self.min_samples = min_samples
        self._samples: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, t: float, x: float) -> None:
        self._samples.append((t, x))
        while self._samples and t - self._samples[0][0] > self.window:
            self._samples.popleft()

    @property
    def last_x(self) -> float | None:
        return self._samples[-1][1] if self._samples else None

    def _fit(self) -> tuple[np.ndarray, float] | None:
        if len(self._samples) < self.min_samples:
            return None
        t, x = np.array(self._samples).T
        design = np.column_stack([np.ones_like(t), np.sin(self.omega * t), np.cos(self.omega * t)])
        coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
        rmse = float(np.sqrt(np.mean((design @ coeffs - x) ** 2)))
        return coeffs, rmse

    def predict(self, t: float) -> float:
        """Hoop x at time ``t``. Falls back to the last seen x when the fit is not trustworthy
        (too few samples, or the hoop just respawned and the window mixes two positions)."""
        fit = self._fit()
        if fit is None or fit[1] > self.tolerance:
            if self.last_x is None:
                raise RuntimeError("HoopTracker has no samples yet")
            return self.last_x
        (a, b, c), _ = fit
        return float(a + b * np.sin(self.omega * t) + c * np.cos(self.omega * t))
