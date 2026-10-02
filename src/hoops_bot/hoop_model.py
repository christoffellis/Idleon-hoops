"""Sine models for the hoop and the player (both move as a sine).

``SineModel``: x(t) = c + a*sin(wt) + b*cos(wt).
With the period known this is a linear least-squares fit, so three samples fix it.
With the period unknown it is found by scanning candidate periods and keeping the one with the
smallest residual. That is more precise than an FFT on a short, unevenly timed recording, and it
needs no uniform sampling.

``SineTrack`` follows one coordinate live, notices when the hoop respawns (the sine jumps), and
re-locks.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SineModel:
    c: float
    a: float = 0.0
    b: float = 0.0
    omega: float = 1.0

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return self.c + self.a * np.sin(self.omega * t) + self.b * np.cos(self.omega * t)

    @property
    def amplitude(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def period(self) -> float:
        return 2 * math.pi / self.omega

    def to_dict(self) -> dict:
        return {"c": self.c, "a": self.a, "b": self.b, "omega": self.omega}

    @classmethod
    def from_dict(cls, data: dict) -> "SineModel":
        return cls(float(data["c"]), float(data["a"]), float(data["b"]), float(data["omega"]))

    @classmethod
    def constant(cls, c: float) -> "SineModel":
        return cls(float(c), 0.0, 0.0, 1.0)


def fit_sine(t, x, omega: float) -> tuple[SineModel, float]:
    """Least-squares fit with a known angular frequency. Returns (model, rms error)."""
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    t0 = t.mean()  # centre the time axis so the fit stays well conditioned
    design = np.column_stack([np.ones_like(t), np.sin(omega * (t - t0)), np.cos(omega * (t - t0))])
    (c, s, k), *_ = np.linalg.lstsq(design, x, rcond=None)
    # re-express in absolute time: sin(w(t-t0)) = sin(wt)cos(wt0) - cos(wt)sin(wt0), etc.
    a = s * math.cos(omega * t0) + k * math.sin(omega * t0)
    b = -s * math.sin(omega * t0) + k * math.cos(omega * t0)
    rms = float(np.sqrt(np.mean((design @ np.array([c, s, k]) - x) ** 2)))
    return SineModel(float(c), float(a), float(b), omega), rms


def scan_period(t, x, low: float = 0.8, high: float = 12.0, steps: int = 1200) -> tuple[float, float]:
    """Find the period with the smallest fit error. Returns (period, rms error).

    Only periods that fit at least 1.2 times into the recording are considered.
    """
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    span = float(t[-1] - t[0])
    high = min(high, span / 1.2)
    if high <= low:
        raise ValueError(f"Need a longer recording to find a period: have {span:.1f} s, need about {1.2 * low:.1f} s or more.")
    periods = np.geomspace(low, high, steps)
    errors = [fit_sine(t, x, 2 * math.pi / p)[1] for p in periods]
    best = int(np.argmin(errors))
    fine = np.linspace(periods[max(0, best - 1)], periods[min(steps - 1, best + 1)], 200)
    fine_errors = [fit_sine(t, x, 2 * math.pi / p)[1] for p in fine]
    i = int(np.argmin(fine_errors))
    return float(fine[i]), float(fine_errors[i])


class SineTrack:
    """Live tracker for one coordinate that moves as a sine (or does not move at all)."""

    def __init__(
        self,
        period: float | None = None,
        tolerance: float = 0.015,
        min_span_fraction: float = 0.25,
        min_samples: int = 6,
        min_amplitude: float = 0.004,
        period_range: tuple[float, float] = (0.8, 12.0),
        unknown_period_span: float = 6.0,
    ):
        self.period = period
        self.tolerance = tolerance
        self.min_span_fraction = min_span_fraction
        self.min_samples = min_samples
        self.min_amplitude = min_amplitude
        self.period_range = period_range
        self.unknown_period_span = unknown_period_span
        self.model: SineModel | None = None
        self.rms = math.inf
        self.resets = 0
        self._samples: deque[tuple[float, float]] = deque()
        self._recent_bad: list[tuple[float, float]] = []

    # ---- feeding -----------------------------------------------------------
    def reset(self, keep: list[tuple[float, float]] | None = None) -> None:
        self._samples.clear()
        self.model, self.rms = None, math.inf
        self._recent_bad = []
        for sample in keep or []:
            self._samples.append(sample)

    def update(self, t: float, x: float) -> bool:
        """Add a sample. Returns True if the signal jumped (e.g. the hoop respawned) and the track restarted."""
        respawned = False
        if self.ready:
            if abs(x - float(self.model(t))) > self.tolerance:
                self._recent_bad.append((t, x))
                if len(self._recent_bad) >= 3:
                    self.reset(keep=self._recent_bad)
                    self.resets += 1
                    respawned = True
                    self._recent_bad = []
                    self._refit()
                    return respawned
                return respawned  # ignore a lone outlier
            self._recent_bad = []
        self._samples.append((t, x))
        horizon = 2 * (self.period or 6.0)
        while self._samples and t - self._samples[0][0] > horizon:
            self._samples.popleft()
        self._refit()
        return respawned

    # ---- fitting -----------------------------------------------------------
    def _refit(self) -> None:
        if len(self._samples) < self.min_samples:
            return
        t, x = (np.array(col) for col in zip(*self._samples))
        span = float(t[-1] - t[0])
        if float(np.ptp(x)) < self.min_amplitude:  # not moving: a constant is the model
            self.model, self.rms = SineModel.constant(float(np.median(x))), float(np.std(x))
            return
        if self.period is None:
            if span < self.unknown_period_span:
                return
            try:
                self.period, _ = scan_period(t, x, *self.period_range)
            except ValueError:
                return
        if span < self.min_span_fraction * self.period:
            return
        self.model, self.rms = fit_sine(t, x, 2 * math.pi / self.period)

    # ---- reading -----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.model is not None and self.rms <= self.tolerance

    @property
    def last(self) -> float | None:
        return self._samples[-1][1] if self._samples else None

    def predict(self, t):
        if self.model is not None:
            return self.model(t)
        if self.last is None:
            raise RuntimeError("SineTrack has no samples yet")
        return self.last + 0 * np.asarray(t, dtype=float)
