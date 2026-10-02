"""Turn a tracked ball path into the numbers the model needs (all in normalised units).

- ``split_at_contact``: where the ball touched the rim or backboard (its velocity changes abruptly).
- ``first_departure``: the first frame where the ball has left the player (the launch).
- ``PathFit`` and ``signed_miss``: where the clean, pre-contact arc crosses the hoop, and by how
  much it was above or below the hoop. Bounces are never modelled, only the arc before them.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Func = Callable[[np.ndarray | float], np.ndarray | float]


@dataclass
class PathFit:
    """Quadratic fit of a ball path: u(t) and v(t), polynomials in (t - t0)."""

    t0: float
    cu: np.ndarray
    cv: np.ndarray
    rms: float

    def u(self, t):
        return np.polyval(self.cu, np.asarray(t) - self.t0)

    def v(self, t):
        return np.polyval(self.cv, np.asarray(t) - self.t0)


def fit_path(t, u, v) -> PathFit:
    t, u, v = (np.asarray(a, dtype=float) for a in (t, u, v))
    t0 = float(t[0])
    cu = np.polyfit(t - t0, u, 2)
    cv = np.polyfit(t - t0, v, 2)
    rms = float(np.sqrt(np.mean((np.polyval(cu, t - t0) - u) ** 2 + (np.polyval(cv, t - t0) - v) ** 2)))
    return PathFit(t0, cu, cv, rms)


def split_at_contact(t, u, v, tol: float = 0.004, min_points: int = 6) -> tuple[int, bool]:
    """Number of leading points that form a clean arc, and whether a contact was found.

    Grows a quadratic fit one point at a time. A point that lands well off the fit's prediction
    (two in a row, so one bad detection does not count) marks the contact.
    """
    t, u, v = (np.asarray(a, dtype=float) for a in (t, u, v))
    n = len(t)
    for k in range(min_points, n - 1):
        fit = fit_path(t[:k], u[:k], v[:k])
        threshold = max(tol, 4 * fit.rms)

        def error(i: int) -> float:
            return float(np.hypot(fit.u(t[i]) - u[i], fit.v(t[i]) - v[i]))

        if error(k) > threshold and error(k + 1) > threshold:
            return max(min_points, k - 1), True  # drop the frame nearest the contact
    return n, False


def first_departure(t, u, v, player_u: Func, player_v: Func, t_press: float, tol: float = 0.015) -> int | None:
    """Index of the first frame after ``t_press`` where the ball has left the player's position."""
    t, u, v = (np.asarray(a, dtype=float) for a in (t, u, v))
    for i in range(len(t)):
        if t[i] < t_press:
            continue
        if max(abs(u[i] - float(player_u(t[i]))), abs(v[i] - float(player_v(t[i])))) > tol:
            return i
    return None


def impact_time(fit: PathFit, hoop_u: Func, t_from: float, t_to: float, steps: int = 4000) -> float | None:
    """First time in [t_from, t_to] when the ball's u equals the hoop's u (the hoop may be moving)."""
    grid = np.linspace(t_from, t_to, steps)
    f = fit.u(grid) - hoop_u(grid)
    crossings = np.where((f[:-1] <= 0) & (f[1:] > 0))[0]
    if len(crossings) == 0:
        return None
    i = int(crossings[0])
    lo, hi = float(grid[i]), float(grid[i + 1])
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(fit.u(mid)) - float(hoop_u(mid)) <= 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def signed_miss(fit: PathFit, hoop_u: Func, hoop_v: float, t_from: float, t_last: float) -> tuple[float, float] | None:
    """(signed vertical miss at the hoop, impact time). Positive means the ball passed below the hoop
    centre (v grows downward). The arc is extrapolated a little past ``t_last`` if the ball touched
    something first."""
    t_hit = impact_time(fit, hoop_u, t_from, t_last + 1.0)
    if t_hit is None:
        return None
    return float(fit.v(t_hit)) - hoop_v, t_hit
