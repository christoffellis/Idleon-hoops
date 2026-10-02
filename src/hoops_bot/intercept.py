"""Decide when to press the key.

Given the arc, the player's motion and the hoop's motion (all deterministic functions of time),
the signed miss for a press at time t is computed exactly:

    release   t_r  = t + latency, from the player's position at t_r
    flight    tau  solves  u_ball(tau) = hoop_u(t_r + tau)   (fixed-point iteration: the ball is far
                                                              faster than the hoop, so it converges fast)
    miss(t)   = v_ball(tau) - hoop_v

The best press times are where miss(t) crosses the target (the centre of the clean-hit window).
Because the motions repeat, such times recur, so we can choose the best one in the next few seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arc_model import ArcModel


@dataclass(frozen=True)
class Release:
    t_press: float
    miss: float       # predicted signed miss at the press time (equals the target at a root)
    slope: float      # d miss / d t_press, in height units per second
    width: float      # seconds of timing slack before leaving the clean band
    t_impact: float


def predicted_miss(arc: ArcModel, t_press, player_u, player_v, hoop_u, hoop_v: float, iterations: int = 12):
    """Vectorised over ``t_press``. Returns (signed miss, impact time), NaN where no valid flight exists."""
    t_press = np.asarray(t_press, dtype=float)
    t_r = t_press + arc.latency
    pu, pv = np.asarray(player_u(t_r)), np.asarray(player_v(t_r))
    u0 = pu + arc.d0
    vx = arc.vx if abs(arc.vx) > 1e-6 else 1e-6
    tau = (np.asarray(hoop_u(t_r)) - u0) / vx
    for _ in range(iterations):
        tau = (np.asarray(hoop_u(t_r + np.clip(tau, 0, 5))) - u0) / vx
    valid = (tau > 0.02) & (tau < 3.0)
    miss = arc.ball_v(tau, pv) - hoop_v
    return np.where(valid, miss, np.nan), np.where(valid, t_r + tau, np.nan)


def find_releases(
    arc: ArcModel, target: float, band_width: float, player_u, player_v, hoop_u, hoop_v: float,
    t_from: float, t_to: float, step: float = 0.002,
) -> list[Release]:
    """Every press time in [t_from, t_to] where the predicted miss equals ``target``."""
    grid = np.arange(t_from, t_to, step)
    miss, impact = predicted_miss(arc, grid, player_u, player_v, hoop_u, hoop_v)
    f = miss - target
    releases = []
    for i in np.where(np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (f[:-1] * f[1:] <= 0) & (f[:-1] != f[1:]))[0]:
        frac = f[i] / (f[i] - f[i + 1])
        slope = float((miss[i + 1] - miss[i]) / step)
        width = band_width / abs(slope) if abs(slope) > 1e-9 else float("inf")
        releases.append(Release(
            t_press=float(grid[i] + frac * step), miss=float(target), slope=slope, width=float(width),
            t_impact=float(impact[i] + frac * (impact[i + 1] - impact[i])),
        ))
    return releases


def choose_release(releases: list[Release], earliest: float, patience: float = 0.7) -> Release | None:
    """The earliest press that is nearly as forgiving as the best one available.

    Waiting costs time but a wider timing slack costs lives, so prefer the first release whose slack is
    at least ``patience`` times the widest."""
    options = [r for r in releases if r.t_press >= earliest]
    if not options:
        return None
    widest = max(r.width for r in options)
    return min((r for r in options if r.width >= patience * widest), key=lambda r: r.t_press)
