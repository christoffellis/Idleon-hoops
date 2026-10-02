"""The ball's flight, identified from a few recorded shots.

After a key press at ``t_press`` the game registers it ``latency`` seconds later (t_r). The ball then
leaves the player's position and flies a fixed arc, independent of the player's momentum:

    u(t) = pu(t_r) + d0 + vx * tau
    v(t) = pv(t_r) + e0 + vy * tau + 0.5 * g * tau^2,        tau = t - t_r

pu and pv are the player's position (a sine, tracked live). The six numbers are shared by every
shot, so they are fitted together: for each candidate latency the rest is a linear least-squares
problem, and the latency with the smallest total error wins. Latency can only be told apart when
the shots were taken at different phases of the player's motion, so the fit reports when it cannot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .hoop_model import SineModel
from .trajectory import first_departure, split_at_contact


@dataclass
class ArcModel:
    latency: float
    d0: float
    vx: float
    e0: float
    vy: float
    g: float
    rms: float = 0.0
    n_shots: int = 0
    n_points: int = 0
    latency_uncertain: bool = True
    latency_low: float = 0.0
    latency_high: float = 0.0

    def ball_u(self, tau, pu):
        return pu + self.d0 + self.vx * tau

    def ball_v(self, tau, pv):
        return pv + self.e0 + self.vy * tau + 0.5 * self.g * tau**2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ArcModel":
        return cls(**data)


@dataclass
class UsableShot:
    t_press: float
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pu: SineModel
    pv: SineModel


def usable_shot(shot: dict, launch_tol: float = 0.015, contact_tol: float = 0.004) -> tuple[UsableShot | None, str]:
    """Cut a logged shot down to its clean, post-launch, pre-contact part. Returns (shot, reason)."""
    path = np.asarray(shot.get("path") or [], dtype=float)
    if len(path) < 8:
        return None, "path too short"
    pu = SineModel.from_dict(shot["player"]["u"])
    pv = SineModel.from_dict(shot["player"]["v"])
    t, u, v = path[:, 0], path[:, 1], path[:, 2]
    start = first_departure(t, u, v, pu, pv, shot["t_press"], launch_tol)
    if start is None:
        return None, "ball never left the player"
    t, u, v = t[start:], u[start:], v[start:]
    n_clean, _ = split_at_contact(t, u, v, contact_tol)
    if n_clean < 6:
        return None, "too few clean frames before contact"
    return UsableShot(float(shot["t_press"]), t[:n_clean], u[:n_clean], v[:n_clean], pu, pv), ""


def _solve(shots: list[UsableShot], latency: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Linear fit of (d0, vx) and (e0, vy, g) for one candidate latency. Returns (u params, v params, SSE)."""
    rows_u, rhs_u, rows_v, rhs_v = [], [], [], []
    for s in shots:
        t_r = s.t_press + latency
        tau = s.t - t_r
        rows_u.append(np.column_stack([np.ones_like(tau), tau]))
        rhs_u.append(s.u - float(s.pu(t_r)))
        rows_v.append(np.column_stack([np.ones_like(tau), tau, 0.5 * tau**2]))
        rhs_v.append(s.v - float(s.pv(t_r)))
    au, bu = np.vstack(rows_u), np.concatenate(rhs_u)
    av, bv = np.vstack(rows_v), np.concatenate(rhs_v)
    pu, *_ = np.linalg.lstsq(au, bu, rcond=None)
    pv, *_ = np.linalg.lstsq(av, bv, rcond=None)
    sse = float(np.sum((au @ pu - bu) ** 2) + np.sum((av @ pv - bv) ** 2))
    return pu, pv, sse


def fit_arc(shots: list[dict], latency_range: tuple[float, float] = (0.0, 0.35), steps: int = 176) -> tuple[ArcModel | None, list[str]]:
    """Fit the shared arc from logged shots. Returns (model or None, notes about skipped shots)."""
    notes: list[str] = []
    usable: list[UsableShot] = []
    for i, shot in enumerate(shots):
        cut, reason = usable_shot(shot)
        if cut is None:
            notes.append(f"shot {shot.get('id', i)} skipped: {reason}")
        else:
            usable.append(cut)
    if not usable:
        return None, notes

    grid = np.linspace(*latency_range, steps)
    sse = np.array([_solve(usable, L)[2] for L in grid])
    best = int(np.argmin(sse))
    fine = np.linspace(grid[max(0, best - 1)], grid[min(steps - 1, best + 1)], 60)
    fine_sse = np.array([_solve(usable, L)[2] for L in fine])
    latency = float(fine[int(np.argmin(fine_sse))])
    pu, pv, total = _solve(usable, latency)
    n_points = int(sum(len(s.t) for s in usable))

    slack = n_points * (1e-3) ** 2  # latencies this close to the best are indistinguishable
    plausible = grid[sse <= sse.min() + slack]
    low, high = float(plausible.min()), float(plausible.max())
    uncertain = len(usable) < 2 or (high - low) > 0.02
    if uncertain:
        notes.append(
            f"latency is only known to within {low * 1000:.0f}-{high * 1000:.0f} ms: "
            "record more shots released at different phases of the player's motion"
        )
    model = ArcModel(
        latency=latency, d0=float(pu[0]), vx=float(pu[1]), e0=float(pv[0]), vy=float(pv[1]), g=float(pv[2]),
        rms=float(np.sqrt(total / max(1, n_points))), n_shots=len(usable), n_points=n_points,
        latency_uncertain=uncertain, latency_low=low, latency_high=high,
    )
    return model, notes
