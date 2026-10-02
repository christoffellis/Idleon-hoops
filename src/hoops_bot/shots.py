"""The shot log (runs/shots.jsonl): one JSON object per shot, appended as the game is played.

Everything needed to refit offline is stored: the ball path, and the player and hoop motion models
at the moment of the press. All positions are normalised (u, v) and times are in seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .arc_model import usable_shot
from .hoop_model import SineModel
from .trajectory import fit_path, signed_miss, split_at_contact


def append_shot(path: str | Path, shot: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as handle:
        handle.write(json.dumps(shot) + "\n")


def load_shots(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def observed_hoop_u(shot: dict, t_end: float):
    """The hoop's u(t) as seen during the flight (the hoop is on screen the whole time, so this beats any
    extrapolated model); the fitted model fills in outside the observed stretch."""
    model = SineModel.from_dict(shot["hoop"]["u"])
    trace = np.asarray([p for p in shot["hoop"].get("trace", []) if p[0] <= t_end + 0.05], dtype=float)
    if len(trace) < 8:
        return model
    t, u = trace[:, 0], trace[:, 1]

    def hoop_u(x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= t[0]) & (x <= t[-1]), np.interp(x, t, u), model(x))

    return hoop_u


def analyse_shot(shot: dict) -> dict:
    """Add ``contact``, ``signed_miss`` and ``t_impact`` to a shot, computed from its own path.

    Positive signed miss means the arc passed below the hoop's centre (v grows downward). The arc is
    cut at the first contact with the rim or backboard, so bounces never affect it.
    """
    cut, reason = usable_shot(shot)
    out = dict(shot, contact=False, signed_miss=None, t_impact=None, analysis_note=reason)
    if cut is None:
        return out
    path = np.asarray(shot["path"], dtype=float)
    after_launch = path[path[:, 0] >= cut.t[0]]
    out["contact"] = bool(len(cut.t) < len(after_launch))
    hoop_u = observed_hoop_u(shot, cut.t[-1])
    result = signed_miss(fit_path(cut.t, cut.u, cut.v), hoop_u, float(shot["hoop"]["v"]), cut.t[0], cut.t[-1])
    if result is not None:
        out["signed_miss"], out["t_impact"] = result
    return out
