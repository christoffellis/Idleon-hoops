"""Reward shaping (script section 6): reward a score, punish a miss harder the further it misses.

Whether a shot hit is decided by the lives display (see env.py). The closest approach of the ball
to the hoop is only used to scale the penalty for a miss.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from .config import Config

Point = tuple[float, float]


def closest_approach(ball_path: Sequence[Point], hoop_path: Sequence[Point]) -> float:
    """Smallest ball-to-hoop distance over a shot. Paths are sampled frame by frame, so each
    ball point is compared with the hoop position from the same frame."""
    if not ball_path or len(ball_path) != len(hoop_path):
        return math.inf
    return min(math.dist(b, h) for b, h in zip(ball_path, hoop_path))


def shot_reward(hit: bool, miss_distance: float, cfg: Config) -> float:
    if hit:
        return cfg.hit_reward
    scaled = min(miss_distance / cfg.miss_scale_px, 1.0)
    return -cfg.miss_penalty * scaled
