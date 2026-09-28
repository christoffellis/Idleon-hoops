"""Turn detections into the AI's input (script section 5).

Four raw numbers (player x/y, hoop x/y) become two relative ones: the vertical gap to the hoop
and the horizontal gap to where the hoop *will be* when the ball arrives.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .detection import Detection


def build_observation(
    player: Detection, hoop: Detection, predicted_hoop_x: float, cfg: Config
) -> np.ndarray:
    dy = (hoop.y - player.y) / cfg.norm_y
    dx = (predicted_hoop_x - player.x) / cfg.norm_x
    return np.array([dy, dx], dtype=np.float32)
