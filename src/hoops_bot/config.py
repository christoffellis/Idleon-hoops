"""Central configuration. Values can be overridden by a config.json in the repo root."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_PATH = Path("config.json")


@dataclass
class Config:
    # --- Screen capture -------------------------------------------------
    # (left, top, width, height) of the Idleon game area. Set with `hoops-calibrate region`.
    region: tuple[int, int, int, int] = (0, 0, 1280, 720)

    # --- Detection (script section 7) -----------------------------------
    ball_template: str = "assets/ball.png"
    hoop_template: str = "assets/hoop_panel.png"
    match_threshold: float = 0.75

    # --- Hoop motion model (script section 8) ---------------------------
    hoop_period: float = 4.0        # seconds per horizontal cycle
    sample_interval: float = 0.25   # 4 Hz sampling = 16x the 0.25 Hz hoop frequency
    flight_time: float = 0.5        # seconds from shot to arrival; measure and tune this
    fit_tolerance_px: float = 12.0  # RMS error above which the sine fit is distrusted

    # --- Observation scaling (script section 5) -------------------------
    norm_x: float = 400.0
    norm_y: float = 300.0

    # --- Environment / reward (script section 6) ------------------------
    step_interval: float = 0.05     # agent decision rate (much faster than the hoop sample rate)
    max_waits_per_shot: int = 60    # truncate the episode if the agent never shoots
    flight_timeout: float = 3.0
    settle_time: float = 0.5        # pause after a shot so the game can register the result
    hit_tolerance_px: float = 25.0  # closest approach (ball centre to hoop centre) that counts as a score
    hit_reward: float = 1.0
    miss_penalty: float = 1.0       # scaled by how far the miss was
    miss_scale_px: float = 300.0    # miss distance at which the penalty saturates
    wait_penalty: float = 0.01      # stops the agent from stalling forever
    terminate_on_miss: bool = True

    # --- Input ----------------------------------------------------------
    shoot_key: str = "space"
    restart_click: tuple[int, int] | None = None  # screen (x, y) of the retry button, if any

    # --- Runs -----------------------------------------------------------
    target_score: int = 40          # score needed for the trophy
    countdown: int = 5              # seconds to focus the game window before starting

    @classmethod
    def load(cls, path: str | Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        p = Path(path)
        if p.exists():
            known = {f.name for f in fields(cls)}
            for key, value in json.loads(p.read_text()).items():
                if key not in known:
                    continue
                if key in ("region", "restart_click") and value is not None:
                    value = tuple(value)
                setattr(cfg, key, value)
        return cfg

    def save(self, path: str | Path = CONFIG_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
