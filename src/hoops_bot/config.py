"""Central configuration. Values can be overridden by a config.json in the repo root.

Every spatial setting is a fraction of the monitor or of the game viewport, never a pixel count.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_PATH = Path("config.json")
FRACTION_FIELDS = ("region", "lives_region", "score_region")


@dataclass
class Config:
    # --- Screen capture: fractions (left, top, width, height) -------------
    monitor: int = 1  # mss monitor index (1 = primary)
    region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # of the monitor: the game viewport
    lives_region: tuple[float, float, float, float] | None = None     # of the viewport
    score_region: tuple[float, float, float, float] | None = None     # of the viewport

    # --- Detection --------------------------------------------------------
    ball_template: str = "assets/ball.png"
    hoop_template: str = "assets/hoop_panel.png"
    life_template: str = "assets/life.png"
    digits_dir: str = "assets/digits"          # 0.png ... 9.png, made by `hoops-calibrate digits`
    reference_file: str = "assets/reference.json"  # viewport width the crops were taken at
    match_threshold: float = 0.75
    lives_threshold: float = 0.8
    score_threshold: float = 0.7
    max_lives: int = 3

    # --- Motion (periods in seconds; None = measure it) -------------------
    hoop_period: float = 4.0
    player_v_period: float | None = None
    player_u_period: float | None = None
    track_tolerance: float = 0.015   # normalised error that counts as "the hoop respawned"

    # --- Shots ------------------------------------------------------------
    shoot_key: str = "space"
    settle_time: float = 0.8         # wait after a shot for score and lives to update
    flight_timeout: float = 2.5
    max_wait: float = 20.0           # give up looking for a good release after this long
    probe_step: float = 0.02         # aim offset (fraction of height) between probe shots
    default_window: float = 0.02     # clean-hit band width assumed until measured

    # --- Runs and controls --------------------------------------------------
    trophy_score: int = 40           # reported when reached; play continues unless --stop-at is given
    countdown: int = 5
    pause_hotkey: str = "<f8>"
    stop_hotkey: str = "<f9>"
    shots_file: str = "runs/shots.jsonl"
    physics_file: str = "models/physics.json"

    # --- Used only by the video illustrations (animations/) ---------------
    sample_interval: float = 0.25
    flight_time: float = 0.5
    fit_tolerance_px: float = 12.0
    miss_penalty: float = 1.0
    miss_scale_px: float = 300.0
    hit_reward: float = 1.0

    @classmethod
    def load(cls, path: str | Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        p = Path(path)
        if not p.exists():
            return cfg
        known = {f.name for f in fields(cls)}
        for key, value in json.loads(p.read_text()).items():
            if key not in known:
                continue
            if key in FRACTION_FIELDS and value is not None:
                value = tuple(value)
                if any(part > 1.5 for part in value):
                    print(f"config.json: ignoring {key}={value}, it looks like pixels. Re-run `hoops-calibrate`.")
                    continue
            setattr(cfg, key, value)
        return cfg

    def save(self, path: str | Path = CONFIG_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def reference_width(self) -> float | None:
        """Viewport width the template crops were taken at (see `hoops-calibrate scale`)."""
        p = Path(self.reference_file)
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("reference_width")
