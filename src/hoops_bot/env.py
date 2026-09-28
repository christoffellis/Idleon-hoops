"""Gymnasium environment wrapping the live game.

Each step the agent picks one of two actions: wait (0) or shoot (1). The player moves on its own,
so the whole skill is *when* to shoot. Observation is [dy, dx]: vertical gap to the hoop and
horizontal gap to where the hoop will be on arrival (see observation.py).
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .capture import ScreenGrabber
from .config import Config
from .controller import GameController
from .detection import Detection, TemplateDetector
from .metrics import ShotStats
from .observation import build_observation
from .reward import closest_approach, shot_reward
from .tracker import HoopTracker


class HoopsEnv(gym.Env):
    def __init__(self, cfg: Config | None = None, grabber=None, controller=None):
        self.cfg = cfg or Config.load()
        self.grabber = grabber or ScreenGrabber(self.cfg.region)
        self.controller = controller or GameController(self.cfg)
        self.ball_detector = TemplateDetector(self.cfg.ball_template, self.cfg.match_threshold)
        self.hoop_detector = TemplateDetector(self.cfg.hoop_template, self.cfg.match_threshold)
        self.tracker = HoopTracker(self.cfg.hoop_period, self.cfg.fit_tolerance_px)

        self.action_space = spaces.Discrete(2)  # 0 = wait, 1 = shoot
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

        self.ball: Detection | None = None
        self.hoop: Detection | None = None
        self.score = 0
        self.stats = ShotStats()
        self._waits = 0
        self._last_sample = -np.inf
        self._needs_restart = False

    # ---- sensing -------------------------------------------------------
    def _sense(self) -> None:
        frame = self.grabber.grab()
        ball = self.ball_detector.find(frame)
        hoop = self.hoop_detector.find(frame)
        self.ball = ball or self.ball
        self.hoop = hoop or self.hoop
        now = time.monotonic()
        if hoop and now - self._last_sample >= self.cfg.sample_interval:
            self.tracker.update(now, hoop.x)
            self._last_sample = now

    def _obs(self) -> np.ndarray:
        predicted_x = self.tracker.predict(time.monotonic() + self.cfg.flight_time)
        return build_observation(self.ball, self.hoop, predicted_x, self.cfg)

    def _wait_for_game(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        self.ball = self.hoop = None
        while self.ball is None or self.hoop is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Could not find the ball and hoop. Run `hoops-calibrate preview` to check "
                    "the capture region and templates."
                )
            self._sense()
            time.sleep(self.cfg.step_interval)

    # ---- gym API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._needs_restart:
            self.controller.restart()
            self._needs_restart = False
            self.score = 0
            self.tracker.reset()
        self._waits = 0
        self._wait_for_game()
        return self._obs(), {}

    def step(self, action: int):
        if int(action) == 1:
            return self._shoot()
        self._waits += 1
        time.sleep(self.cfg.step_interval)
        self._sense()
        truncated = self._waits >= self.cfg.max_waits_per_shot
        return self._obs(), -self.cfg.wait_penalty, False, truncated, {}

    def _shoot(self):
        self._waits = 0
        self.controller.shoot()
        ball_path: list[tuple[float, float]] = []
        hoop_path: list[tuple[float, float]] = []
        start = time.monotonic()
        while time.monotonic() - start < self.cfg.flight_timeout:
            self._sense()
            if self.ball and self.hoop:
                ball_path.append((self.ball.x, self.ball.y))
                hoop_path.append((self.hoop.x, self.hoop.y))
                if self.ball.x > self.hoop.x + self.hoop.w:  # ball has flown past the hoop
                    break
        miss_distance = closest_approach(ball_path, hoop_path)
        reward, hit = shot_reward(miss_distance, self.cfg)
        self.stats.record(hit)
        self.score += int(hit)
        terminated = (not hit) and self.cfg.terminate_on_miss
        self._needs_restart = terminated
        time.sleep(self.cfg.settle_time)
        self._sense()
        info = {
            "hit": hit,
            "miss_distance": miss_distance,
            "score": self.score,
            "shots": self.stats.shots,
        }
        return self._obs(), reward, terminated, False, info
