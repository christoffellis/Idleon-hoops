"""Gymnasium environment wrapping the live game.

Each step the agent picks one of two actions: wait (0) or shoot (1). The player moves on its own,
so the whole skill is *when* to shoot. Observation is [dy, dx]: vertical gap to the hoop and
horizontal gap to where the hoop will be on arrival (see observation.py).

One episode is one game (three lives). After each shot the lives display decides the outcome: a
lost life is a miss, an unchanged count is a hit. When the last life goes, the game is over and
the environment waits for you to start the next one (the game has a cooldown), resuming on its own
once it sees a fresh game. Hooks in ``on_game_over`` run first, which is where the trainer saves.
"""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .capture import ScreenGrabber
from .config import Config
from .control import TrainingControl
from .controller import GameController
from .detection import Detection, LivesCounter, TemplateDetector
from .metrics import ShotStats
from .observation import build_observation
from .reward import closest_approach, shot_reward
from .tracker import HoopTracker


class HoopsEnv(gym.Env):
    def __init__(
        self,
        cfg: Config | None = None,
        grabber=None,
        controller=None,
        control: TrainingControl | None = None,
        ball_detector: TemplateDetector | None = None,
        hoop_detector: TemplateDetector | None = None,
        lives_counter: LivesCounter | None = None,
    ):
        self.cfg = cfg or Config.load()
        self.grabber = grabber or ScreenGrabber(self.cfg.region)
        self.controller = controller or GameController(self.cfg)
        self.control = control or TrainingControl()
        self.ball_detector = ball_detector or TemplateDetector(self.cfg.ball_template, self.cfg.match_threshold)
        self.hoop_detector = hoop_detector or TemplateDetector(self.cfg.hoop_template, self.cfg.match_threshold)
        self.lives_counter = lives_counter or LivesCounter(
            self.cfg.lives_template, self.cfg.lives_threshold, self.cfg.lives_region
        )
        self.tracker = HoopTracker(self.cfg.hoop_period, self.cfg.fit_tolerance_px)

        self.action_space = spaces.Discrete(2)  # 0 = wait, 1 = shoot
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

        self.on_game_over: list[Callable[[dict], None]] = []
        self.ball: Detection | None = None
        self.hoop: Detection | None = None
        self.lives = self.cfg.max_lives
        self.score = 0
        self.game_shots = 0
        self.stats = ShotStats()  # whole session, excludes cooldowns and pauses
        self.game = ShotStats()  # current game
        self._game_started = False
        self._game_over = False
        self._waits = 0
        self._last_sample = -np.inf

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

    def _read_lives(self, samples: int = 5) -> int:
        """Most common lives count over a few frames, so a flash or animation cannot fake a miss."""
        counts = []
        for _ in range(samples):
            counts.append(self.lives_counter.count(self.grabber.grab()))
            time.sleep(self.cfg.step_interval)
        return Counter(counts).most_common(1)[0][0]

    def _wait_for_game(self, need_new_game: bool) -> None:
        if need_new_game:
            print("Game over. Waiting for the next game: start it in Idleon and training resumes on its own.")
        self.ball = self.hoop = None
        last_hint = time.monotonic()
        while True:
            self.control.check()
            frame = self.grabber.grab()
            ball = self.ball_detector.find(frame)
            hoop = self.hoop_detector.find(frame)
            lives = self.lives_counter.count(frame)
            fresh = lives >= self.cfg.max_lives if need_new_game else lives > 0
            if ball and hoop and fresh:
                self.ball, self.hoop, self.lives = ball, hoop, lives
                return
            if time.monotonic() - last_hint > 30:
                print("Still waiting. If the game is already running, check `hoops-calibrate preview`.")
                last_hint = time.monotonic()
            time.sleep(0.25 if need_new_game else self.cfg.step_interval)

    def _pause_point(self) -> None:
        """Honour stop and pause requests between actions."""
        self.control.check()
        if self.control.paused:
            self.stats.pause()
            self.game.pause()
            self.control.wait_if_paused()
            self.stats.resume()
            self.game.resume()
            self.tracker.reset()  # the hoop kept moving while we were paused
            self._last_sample = -np.inf
            self._sense()

    # ---- gym API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._waits = 0
        new_game = self._game_over or not self._game_started
        self._wait_for_game(need_new_game=self._game_over)
        if new_game:
            self.tracker.reset()
            self.score = 0
            self.game_shots = 0
            self.game = ShotStats()
            self._game_started = True
            self._game_over = False
        self.stats.resume()
        self.tracker.update(time.monotonic(), self.hoop.x)
        self._last_sample = time.monotonic()
        return self._obs(), {}

    def step(self, action: int):
        self._pause_point()
        if int(action) == 1:
            return self._shoot()
        self._waits += 1
        time.sleep(self.cfg.step_interval)
        self._sense()
        truncated = self._waits >= self.cfg.max_waits_per_shot
        return self._obs(), -self.cfg.wait_penalty, False, truncated, {}

    def _shoot(self):
        self._waits = 0
        lives_before = self.lives
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

        time.sleep(self.cfg.settle_time)
        self.lives = self._read_lives()
        hit = self.lives >= lives_before
        reward = shot_reward(hit, closest_approach(ball_path, hoop_path), self.cfg)
        for stats in (self.stats, self.game):
            stats.record(hit)
        self.score += int(hit)
        self.game_shots += 1

        game_over = self.lives <= 0
        info = {
            "hit": hit,
            "score": self.score,
            "shots": self.stats.shots,
            "lives": self.lives,
            "game_over": game_over,
        }
        if game_over:
            self._game_over = True
            self.stats.pause()  # the cooldown is not playing time
            self.game.pause()
            info.update(
                game_score=self.score,
                game_shots=self.game_shots,
                game_minutes=self.game.minutes,
            )
            for hook in self.on_game_over:
                try:
                    hook(info)
                except Exception as exc:  # never lose a training session to a failed save
                    print(f"on_game_over hook failed: {exc}")
        else:
            self._sense()
        return self._obs(), reward, game_over, False, info
