"""A live game session: sense the screen, track the motions, fire a shot, log what happened.

Everything time-related goes through ``clock`` and ``sleep`` so the whole loop can be tested against
a simulated game with a fake clock.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .detection import Detection, LivesCounter, TemplateDetector
from .hoop_model import SineModel, SineTrack
from .score import ScoreReader
from .shots import analyse_shot, append_shot
from .window import classify_outcome


class PlanInvalid(Exception):
    """The hoop respawned while we were waiting to fire, so the plan no longer holds."""


@dataclass
class Observation:
    t: float
    frame: np.ndarray
    ball: Detection | None
    hoop: Detection | None


class GameSession:
    def __init__(
        self,
        cfg: Config,
        grabber,
        controller,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        control=None,
        ball_detector=None,
        hoop_detector=None,
        lives_counter=None,
        score_reader=None,
    ):
        self.cfg, self.grabber, self.controller = cfg, grabber, controller
        self.clock, self.sleep, self.control = clock, sleep, control
        ref = cfg.reference_width()
        self.ball_detector = ball_detector or TemplateDetector(cfg.ball_template, cfg.match_threshold, ref)
        self.hoop_detector = hoop_detector or TemplateDetector(cfg.hoop_template, cfg.match_threshold, ref)
        self.lives_counter = lives_counter or LivesCounter(cfg.life_template, cfg.lives_threshold, cfg.lives_region, ref)
        self.score_reader = score_reader or ScoreReader(cfg.digits_dir, cfg.score_region, cfg.score_threshold, ref)
        self.log_path = Path(cfg.shots_file)
        self._new_tracks()
        self.frame_dt = 0.03
        self._last_t: float | None = None
        self.in_flight = False
        self.ball_history: deque[tuple[float, float, float]] = deque(maxlen=60)
        self.hoop_history: deque[tuple[float, float, float]] = deque(maxlen=120)
        self.last_frame: np.ndarray | None = None
        self.lives: int | None = None
        self.score: int | None = None
        self.shot_count = 0

    # ---- tracking state ---------------------------------------------------------------
    def _new_tracks(self) -> None:
        tol, cfg = self.cfg.track_tolerance, self.cfg
        self.player_u = SineTrack(cfg.player_u_period, tol)
        self.player_v = SineTrack(cfg.player_v_period, tol)
        self.hoop_u = SineTrack(cfg.hoop_period, tol)
        self.hoop_v_samples: deque[float] = deque(maxlen=60)
        self.hoop_resets = 0
        self._v_jump = 0

    @property
    def hoop_v(self) -> float | None:
        return float(np.median(self.hoop_v_samples)) if self.hoop_v_samples else None

    def ready(self) -> bool:
        return (self.hoop_u.ready and self.player_v.ready and self.player_u.ready
                and len(self.hoop_v_samples) >= 5)

    # ---- sensing ----------------------------------------------------------------------
    def sense(self) -> Observation:
        frame, t = self.grabber.grab_timed()
        if self._last_t is not None:
            self.frame_dt = 0.9 * self.frame_dt + 0.1 * max(1e-3, t - self._last_t)
        self._last_t = t
        self.last_frame = frame
        ball = self.ball_detector.find(frame)
        hoop = self.hoop_detector.find(frame)
        if hoop is not None:
            self._feed_hoop(t, hoop)
            self.hoop_history.append((t, hoop.u, hoop.v))
        if ball is not None:
            self.ball_history.append((t, ball.u, ball.v))
            if not self.in_flight:
                self.player_u.update(t, ball.u)
                self.player_v.update(t, ball.v)
        return Observation(t, frame, ball, hoop)

    def _feed_hoop(self, t: float, hoop: Detection) -> None:
        respawned = self.hoop_u.update(t, hoop.u)
        median_v = self.hoop_v
        if median_v is not None and abs(hoop.v - median_v) > self.cfg.track_tolerance:
            self._v_jump += 1
            if self._v_jump >= 3 and not respawned:  # the hoop moved vertically: also a respawn
                self.hoop_u.reset(keep=[(t, hoop.u)])
                respawned = True
        else:
            self._v_jump = 0
        if respawned:
            self.hoop_v_samples.clear()
            self.hoop_resets += 1
            self._v_jump = 0
        self.hoop_v_samples.append(hoop.v)

    def idle(self, seconds: float) -> None:
        end = self.clock() + seconds
        while self.clock() < end:
            self.sense()

    def _control_point(self) -> None:
        if self.control is not None:
            self.control.check()
            self.control.wait_if_paused()

    def read_lives(self, samples: int = 3) -> int:
        counts = []
        for _ in range(samples):
            counts.append(self.lives_counter.count(self.sense().frame))
        return Counter(counts).most_common(1)[0][0]

    def read_score(self, samples: int = 3) -> int | None:
        reads = [r for r in (self.score_reader.read(self.sense().frame) for _ in range(samples)) if r is not None]
        return Counter(reads).most_common(1)[0][0] if reads else None

    # ---- waiting ----------------------------------------------------------------------
    def wait_ready(self, timeout: float = 60.0) -> None:
        end = self.clock() + timeout
        while not self.ready():
            self._control_point()
            if self.clock() > end:
                raise TimeoutError(
                    "Could not lock onto the player and hoop. Check `hoops-calibrate preview`: "
                    "the ball and hoop should both be boxed, and the motion periods set."
                )
            self.sense()

    def wait_for_new_game(self, hint_every: float = 30.0) -> None:
        """After the last life is lost: wait out the cooldown until a fresh game (full lives) is on screen."""
        print("Game over. Start the next game in Idleon when the cooldown ends; the bot resumes by itself.")
        last_hint = self.clock()
        while True:
            self._control_point()
            obs = self.sense()
            if obs.ball and obs.hoop and self.lives_counter.count(obs.frame) >= self.cfg.max_lives:
                break
            if self.clock() - last_hint > hint_every:
                print("Still waiting for a fresh game (ball, hoop and full lives on screen)...")
                last_hint = self.clock()
            self.sleep(0.1)
        self._new_tracks()
        self.ball_history.clear()
        self.hoop_history.clear()
        self.lives = self.cfg.max_lives
        self.score = self.read_score()

    def wait_until(self, t_fire: float) -> None:
        """Keep sensing until just before ``t_fire``, then spin so the press is not frame-quantised."""
        resets = self.hoop_resets
        while t_fire - self.clock() > 1.5 * self.frame_dt:
            self.sense()
            if self.hoop_resets != resets:
                raise PlanInvalid
        while self.clock() < t_fire:  # final stretch: spin, so the press lands on time
            pass

    # ---- shots ------------------------------------------------------------------------
    def snapshot(self) -> tuple[dict, dict]:
        """The player and hoop motion models at this moment, as plain dicts for the shot log."""
        def model(track: SineTrack) -> dict:
            if track.model is not None:
                return track.model.to_dict()
            return SineModel.constant(track.last if track.last is not None else 0.0).to_dict()

        return {"u": model(self.player_u), "v": model(self.player_v)}, {"u": model(self.hoop_u), "v": self.hoop_v}

    def press(self) -> float:
        t = self.clock()
        self.controller.shoot()
        return t

    def _far_from(self, last: list[float], ball: Detection, limit: float) -> bool:
        aspect = self.grabber.viewport.aspect
        return float(np.hypot((ball.u - last[1]) * aspect, ball.v - last[2])) > limit

    def track_flight(self, t_press: float) -> list[list[float]]:
        """Follow the ball from the press until it leaves the screen, stops, or times out."""
        self.in_flight = True
        path = [[t, u, v] for t, u, v in self.ball_history if t >= t_press]
        last_seen = path[-1][0] if path else t_press
        try:
            while True:
                obs = self.sense()
                if obs.ball is not None:
                    if path and self._far_from(path[-1], obs.ball, self.cfg.respawn_jump):
                        break  # a new ball appeared at the player: the flight is over
                    path.append([obs.t, obs.ball.u, obs.ball.v])
                    last_seen = obs.t
                    if not (-0.05 <= obs.ball.v <= 1.05) or obs.ball.u > 1.03:
                        break
                if obs.t - t_press > self.cfg.flight_timeout or obs.t - last_seen > 0.3:
                    break
        finally:
            self.in_flight = False
        return path

    def _finish(self, t_press: float, path, players: dict, hoop: dict, mode: str, target: float | None) -> dict:
        lives_before, score_before = self.lives, self.score
        self.idle(self.cfg.settle_time)
        lives_after, score_after = self.read_lives(), self.read_score()
        score_delta = None if score_before is None or score_after is None else score_after - score_before
        lives_delta = 0 if lives_before is None else lives_after - lives_before
        self.shot_count += 1
        viewport = self.grabber.viewport
        end = path[-1][0] if path else t_press
        hoop = dict(hoop, trace=[[t, u, v] for t, u, v in self.hoop_history if t_press - 0.6 <= t <= end + 0.05])
        shot = {
            "id": self.shot_count, "mode": mode, "t_press": t_press, "target_miss": target,
            "viewport": {"w": viewport.width, "h": viewport.height},
            "player": players, "hoop": hoop, "path": path,
            "score_before": score_before, "score_after": score_after,
            "lives_before": lives_before, "lives_after": lives_after,
        }
        shot = analyse_shot(shot)
        shot["outcome"] = classify_outcome(score_delta, lives_delta, shot["contact"])
        append_shot(self.log_path, shot)
        self.lives, self.score = lives_after, score_after if score_after is not None else self.score
        return shot

    def play_shot(self, t_fire: float, mode: str, target: float | None = None) -> dict:
        """Wait for ``t_fire``, press, follow the ball, read the result, log it."""
        self.wait_until(t_fire)
        players, hoop = self.snapshot()
        t_press = self.press()
        path = self.track_flight(t_press)
        return self._finish(t_press, path, players, hoop, mode, target)

    def observe_shot(self, t_press: float) -> dict:
        """A shot the human took at ``t_press``: follow it and log it the same way."""
        players, hoop = self.snapshot()
        path = self.track_flight(t_press)
        return self._finish(t_press, path, players, hoop, "observe", None)
