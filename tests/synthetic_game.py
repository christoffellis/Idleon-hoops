"""A simulated Idleon hoops game with a fake clock, for end-to-end tests.

Hidden truth: the arc, the input latency, and where the clean-hit band sits relative to the hoop
panel's centre. Phases follow the real game: the hoop is static below 10 points and moves as a sine from
10; the player's x starts moving at 20. A hit makes the hoop respawn; three misses end the game, followed
by a cooldown. Every call to ``now()`` advances the clock a little, as polling would in real life.
"""
import math

import cv2
import numpy as np
from synthetic import BALL, HOOP, LIFE, LIFE_SPOTS, blank, draw

from hoops_bot.geometry import Viewport
from hoops_bot.hoop_model import SineModel

TWO_PI = 2 * math.pi
SCORE_REGION = (0.70, 0.02, 0.28, 0.14)
ARC = dict(d0=0.012, vx=0.9, e0=-0.012, vy=-0.6, g=2.2)


def draw_score(frame: np.ndarray, text: str) -> None:
    width, height = frame.shape[1], frame.shape[0]
    scale = width / 960
    cv2.putText(frame, text, (int(0.71 * width), int(0.10 * height)), cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * scale, (255, 255, 255), max(1, round(2 * scale)), cv2.LINE_AA)


class FakeGame:
    FRAME_DT = 1 / 30

    def __init__(self, width=480, seed=0, start_score=0, cooldown=4.0, latency=0.07,
                 center_offset=0.025, clean_half=0.012, contact_half=0.03, hoop_period=4.0):
        self.rng = np.random.default_rng(seed)
        self.width, self.height = width, round(width * 9 / 16)
        self.viewport = Viewport(self.width, self.height)
        self.t = 1000.0
        self.latency, self.cooldown = latency, cooldown
        self.center_offset, self.clean_half, self.contact_half = center_offset, clean_half, contact_half
        self.hoop_period = hoop_period
        self.player_v = SineModel(0.5, 0.10, 0.06, TWO_PI / 2.4)
        self.player_u_moving = SineModel(0.12, 0.03, 0.02, TWO_PI / 3.1)
        self.score, self.lives, self.start_score = start_score, 3, start_score
        self.over_until: float | None = None
        self.flights: list[dict] = []
        self.events: list[tuple[float, dict]] = []
        self.shots_taken = 0
        self._spawn_hoop()

    # ---- the fake clock ------------------------------------------------------------------
    def now(self) -> float:
        self.t += 0.0002
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    # ---- truth -----------------------------------------------------------------------------
    def player_u(self, t):
        return self.player_u_moving(t) if self.score >= 20 else 0.12 + 0 * np.asarray(t, dtype=float)

    def _spawn_hoop(self) -> None:
        self.hoop_v = float(self.rng.uniform(0.60, 0.72))
        centre = float(self.rng.uniform(0.68, 0.86))
        if self.score >= 10:
            phase = float(self.rng.uniform(0, TWO_PI))
            omega = TWO_PI / self.hoop_period
            self.hoop_u = SineModel(centre, 0.06 * math.cos(phase), 0.06 * math.sin(phase), omega)
        else:
            self.hoop_u = SineModel.constant(centre)

    def _ball_at(self, f: dict, tau: float):
        a = ARC
        if f["bounce"] and tau > f["tb"]:
            s = tau - f["tb"]
            return f["ub"] - 0.4 * s, f["vb"] + 0.7 * s + 0.5 * a["g"] * s * s
        return (f["pu"] + a["d0"] + a["vx"] * tau,
                f["pv"] + a["e0"] + a["vy"] * tau + 0.5 * a["g"] * tau * tau)

    # ---- input -----------------------------------------------------------------------------
    def shoot(self) -> None:
        if self.over_until is not None or self.flights:
            return
        self.shots_taken += 1
        self.truth = getattr(self, "truth", [])
        t_r = self.t + self.latency
        pu, pv = float(self.player_u(t_r)), float(self.player_v(t_r))
        lo, hi = 0.0, 3.0
        for _ in range(60):  # bisection for the moment the ball reaches the hoop's u
            tau = 0.5 * (lo + hi)
            if pu + ARC["d0"] + ARC["vx"] * tau < float(self.hoop_u(t_r + tau)):
                lo = tau
            else:
                hi = tau
        tau_hit = 0.5 * (lo + hi)
        miss = pv + ARC["e0"] + ARC["vy"] * tau_hit + 0.5 * ARC["g"] * tau_hit**2 - self.hoop_v
        offset = abs(miss - self.center_offset)
        outcome = "clean" if offset <= self.clean_half else "contact" if offset <= self.contact_half else "miss"
        flight = {"t_r": t_r, "pu": pu, "pv": pv, "tau_hit": tau_hit, "miss": miss, "outcome": outcome,
                  "bounce": outcome == "contact", "tb": tau_hit - 0.03}
        if flight["bounce"]:
            flight["ub"], flight["vb"] = self._ball_at(dict(flight, bounce=False), flight["tb"])
        tau = 0.0
        while tau < tau_hit + 0.6:
            u, v = self._ball_at(flight, tau)
            if u > 1.03 or v > 1.05 or v < -0.05:
                break
            tau += 0.01
        flight["t_end"] = t_r + tau
        self.flights.append(flight)
        self.truth.append(flight)
        self.events.append((t_r + tau_hit + 0.15, flight))

    # ---- time and events -------------------------------------------------------------------
    def _advance(self) -> None:
        for event in [e for e in self.events if e[0] <= self.t]:
            self.events.remove(event)
            outcome = event[1]["outcome"]
            self.flights = [f for f in self.flights if f is not event[1]]
            if outcome == "miss":
                self.lives -= 1
                if self.lives <= 0:
                    self.over_until = self.t + self.cooldown
            else:
                self.score += 2 if outcome == "clean" else 1
                self._spawn_hoop()
        if self.over_until is not None and self.t >= self.over_until:
            self.over_until, self.lives, self.score, self.flights, self.events = None, 3, self.start_score, [], []
            self._spawn_hoop()

    def render(self) -> np.ndarray:
        frame = blank(self.width, self.height)
        if self.over_until is not None:
            return frame
        for i in range(self.lives):
            draw(frame, LIFE, *LIFE_SPOTS[i])
        draw_score(frame, str(self.score))
        draw(frame, HOOP, float(self.hoop_u(self.t)), self.hoop_v)
        in_flight = False
        for f in self.flights:
            if f["t_r"] <= self.t <= f["t_end"] + 0.3:
                in_flight = True
                if self.t <= f["t_end"]:
                    draw(frame, BALL, *self._ball_at(f, self.t - f["t_r"]))
        if not in_flight:
            draw(frame, BALL, float(self.player_u(self.t)), float(self.player_v(self.t)))
        return frame

    def grab_timed(self):
        self.t += self.FRAME_DT
        self._advance()
        return self.render(), self.t

    def digits_frame(self) -> np.ndarray:
        frame = blank(self.width, self.height)
        draw_score(frame, "0123456789")
        return frame
