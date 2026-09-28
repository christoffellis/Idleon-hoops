"""End-to-end environment test against a scripted fake game (needs gymnasium)."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from hoops_bot.config import Config
from hoops_bot.control import StopRequested, TrainingControl
from hoops_bot.detection import LivesCounter, TemplateDetector
from hoops_bot.env import HoopsEnv


def pattern(size, seed):
    return np.random.default_rng(seed).integers(30, 255, (size, size, 3), dtype=np.uint8)


BALL, HOOP, LIFE = pattern(16, 1), pattern(24, 2), pattern(12, 3)


class FakeGame:
    """Shots resolve to hit or miss in the given order. A miss removes a life; losing the last one
    blanks the screen for a short cooldown, then a fresh game with full lives appears."""

    def __init__(self, outcomes, cooldown=0.3):
        self.outcomes = list(outcomes)
        self.shots = 0
        self.lives = 3
        self.cooldown = cooldown
        self.reopen_at = 0.0

    def frame(self):
        image = np.zeros((300, 400, 3), np.uint8)
        if self.lives == 0:
            if time.monotonic() < self.reopen_at:
                return image
            self.lives = 3
        image[142:158, 42:58] = BALL
        image[88:112, 288:312] = HOOP
        for i in range(self.lives):
            image[10:22, 10 + 20 * i : 22 + 20 * i] = LIFE
        return image

    def shoot(self):
        hit = self.outcomes[self.shots]
        self.shots += 1
        if not hit:
            self.lives -= 1
            if self.lives == 0:
                self.reopen_at = time.monotonic() + self.cooldown


def make_env(outcomes):
    game = FakeGame(outcomes)
    cfg = Config(step_interval=0.001, sample_interval=0.01, flight_timeout=0.05, settle_time=0.02)
    env = HoopsEnv(
        cfg,
        grabber=SimpleNamespace(grab=game.frame),
        controller=SimpleNamespace(shoot=game.shoot),
        control=TrainingControl(),
        ball_detector=TemplateDetector(BALL, 0.9),
        hoop_detector=TemplateDetector(HOOP, 0.9),
        lives_counter=LivesCounter(LIFE, 0.9),
    )
    return env, game


def shoot_until_game_over(env):
    results = []
    while True:
        _, reward, terminated, _, info = env.step(1)
        results.append((reward, info))
        if terminated:
            return results


def test_lives_decide_hit_or_miss_and_end_the_game():
    env, _ = make_env([True, False, True, False, False])
    saved = []
    env.on_game_over.append(saved.append)
    env.reset()
    results = shoot_until_game_over(env)

    assert [info["hit"] for _, info in results] == [True, False, True, False, False]
    assert results[0][0] > 0 > results[1][0]
    last = results[-1][1]
    assert last["game_over"] and last["lives"] == 0
    assert (last["game_score"], last["game_shots"]) == (2, 5)
    assert saved == [last]  # the save hook ran before the cooldown wait


def test_next_game_starts_automatically_and_clears_the_score():
    env, game = make_env([False, False, False, True])
    env.reset()
    shoot_until_game_over(env)
    start = time.monotonic()
    env.reset()  # blocks through the cooldown, then sees a fresh game
    assert time.monotonic() - start >= 0.2
    assert (game.lives, env.lives, env.score, env.game_shots) == (3, 3, 0, 0)


def test_stop_request_ends_training_cleanly():
    env, _ = make_env([True])
    env.reset()
    env.control.request_stop()
    with pytest.raises(StopRequested):
        env.step(0)
