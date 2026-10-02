"""The whole loop against a simulated game: cold start, probing, window search, aiming, game over."""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from synthetic import BALL, HOOP, LIFE, REF_W
from synthetic_game import ARC, SCORE_REGION, FakeGame

from hoops_bot.aim import AimLoop
from hoops_bot.config import Config
from hoops_bot.control import StopRequested, TrainingControl
from hoops_bot.detection import LivesCounter, TemplateDetector
from hoops_bot.metrics import GameLog
from hoops_bot.score import ScoreReader, learn_digits
from hoops_bot.session import GameSession


def build(game, folder, max_shots, **cfg_over):
    digits = Path(folder) / "digits"
    learn_digits(game.digits_frame(), SCORE_REGION, "0123456789", digits)
    cfg = Config(
        shots_file=str(Path(folder) / "shots.jsonl"), digits_dir=str(digits), score_region=SCORE_REGION,
        hoop_period=4.0, player_v_period=2.4, settle_time=0.5, flight_timeout=2.0, max_wait=15.0, **cfg_over,
    )
    session = GameSession(
        cfg, game, SimpleNamespace(shoot=game.shoot), clock=game.now, sleep=game.sleep, control=TrainingControl(),
        ball_detector=TemplateDetector(BALL, 0.8, REF_W), hoop_detector=TemplateDetector(HOOP, 0.8, REF_W),
        lives_counter=LivesCounter(LIFE, 0.8, None, REF_W), score_reader=ScoreReader(digits, SCORE_REGION, 0.7),
    )
    lines = []
    loop = AimLoop(session, cfg, max_shots=max_shots, log=lines.append, game_log=GameLog(Path(folder) / "games.csv"))
    return loop, lines


def settled(shots, tail=4, need=3):
    """The last few shots are mostly clean hits: the window was found and is being aimed at."""
    outcomes = [s["outcome"] for s in shots][-tail:]
    assert outcomes.count("clean") >= need, outcomes
    return True


def test_cold_start_static_hoop_finds_the_window_and_then_keeps_hitting():
    game = FakeGame(seed=1)
    with tempfile.TemporaryDirectory() as folder:
        loop, lines = build(game, folder, max_shots=16)
        loop.run()
        shots = loop.shots
    assert len(shots) == 16
    assert shots[0]["mode"] == "probe"                     # nothing known yet: probe at varied phases
    assert loop.arc is not None
    assert loop.arc.vx == pytest.approx(ARC["vx"], abs=0.04)
    assert loop.arc.latency == pytest.approx(game.latency, abs=0.02)
    assert settled(shots)                                  # once the window is found, aiming keeps hitting it
    assert loop.window.band() is not None
    centre = sum(loop.window.band()) / 2
    assert centre == pytest.approx(game.center_offset, abs=0.012)  # the hidden offset was learned


def test_moving_hoop_is_intercepted():
    game = FakeGame(seed=2, start_score=10)
    with tempfile.TemporaryDirectory() as folder:
        loop, _ = build(game, folder, max_shots=14, player_u_period=3.1)  # from `hoops-calibrate motion`
        loop.run()
        shots = loop.shots
    assert settled(shots)


def test_moving_hoop_and_moving_player():
    game = FakeGame(seed=3, start_score=20)
    with tempfile.TemporaryDirectory() as folder:
        loop, _ = build(game, folder, max_shots=16, player_u_period=3.1)
        loop.run()
        shots = loop.shots
    assert settled(shots)


def test_game_over_is_handled_and_play_resumes_after_the_cooldown():
    game = FakeGame(seed=4, center_offset=0.2)             # window far from the first probes: lives get lost
    with tempfile.TemporaryDirectory() as folder:
        loop, lines = build(game, folder, max_shots=7)
        loop.run()
        games = (Path(folder) / "games.csv").read_text().strip().splitlines()
    assert any("Game 1 over" in line for line in lines)
    assert len(games) >= 2                                  # header plus at least one finished game
    assert loop.shots[-1]["lives_before"] is not None


def test_stop_request_ends_the_loop_cleanly():
    game = FakeGame(seed=5)
    with tempfile.TemporaryDirectory() as folder:
        loop, _ = build(game, folder, max_shots=50)
        loop.s.control.request_stop()
        with pytest.raises(StopRequested):
            loop.run()
