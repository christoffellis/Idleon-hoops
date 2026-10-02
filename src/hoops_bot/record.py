"""`hoops-record`: you play, the bot logs every shot (no keys are pressed by the bot).

Use it to collect shots by hand, e.g. in a game you are playing anyway. Each press of the shoot key is
logged with the ball's path, the score and the lives, exactly like a bot shot.
"""
from __future__ import annotations

import argparse
import queue
import time

from .calibrate import build_session
from .config import Config
from .control import StopRequested
from .metrics import GameLog, ShotStats


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    from pynput import keyboard

    cfg = Config.load()
    session, control = build_session(cfg)
    presses: queue.Queue[float] = queue.Queue()
    key = cfg.shoot_key
    wanted = keyboard.Key.space if key == "space" else keyboard.KeyCode.from_char(key)

    def on_press(k):
        if k == wanted and session.ready() and not session.in_flight:
            presses.put(session.clock())

    listener = keyboard.Listener(on_press=on_press)
    print(f"Controls: {cfg.pause_hotkey} pause, {cfg.stop_hotkey} stop. Play normally; shots are logged.")
    time.sleep(cfg.countdown)
    control.start_hotkeys()
    listener.start()
    stats, games = ShotStats(), GameLog()
    session.wait_ready()
    session.lives, session.score = session.read_lives(), session.read_score()
    try:
        while True:
            session._control_point()
            session.sense()
            try:
                t_press = presses.get_nowait()
            except queue.Empty:
                continue
            shot = session.observe_shot(t_press)
            stats.record(shot["outcome"] != "miss")
            print(f"shot {shot['id']:>3} {shot['outcome']:<7} miss {shot['signed_miss']}  score {shot['score_after']}")
            if shot["lives_after"] is not None and shot["lives_after"] <= 0:
                games.record(shot["score_after"] or 0, stats.shots, stats.minutes)
                session.wait_for_new_game()
    except (StopRequested, KeyboardInterrupt):
        print(f"Stopped. {stats.summary()}")
    finally:
        listener.stop()
        control.stop_hotkeys()


if __name__ == "__main__":
    main()
