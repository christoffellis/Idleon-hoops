"""Run a trained agent until it reaches the trophy score, then print the sign-off numbers.

Games that end early are followed by the same wait-for-the-next-game behaviour as training, and
cooldowns and pauses do not count towards the reported minutes.
Controls: F8 pause / resume, F9 stop.
"""
from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from .config import Config
from .control import StopRequested, TrainingControl
from .env import HoopsEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Idleon hoops with a trained model.")
    parser.add_argument("--model", type=str, default="models/latest")
    parser.add_argument("--target", type=int, default=None, help="stop at this score in one game")
    args = parser.parse_args()

    cfg = Config.load()
    target = args.target or cfg.target_score
    control = TrainingControl(cfg.pause_hotkey, cfg.stop_hotkey)
    model = PPO.load(args.model)
    env = HoopsEnv(cfg, control=control)

    print(f"Controls: {cfg.pause_hotkey} pause/resume, {cfg.stop_hotkey} stop (Ctrl+C also works).")
    print(f"Focus the Idleon window. Starting in {cfg.countdown}s...")
    time.sleep(cfg.countdown)
    control.start_hotkeys()
    try:
        obs, _ = env.reset()
        while env.score < target:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            if info.get("hit"):
                print(f"score {env.score}/{target}  ({env.stats.summary()})")
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"Target reached: {env.stats.summary()}")
    except (StopRequested, KeyboardInterrupt):
        print(f"Stopped. {env.stats.summary()}")
    finally:
        control.stop_hotkeys()


if __name__ == "__main__":
    main()
