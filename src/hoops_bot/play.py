"""Run a trained agent until it reaches the trophy score, then print the sign-off numbers."""
from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from .config import Config
from .env import HoopsEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Idleon hoops with a trained model.")
    parser.add_argument("--model", type=str, default="models/hoops_ppo")
    parser.add_argument("--target", type=int, default=None, help="stop at this score")
    args = parser.parse_args()

    cfg = Config.load()
    target = args.target or cfg.target_score
    model = PPO.load(args.model)
    env = HoopsEnv(cfg)

    print(f"Focus the Idleon window. Starting in {cfg.countdown}s...")
    time.sleep(cfg.countdown)

    obs, _ = env.reset()
    while env.score < target:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        if info.get("hit"):
            print(f"score {env.score}/{target}  ({env.stats.summary()})")
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Target reached: {env.stats.summary()}")


if __name__ == "__main__":
    main()
