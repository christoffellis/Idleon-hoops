"""Train the PPO agent against the live game (script section 6)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .config import Config
from .env import HoopsEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent on Idleon hoops.")
    parser.add_argument("--steps", type=int, default=20_000, help="total environment steps")
    parser.add_argument("--resume", type=str, default=None, help="path to a saved model to continue")
    parser.add_argument("--out", type=str, default="models/hoops_ppo")
    args = parser.parse_args()

    cfg = Config.load()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Focus the Idleon window. Starting in {cfg.countdown}s...")
    time.sleep(cfg.countdown)

    env = Monitor(HoopsEnv(cfg))
    if args.resume:
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.95,
            verbose=1,
            tensorboard_log="runs/tb",
        )
    checkpoints = CheckpointCallback(save_freq=2_000, save_path="models/checkpoints", name_prefix="hoops")
    model.learn(total_timesteps=args.steps, callback=checkpoints, reset_num_timesteps=not args.resume)
    model.save(args.out)
    print(f"Saved {args.out}.zip")


if __name__ == "__main__":
    main()
