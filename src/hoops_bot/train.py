"""Train the PPO agent against the live game (script section 6).

Training runs one game at a time. After every game (three lives) the model is saved to
models/latest.zip and a row is appended to runs/games.csv. Between games the environment waits
for you to start the next one. Run the command again later and it picks up from models/latest.zip.

Controls: F8 pause / resume, F9 save and stop, or Ctrl+C in the terminal.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .config import Config
from .control import StopRequested, TrainingControl
from .env import HoopsEnv
from .metrics import GameLog

LATEST = Path("models/latest.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent on Idleon hoops.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="upper bound; you normally stop with F9")
    parser.add_argument("--fresh", action="store_true", help="ignore models/latest.zip and start over")
    args = parser.parse_args()

    cfg = Config.load()
    control = TrainingControl(cfg.pause_hotkey, cfg.stop_hotkey)
    game_env = HoopsEnv(cfg, control=control)
    env = Monitor(game_env)

    LATEST.parent.mkdir(parents=True, exist_ok=True)
    resumed = LATEST.exists() and not args.fresh
    if resumed:
        print(f"Resuming from {LATEST} (use --fresh to start over).")
        model = PPO.load(LATEST, env=env, tensorboard_log="runs/tb")
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

    log = GameLog()

    def on_game_over(info: dict) -> None:
        game = log.record(info["game_score"], info["game_shots"], info["game_minutes"])
        model.save(LATEST)
        if game % cfg.snapshot_every_games == 0:
            model.save(LATEST.parent / "games" / f"game_{game:04d}")
        print(
            f"Game {game} over: score {info['game_score']} from {info['game_shots']} shots "
            f"in {info['game_minutes']:.1f} min. Saved {LATEST}."
        )

    game_env.on_game_over.append(on_game_over)

    print(f"Controls: {cfg.pause_hotkey} pause/resume, {cfg.stop_hotkey} save and stop (Ctrl+C also works).")
    print(f"Focus the Idleon window. Starting in {cfg.countdown}s...")
    time.sleep(cfg.countdown)
    control.start_hotkeys()
    try:
        model.learn(total_timesteps=args.steps, reset_num_timesteps=not resumed, tb_log_name="ppo")
    except (StopRequested, KeyboardInterrupt):
        print("Stopping.")
    finally:
        control.stop_hotkeys()
        model.save(LATEST)
        print(f"Saved {LATEST}. {game_env.stats.summary()}")


if __name__ == "__main__":
    main()
