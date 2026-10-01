"""Render the illustrations with a chosen theme.

    python -m animations.render --list
    python -m animations.render nyquist
    python -m animations.render --all --theme forest -q m
    python -m animations.render reward --set primary=#aa5533 --transparent

Wraps the `manim` command. Themes are files in animations/themes/ (or a path to your own JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from animations.theme import builtin_themes

ROOT = Path(__file__).resolve().parents[1]
SCENES_DIR = Path(__file__).parent / "scenes"

# key -> (file, class, script section)
SCENES: dict[str, tuple[str, str, str]] = {
    "plan": ("plan.py", "GamePlan", "3 The game plan"),
    "phases": ("plan.py", "GamePhases", "4 How the game escalates"),
    "inputs": ("inputs.py", "RelativeInputs", "5 Four inputs to two"),
    "reward": ("learning.py", "RewardFunction", "6 The reward function"),
    "detection": ("learning.py", "DetectionDemo", "7 Finding the ball"),
    "nyquist": ("hoop_motion.py", "NyquistSampling", "8 Nyquist sampling"),
    "prediction": ("hoop_motion.py", "HoopPrediction", "8 Predicting the hoop"),
    "moving-player": ("inputs.py", "MovingPlayer", "9 Phase 3, moving player"),
}


def build_command(key: str, quality: str = "h", media_dir: str = "media", transparent: bool = False, preview: bool = False) -> list[str]:
    filename, class_name, _ = SCENES[key]
    command = [sys.executable, "-m", "manim", f"-q{quality}", "--media_dir", media_dir]
    if transparent:
        command.append("--transparent")
    if preview:
        command.append("-p")
    command += [str(SCENES_DIR / filename), class_name]
    return command


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*", help="scene keys (see --list)")
    parser.add_argument("--all", action="store_true", help="render every scene")
    parser.add_argument("--list", action="store_true", help="list scenes and themes")
    parser.add_argument("--theme", default=None, help=f"theme name ({', '.join(builtin_themes())}) or path to a JSON file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override one theme value")
    parser.add_argument("-q", "--quality", default="h", choices=["l", "m", "h", "p", "k"], help="l 480p, m 720p, h 1080p, p 1440p, k 4K")
    parser.add_argument("--transparent", action="store_true", help="transparent background, to overlay in your editor")
    parser.add_argument("-p", "--preview", action="store_true", help="open each render when done")
    parser.add_argument("--media-dir", default="media")
    args = parser.parse_args()

    if args.list:
        print("Scenes:")
        for key, (_, class_name, section) in SCENES.items():
            print(f"  {key:<14} {class_name:<16} script section {section}")
        print(f"Themes: {', '.join(builtin_themes())}")
        return

    keys = list(SCENES) if args.all else args.scenes
    if not keys:
        parser.error("name at least one scene, or use --all (see --list)")
    unknown = [key for key in keys if key not in SCENES]
    if unknown:
        parser.error(f"unknown scene(s): {', '.join(unknown)}. See --list.")

    env = os.environ.copy()
    if args.theme:
        env["HOOPS_THEME"] = args.theme
    if args.overrides:
        env["HOOPS_THEME_OVERRIDES"] = json.dumps(parse_overrides(args.overrides))
    for key in keys:
        print(f"Rendering {key}...")
        result = subprocess.run(build_command(key, args.quality, args.media_dir, args.transparent, args.preview), env=env, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
