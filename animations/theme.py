"""Illustration theme: colours, font and sizes, loaded from a small JSON file.

This module deliberately does not import manim, so themes can be validated and tested anywhere.

Pick a theme by name (a file in animations/themes/) or by path to your own JSON file:

    HOOPS_THEME=forest manim ...
    HOOPS_THEME=./my-theme.json manim ...

Single values can be overridden without a new file, using a JSON object in HOOPS_THEME_OVERRIDES:

    HOOPS_THEME_OVERRIDES='{"primary": "#aa5533"}' manim ...
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

THEMES_DIR = Path(__file__).parent / "themes"
DEFAULT_THEME = "earthy"
COLOUR_FIELDS = (
    "background", "text", "muted", "grid", "surface",
    "primary", "secondary", "accent", "hit", "miss",
)
NUMBER_FIELDS = ("stroke_width", "title_size", "label_size", "small_size")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class Theme:
    name: str = "earthy"
    font: str = "sans-serif"

    # Surfaces and text
    background: str = "#F4EFE6"  # warm paper
    surface: str = "#EAE1D0"     # cards and the mini game field
    grid: str = "#DDD3C2"        # chart grid lines
    text: str = "#3B2F2F"        # dark bark
    muted: str = "#8C7B6B"       # axes, captions

    # Roles. primary = the hoop and the signal, secondary = the player and samples,
    # accent = highlights and the thing being measured.
    primary: str = "#B85C38"     # terracotta
    secondary: str = "#6B7F4F"   # olive
    accent: str = "#D9A441"      # ochre
    hit: str = "#6B7F4F"
    miss: str = "#A8442F"

    # Sizes
    stroke_width: float = 4.0
    title_size: int = 44
    label_size: int = 28
    small_size: int = 22


def builtin_themes() -> list[str]:
    return sorted(path.stem for path in THEMES_DIR.glob("*.json"))


def _validate(data: dict) -> Theme:
    known = {f.name for f in fields(Theme)}
    unknown = sorted(set(data) - known - {"comment"})
    if unknown:
        raise ValueError(f"Unknown theme keys: {', '.join(unknown)}. Valid keys: {', '.join(sorted(known))}")
    data = {key: value for key, value in data.items() if key != "comment"}
    merged = {**asdict(Theme()), **data}
    for key in COLOUR_FIELDS:
        if not isinstance(merged[key], str) or not _HEX.match(merged[key]):
            raise ValueError(f"Theme colour '{key}' must be a hex colour like #A1B2C3, got {merged[key]!r}")
    for key in NUMBER_FIELDS:
        if not isinstance(merged[key], (int, float)) or isinstance(merged[key], bool) or merged[key] <= 0:
            raise ValueError(f"Theme value '{key}' must be a positive number, got {merged[key]!r}")
    return Theme(**merged)


def load_theme(spec: str | None = None) -> Theme:
    """Load a theme by built-in name or JSON path. Falls back to HOOPS_THEME, then the default."""
    spec = spec or os.environ.get("HOOPS_THEME") or DEFAULT_THEME
    path = Path(spec).expanduser()
    if path.suffix != ".json":
        path = THEMES_DIR / f"{spec}.json"
    if not path.exists():
        raise FileNotFoundError(f"Theme not found: {spec}. Built-in themes: {', '.join(builtin_themes())}")
    data = json.loads(path.read_text())
    overrides = os.environ.get("HOOPS_THEME_OVERRIDES")
    if overrides:
        data.update(json.loads(overrides))
    return _validate(data)
