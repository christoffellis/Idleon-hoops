"""Game input. pyautogui is imported lazily so the rest of the package works headless."""
from __future__ import annotations

from .config import Config


class GameController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        import pyautogui

        pyautogui.PAUSE = 0  # the default 0.1s pause after every call would wreck the timing

    def shoot(self) -> None:
        import pyautogui

        pyautogui.press(self.cfg.shoot_key)
