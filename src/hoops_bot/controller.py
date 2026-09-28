"""Game input. pyautogui is imported lazily so the rest of the package works headless."""
from __future__ import annotations

from .config import Config


class GameController:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def shoot(self) -> None:
        import pyautogui

        pyautogui.press(self.cfg.shoot_key)

    def restart(self) -> None:
        """Click the retry button if one is configured; otherwise the game restarts itself."""
        if self.cfg.restart_click is None:
            return
        import pyautogui

        pyautogui.click(*self.cfg.restart_click)
