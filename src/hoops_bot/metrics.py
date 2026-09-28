"""Shot, time and game counters, for the sign-off numbers in the video (script section 10)."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ShotStats:
    shots: int = 0
    hits: int = 0
    started: float = field(default_factory=time.monotonic)
    _paused_total: float = field(default=0.0, repr=False)
    _pause_started: float | None = field(default=None, repr=False)

    def record(self, hit: bool) -> None:
        self.shots += 1
        self.hits += int(hit)

    def pause(self) -> None:
        """Stop the clock (cooldowns and manual pauses do not count as playing time)."""
        if self._pause_started is None:
            self._pause_started = time.monotonic()

    def resume(self) -> None:
        if self._pause_started is not None:
            self._paused_total += time.monotonic() - self._pause_started
            self._pause_started = None

    @property
    def minutes(self) -> float:
        end = self._pause_started if self._pause_started is not None else time.monotonic()
        return (end - self.started - self._paused_total) / 60

    @property
    def accuracy(self) -> float:
        return self.hits / self.shots if self.shots else 0.0

    def summary(self) -> str:
        return (
            f"{self.hits} scores from {self.shots} shots "
            f"({self.accuracy:.0%}) in {self.minutes:.1f} minutes"
        )


class GameLog:
    """Appends one CSV row per finished game. Game numbers continue across sessions."""

    FIELDS = ["game", "finished_at", "score", "shots", "minutes"]

    def __init__(self, path: str | Path = "runs/games.csv"):
        self.path = Path(path)
        self.games = 0
        if self.path.exists():
            self.games = max(0, len(self.path.read_text().strip().splitlines()) - 1)

    def record(self, score: int, shots: int, minutes: float) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self.games += 1
        with self.path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(self.FIELDS)
            writer.writerow(
                [self.games, datetime.now().isoformat(timespec="seconds"), score, shots, f"{minutes:.2f}"]
            )
        return self.games
