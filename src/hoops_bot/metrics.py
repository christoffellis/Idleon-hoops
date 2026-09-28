"""Shot and time counters, for the sign-off numbers in the video (script section 10)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ShotStats:
    shots: int = 0
    hits: int = 0
    started: float = field(default_factory=time.monotonic)

    def record(self, hit: bool) -> None:
        self.shots += 1
        self.hits += int(hit)

    @property
    def minutes(self) -> float:
        return (time.monotonic() - self.started) / 60

    @property
    def accuracy(self) -> float:
        return self.hits / self.shots if self.shots else 0.0

    def summary(self) -> str:
        return (
            f"{self.hits} scores from {self.shots} shots "
            f"({self.accuracy:.0%}) in {self.minutes:.1f} minutes"
        )
