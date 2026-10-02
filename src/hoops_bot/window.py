"""The hit window: which signed misses score clean, which only score on a bounce, which miss.

The signed miss is how far (vertically, in height units) the ball's arc passes from the hoop
panel's centre at the hoop. The window is learned online from outcomes, so hits teach it for free:
the clean band sits around the clean hits, and its edges are tightened by the nearest misses.
"""
from __future__ import annotations

CLEAN, CONTACT, MISS = "clean", "contact", "miss"


def classify_outcome(score_delta: int | None, lives_delta: int, contact: bool) -> str:
    """Score change is the ground truth when it can be read (+2 clean, +1 contact, 0 miss).
    Otherwise fall back on the lives and the contact flag."""
    if score_delta is not None:
        if score_delta >= 2:
            return CLEAN
        if score_delta == 1:
            return CONTACT
        return MISS
    if lives_delta < 0:
        return MISS
    return CONTACT if contact else CLEAN


class HitWindow:
    def __init__(self, default_width: float = 0.02, probe_step: float = 0.02, prior: float = 0.0):
        self.default_width = default_width
        self.probe_step = probe_step
        self.prior = prior
        self.observations: list[tuple[float, str]] = []

    def update(self, miss: float, outcome: str) -> None:
        self.observations.append((float(miss), outcome))

    def band(self) -> tuple[float, float] | None:
        """Estimated (low, high) edges of the clean band, or None before the first clean hit."""
        clean = [m for m, o in self.observations if o == CLEAN]
        if not clean:
            return None
        a, b = min(clean), max(clean)
        others = [m for m, o in self.observations if o != CLEAN]
        below = [m for m in others if m < a]
        above = [m for m in others if m > b]
        low = (max(below) + a) / 2 if below else a - self.default_width / 2
        high = (min(above) + b) / 2 if above else b + self.default_width / 2
        return low, high

    @property
    def found(self) -> bool:
        return self.band() is not None

    @property
    def width(self) -> float:
        band = self.band()
        return self.default_width if band is None else band[1] - band[0]

    def next_probe(self) -> float:
        """The signed miss to aim for while no clean hit has been found: the prior, then outward
        in steps, alternating sides, skipping places already tried."""
        tried = [m for m, _ in self.observations]
        k = 0
        while True:
            offset = 0.0 if k == 0 else ((k + 1) // 2) * self.probe_step * (1 if k % 2 else -1)
            candidate = self.prior + offset
            if all(abs(candidate - m) >= self.probe_step / 2 for m in tried):
                return candidate
            k += 1

    def target(self) -> float:
        band = self.band()
        return self.next_probe() if band is None else (band[0] + band[1]) / 2

    def to_dict(self) -> dict:
        return {"observations": self.observations, "band": self.band(), "target": self.target()}
