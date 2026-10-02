"""Read the score from the screen with digit templates.

Only the change in score per shot matters (+2 clean, +1 contact, 0 miss), but the whole number is
read so that a missed digit shows up as an obvious jump rather than a silent error.

Digit templates are made once with `hoops-calibrate digits`, which cuts the glyphs out of a frame
while you tell it what the score says.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .detection import ScaledTemplate, crop_fraction


class ScoreReader:
    def __init__(
        self,
        digits_dir: str | Path,
        region: tuple[float, float, float, float] | None,
        threshold: float = 0.7,
        reference_width: float | None = None,
    ):
        self.region = region
        self.threshold = threshold
        folder = Path(digits_dir)
        ref_file = folder / "reference.json"
        if ref_file.exists():
            reference_width = json.loads(ref_file.read_text()).get("reference_width", reference_width)
        self.templates: dict[str, ScaledTemplate] = {}
        for digit in "0123456789":
            path = folder / f"{digit}.png"
            if path.exists():
                self.templates[digit] = ScaledTemplate(cv2.imread(str(path), cv2.IMREAD_COLOR), reference_width)

    @property
    def missing(self) -> list[str]:
        return [d for d in "0123456789" if d not in self.templates]

    def read(self, frame: np.ndarray) -> int | None:
        """The score currently shown, or None if no digits are recognised."""
        if not self.templates:
            return None
        crop = crop_fraction(frame, self.region)
        hits: list[tuple[float, int, str, int]] = []  # (score, x, digit, width)
        for digit, template in self.templates.items():
            tpl = template.for_frame(frame)
            h, w = tpl.shape[:2]
            if crop.shape[0] < h or crop.shape[1] < w:
                continue
            result = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= self.threshold)
            for y, x in zip(ys, xs):
                hits.append((float(result[y, x]), int(x), digit, w))
        if not hits:
            return None
        min_width = min(h[3] for h in hits)
        kept: list[tuple[float, int, str, int]] = []
        for hit in sorted(hits, reverse=True):  # strongest first; drop weaker overlapping matches
            if all(abs(hit[1] - other[1]) >= 0.6 * min_width for other in kept):
                kept.append(hit)
        return int("".join(h[2] for h in sorted(kept, key=lambda h: h[1])))


def learn_digits(
    frame: np.ndarray,
    region: tuple[float, float, float, float] | None,
    text: str,
    digits_dir: str | Path,
) -> list[str]:
    """Cut the digit glyphs out of ``frame`` (which shows the score ``text``) and save them as
    templates. Returns the digits saved. Raises ValueError if the glyph count does not match."""
    crop = crop_fraction(frame, region)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size / 2:  # make the glyphs the foreground
        binary = cv2.bitwise_not(binary)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = [stats[i] for i in range(1, count) if stats[i][cv2.CC_STAT_AREA] >= 6]
    boxes.sort(key=lambda s: s[cv2.CC_STAT_LEFT])
    if len(boxes) != len(text):
        raise ValueError(
            f"Found {len(boxes)} glyphs but you typed {len(text)} digits. "
            "Tighten the score region so it holds only the number."
        )
    folder = Path(digits_dir)
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for char, s in zip(text, boxes):
        x, y, w, h = (int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]),
                      int(s[cv2.CC_STAT_WIDTH]), int(s[cv2.CC_STAT_HEIGHT]))
        cv2.imwrite(str(folder / f"{char}.png"), crop[max(0, y - 1) : y + h + 1, max(0, x - 1) : x + w + 1])
        saved.append(char)
    (folder / "reference.json").write_text(json.dumps({"reference_width": frame.shape[1]}))
    return saved
