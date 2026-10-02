"""Synthetic game frames for tests: sprites defined at a reference size, drawn at any resolution.

The detector and these helpers resize sprites with the same function, so a scene rendered at
960x540 and at 1920x1080 should give the same normalised detections.
"""
import numpy as np

from hoops_bot.detection import resize_template

REF_W, REF_H = 960, 540


def texture(h: int, w: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(40, 255, (h, w, 3), dtype=np.uint8)


BALL = texture(20, 20, 1)
HOOP = texture(44, 16, 2)
LIFE = texture(14, 14, 3)
LIFE_SPOTS = [(0.04 + 0.03 * i, 0.05) for i in range(3)]


def blank(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 90, dtype=np.uint8)


def draw(frame: np.ndarray, sprite: np.ndarray, u: float, v: float) -> None:
    """Draw a reference-size sprite centred at normalised (u, v), scaled to the frame width."""
    height, width = frame.shape[:2]
    scaled = resize_template(sprite, width / REF_W)
    h, w = scaled.shape[:2]
    x0 = int(round(u * width - w / 2))
    y0 = int(round(v * height - h / 2))
    x1, y1 = min(width, x0 + w), min(height, y0 + h)
    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
        return
    sx, sy = max(0, -x0), max(0, -y0)
    frame[max(0, y0) : y1, max(0, x0) : x1] = scaled[sy : sy + (y1 - max(0, y0)), sx : sx + (x1 - max(0, x0))]


def scene(width: int, ball=None, hoop=None, lives: int = 3) -> np.ndarray:
    """A frame at the given width (16:9) with the ball, hoop and life icons."""
    frame = blank(width, round(width * 9 / 16))
    for i in range(lives):
        draw(frame, LIFE, *LIFE_SPOTS[i])
    if hoop is not None:
        draw(frame, HOOP, *hoop)
    if ball is not None:
        draw(frame, BALL, *ball)
    return frame
