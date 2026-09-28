import numpy as np
import pytest

from hoops_bot.detection import LivesCounter, TemplateDetector


def make_template(size=20, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_finds_template_centre():
    template = make_template()
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[100:120, 250:270] = template
    det = TemplateDetector(template).find(frame)
    assert det is not None
    assert (det.x, det.y) == (260.0, 110.0)
    assert det.score > 0.99


def test_returns_none_when_absent():
    template = make_template()
    frame = np.random.default_rng(5).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    assert TemplateDetector(template, threshold=0.9).find(frame) is None


def test_missing_template_file_gives_helpful_error():
    with pytest.raises(FileNotFoundError, match="assets"):
        TemplateDetector("assets/does_not_exist.png")


def draw_lives(n, icon, offset=(10, 10), spacing=20, shape=(100, 200, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    y, x = offset
    for i in range(n):
        frame[y : y + icon.shape[0], x + spacing * i : x + spacing * i + icon.shape[1]] = icon
    return frame


def test_lives_counter_counts_icons():
    icon = make_template(12, seed=3)
    counter = LivesCounter(icon, threshold=0.9)
    for lives in (0, 1, 2, 3):
        assert counter.count(draw_lives(lives, icon)) == lives


def test_lives_counter_only_looks_inside_its_region():
    icon = make_template(12, seed=3)
    frame = draw_lives(2, icon)
    frame[60:72, 150:162] = icon  # decoy outside the lives display
    assert LivesCounter(icon, 0.9).count(frame) == 3
    assert LivesCounter(icon, 0.9, region=(0, 0, 100, 40)).count(frame) == 2
