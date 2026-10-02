import tempfile

import cv2
import numpy as np
import pytest

from hoops_bot.score import ScoreReader, learn_digits

REGION = (0.70, 0.02, 0.28, 0.14)
ALL_DIGITS = "0123456789"


def score_frame(width: int, text: str, dark_on_light: bool = False) -> np.ndarray:
    height = round(width * 9 / 16)
    background, ink = ((230, 230, 230), (20, 20, 20)) if dark_on_light else ((30, 30, 30), (255, 255, 255))
    frame = np.full((height, width, 3), background, dtype=np.uint8)
    scale = width / 960
    cv2.putText(frame, text, (int(0.71 * width), int(0.10 * height)), cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * scale, ink, max(1, round(2 * scale)), cv2.LINE_AA)
    return frame


def learned_reader(dark_on_light: bool = False):
    folder = tempfile.mkdtemp()
    saved = learn_digits(score_frame(960, ALL_DIGITS, dark_on_light), REGION, ALL_DIGITS, folder)
    assert saved == list(ALL_DIGITS)
    return ScoreReader(folder, REGION, threshold=0.7)


@pytest.mark.parametrize("width", [640, 960, 1280, 1920])
def test_reads_the_score_at_every_resolution(width):
    reader = learned_reader()
    assert reader.missing == []
    for text in ("17", "40", "908", "5"):
        assert reader.read(score_frame(width, text)) == int(text)


def test_works_for_dark_digits_on_a_light_background():
    reader = learned_reader(dark_on_light=True)
    assert reader.read(score_frame(960, "26", dark_on_light=True)) == 26


def test_returns_none_when_there_are_no_digits():
    reader = learned_reader()
    assert reader.read(score_frame(960, "")) is None


def test_wrong_glyph_count_is_a_clear_error():
    with pytest.raises(ValueError, match="glyphs"):
        learn_digits(score_frame(960, "123"), REGION, "12", tempfile.mkdtemp())


def test_reader_without_templates_returns_none():
    assert ScoreReader(tempfile.mkdtemp(), REGION).read(score_frame(960, "7")) is None
