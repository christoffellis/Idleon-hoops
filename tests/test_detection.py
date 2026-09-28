import numpy as np
import pytest

from hoops_bot.detection import TemplateDetector


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
