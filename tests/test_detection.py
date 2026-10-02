import json
import tempfile
from pathlib import Path

import pytest
from synthetic import BALL, HOOP, LIFE, REF_W, scene

from hoops_bot.config import Config
from hoops_bot.detection import LivesCounter, TemplateDetector, _peak_offset, find_best_scale
from hoops_bot.geometry import Viewport

BALL_AT = (0.31, 0.62)
HOOP_AT = (0.84, 0.41)


def test_finds_ball_and_hoop_in_normalised_units():
    frame = scene(REF_W, ball=BALL_AT, hoop=HOOP_AT)
    ball = TemplateDetector(BALL, 0.9, REF_W).find(frame)
    hoop = TemplateDetector(HOOP, 0.9, REF_W).find(frame)
    assert abs(ball.u - BALL_AT[0]) < 0.002 and abs(ball.v - BALL_AT[1]) < 0.002
    assert abs(hoop.u - HOOP_AT[0]) < 0.002 and abs(hoop.v - HOOP_AT[1]) < 0.002
    assert ball.score > 0.99


@pytest.mark.parametrize("width", [640, 960, 1280, 1920])
def test_same_normalised_position_at_every_resolution(width):
    frame = scene(width, ball=BALL_AT, hoop=HOOP_AT)
    ball = TemplateDetector(BALL, 0.8, REF_W).find(frame)
    hoop = TemplateDetector(HOOP, 0.8, REF_W).find(frame)
    assert ball is not None and hoop is not None
    assert abs(ball.u - BALL_AT[0]) < 0.003 and abs(ball.v - BALL_AT[1]) < 0.003
    assert abs(hoop.u - HOOP_AT[0]) < 0.003 and abs(hoop.v - HOOP_AT[1]) < 0.003


def test_without_the_reference_width_a_resized_screen_breaks_matching():
    frame = scene(1920, ball=BALL_AT)
    det = TemplateDetector(BALL, 0.8, reference_width=None).find(frame)
    assert det is None or abs(det.u - BALL_AT[0]) > 0.003


def test_returns_none_when_absent():
    assert TemplateDetector(BALL, 0.9, REF_W).find(scene(REF_W, hoop=HOOP_AT)) is None


def test_missing_template_file_gives_helpful_error():
    with pytest.raises(FileNotFoundError, match="assets"):
        TemplateDetector("assets/does_not_exist.png")


def test_subpixel_peak_offset():
    f = lambda x: -((x - 0.3) ** 2)
    assert _peak_offset(f(-1), f(0), f(1)) == pytest.approx(0.3, abs=1e-9)
    assert _peak_offset(1.0, 1.0, 1.0) == 0.0


@pytest.mark.parametrize("width", [640, 960, 1920])
def test_lives_counted_at_every_resolution(width):
    for lives in (0, 1, 2, 3):
        counter = LivesCounter(LIFE, 0.8, reference_width=REF_W)
        assert counter.count(scene(width, lives=lives)) == lives


def test_lives_region_is_a_fraction_of_the_viewport():
    frame = scene(1280, lives=3)
    assert LivesCounter(LIFE, 0.8, region=(0.0, 0.0, 0.2, 0.15), reference_width=REF_W).count(frame) == 3
    assert LivesCounter(LIFE, 0.8, region=(0.5, 0.5, 0.5, 0.5), reference_width=REF_W).count(frame) == 0


def test_find_best_scale_recovers_the_reference_width():
    frame = scene(1280, ball=BALL_AT)
    scale, score = find_best_scale(frame, BALL)
    assert score > 0.9
    assert 1280 / scale == pytest.approx(REF_W, rel=0.03)


def test_viewport_maths():
    view = Viewport(1920, 1080)
    assert view.to_norm(960, 540) == (0.5, 0.5)
    assert view.to_px(0.25, 0.5) == (480, 540)
    assert view.dist(0.0, 0.1) == pytest.approx(0.1)
    assert view.dist(0.1, 0.0) == pytest.approx(0.1 * 16 / 9)
    assert view.template_scale(960) == 2.0 and view.template_scale(None) == 1.0


def test_config_fractions_load_and_pixel_values_are_rejected():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.json"
        path.write_text(json.dumps({"region": [0.1, 0.2, 0.5, 0.6], "lives_region": [10, 10, 300, 80]}))
        cfg = Config.load(path)
    assert cfg.region == (0.1, 0.2, 0.5, 0.6)
    assert cfg.lives_region is None  # pixel-looking values are ignored
