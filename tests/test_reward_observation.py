import numpy as np

from hoops_bot.config import Config
from hoops_bot.detection import Detection
from hoops_bot.observation import build_observation
from hoops_bot.reward import closest_approach, shot_reward


def test_closest_approach_uses_same_frame_pairs():
    ball = [(0, 0), (50, 50), (100, 100)]
    hoop = [(100, 100), (100, 100), (100, 130)]
    assert closest_approach(ball, hoop) == 30.0


def test_closest_approach_without_data_is_infinite():
    assert closest_approach([], []) == float("inf")


def test_hit_is_rewarded():
    reward, hit = shot_reward(10.0, Config())
    assert hit and reward == Config().hit_reward


def test_penalty_grows_with_miss_distance_and_saturates():
    cfg = Config()
    near, _ = shot_reward(cfg.hit_tolerance_px + 5, cfg)
    far, _ = shot_reward(cfg.miss_scale_px / 2, cfg)
    huge, _ = shot_reward(10_000, cfg)
    assert 0 > near > far > huge
    assert huge == -cfg.miss_penalty


def test_observation_is_two_relative_values():
    cfg = Config()
    player = Detection(x=100, y=400, score=1, w=10, h=10)
    hoop = Detection(x=900, y=250, score=1, w=10, h=10)
    obs = build_observation(player, hoop, predicted_hoop_x=700, cfg=cfg)
    assert obs.shape == (2,) and obs.dtype == np.float32
    assert obs[0] == np.float32((250 - 400) / cfg.norm_y)
    assert obs[1] == np.float32((700 - 100) / cfg.norm_x)
