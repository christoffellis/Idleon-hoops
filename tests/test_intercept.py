import math

import numpy as np
import pytest

from hoops_bot.arc_model import ArcModel
from hoops_bot.hoop_model import SineModel
from hoops_bot.intercept import choose_release, find_releases, predicted_miss
from hoops_bot.shots import analyse_shot

TWO_PI = 2 * math.pi
ARC = ArcModel(latency=0.07, d0=0.012, vx=0.9, e0=-0.012, vy=-0.6, g=2.2, latency_uncertain=False)
PLAYER_U = SineModel.constant(0.12)
PLAYER_V = SineModel(0.5, 0.10, 0.06, TWO_PI / 2.4)
HOOP_V = 0.68  # the arc lands about 0.17 below the player, so this is reachable


def brute_force_miss(t_press, hoop_u):
    """Independent check: march the ball forward in tiny steps until it reaches the hoop's u."""
    t_r = t_press + ARC.latency
    pu, pv = float(PLAYER_U(t_r)), float(PLAYER_V(t_r))
    tau = 0.0
    while pu + ARC.d0 + ARC.vx * tau < float(hoop_u(t_r + tau)):
        tau += 1e-5
    return float(ARC.ball_v(tau, pv)) - HOOP_V


@pytest.mark.parametrize("hoop", [SineModel.constant(0.82), SineModel(0.80, 0.05, 0.02, TWO_PI / 4.0)])
def test_every_release_found_really_produces_the_target_miss(hoop):
    releases = find_releases(ARC, 0.03, 0.02, PLAYER_U, PLAYER_V, hoop, HOOP_V, 1000.0, 1008.0)
    assert len(releases) >= 3
    for r in releases:
        assert brute_force_miss(r.t_press, hoop) == pytest.approx(0.03, abs=2e-4)


def test_timing_slack_is_the_band_width_divided_by_the_slope():
    hoop = SineModel.constant(0.82)
    r = find_releases(ARC, 0.0, 0.02, PLAYER_U, PLAYER_V, hoop, HOOP_V, 1000.0, 1003.0)[0]
    miss_early, _ = predicted_miss(ARC, r.t_press - 0.001, PLAYER_U, PLAYER_V, hoop, HOOP_V)
    miss_late, _ = predicted_miss(ARC, r.t_press + 0.001, PLAYER_U, PLAYER_V, hoop, HOOP_V)
    assert r.slope == pytest.approx(float(miss_late - miss_early) / 0.002, rel=1e-3)
    assert r.width == pytest.approx(0.02 / abs(r.slope), rel=1e-9)


def test_unreachable_hoop_gives_no_release():
    hoop = SineModel.constant(0.12)  # hoop is behind the player: the ball can never reach it
    assert find_releases(ARC, 0.0, 0.02, PLAYER_U, PLAYER_V, hoop, HOOP_V, 1000.0, 1004.0) == []


def test_choose_release_prefers_the_earliest_forgiving_one():
    from hoops_bot.intercept import Release

    releases = [Release(10.0, 0, 1, 0.010, 10.5), Release(11.0, 0, 1, 0.050, 11.5), Release(12.0, 0, 1, 0.048, 12.5)]
    assert choose_release(releases, earliest=0.0).t_press == 11.0     # first one within 70% of the widest
    assert choose_release(releases, earliest=11.5).t_press == 12.0    # cannot use one in the past
    assert choose_release(releases, earliest=13.0) is None


def test_analyse_shot_finds_the_signed_miss_and_contact_from_the_path():
    hoop = SineModel.constant(0.82)
    t_press = 1000.37
    t_r = t_press + ARC.latency
    rows = []
    for i in range(40):
        t = t_press + i / 60
        if t < t_r:
            u, v = float(PLAYER_U(t)), float(PLAYER_V(t))
        else:
            tau = t - t_r
            u = float(PLAYER_U(t_r)) + ARC.d0 + ARC.vx * tau
            v = float(PLAYER_V(t_r)) + ARC.e0 + ARC.vy * tau + 0.5 * ARC.g * tau**2
        rows.append([t, u, v])
    shot = {"t_press": t_press, "path": rows, "player": {"u": PLAYER_U.to_dict(), "v": PLAYER_V.to_dict()},
            "hoop": {"u": hoop.to_dict(), "v": HOOP_V}}
    out = analyse_shot(shot)
    assert out["contact"] is False
    assert out["signed_miss"] == pytest.approx(brute_force_miss(t_press, hoop), abs=3e-4)
