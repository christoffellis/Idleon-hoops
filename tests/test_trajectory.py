import math

import numpy as np
import pytest

from hoops_bot.trajectory import (
    fit_path,
    first_departure,
    impact_time,
    signed_miss,
    split_at_contact,
)

DT = 1 / 60


def arc(n=30, u0=0.1, vx=0.8, v0=0.5, vy=-0.9, g=2.4, noise=0.0, seed=0):
    t = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    u = u0 + vx * t + rng.normal(0, noise, n)
    v = v0 + vy * t + 0.5 * g * t**2 + rng.normal(0, noise, n)
    return t, u, v


def test_clean_arc_has_no_contact_even_with_detection_noise():
    t, u, v = arc(noise=0.0008)
    assert split_at_contact(t, u, v) == (30, False)


def test_bounce_is_found_within_a_frame_or_two():
    t, u, v = arc(n=24)
    k = 14  # the ball bounces back at frame 14
    t_after = t[k:] - t[k]
    u[k:] = u[k] - 0.3 * t_after
    v[k:] = v[k] + 0.6 * t_after + 0.5 * 2.4 * t_after**2
    n_clean, contact = split_at_contact(t, u, v)
    assert contact and k - 2 <= n_clean <= k


def test_signed_miss_for_a_static_hoop_matches_the_algebra():
    t, u, v = arc()
    fit = fit_path(t, u, v)
    assert fit.rms < 1e-9
    hoop_u, hoop_v = 0.8, 0.45
    miss, t_hit = signed_miss(fit, lambda x: hoop_u + 0 * np.asarray(x), hoop_v, t[0], t[-1])
    t_expect = (hoop_u - 0.1) / 0.8
    assert t_hit == pytest.approx(t_expect, abs=1e-6)
    assert miss == pytest.approx(0.5 - 0.9 * t_expect + 1.2 * t_expect**2 - hoop_v, abs=1e-6)


def test_moving_hoop_impact_is_where_the_ball_and_hoop_share_u():
    t, u, v = arc()
    fit = fit_path(t, u, v)
    hoop_u = lambda x: 0.75 + 0.06 * np.sin(2 * math.pi * np.asarray(x) / 4.0)
    t_hit = impact_time(fit, hoop_u, 0.0, 1.5)
    assert abs(float(fit.u(t_hit)) - float(hoop_u(t_hit))) < 1e-9


def test_no_impact_when_the_ball_never_reaches_the_hoop():
    t, u, v = arc(vx=0.05)
    fit = fit_path(t, u, v)
    assert signed_miss(fit, lambda x: 0.9 + 0 * np.asarray(x), 0.4, t[0], t[-1]) is None


def test_launch_is_the_first_frame_the_ball_leaves_the_player():
    t = np.arange(20) * DT
    player_u = lambda x: 0.1 + 0 * np.asarray(x)
    player_v = lambda x: 0.5 + 0.1 * np.sin(2 * math.pi * np.asarray(x) / 3.0)
    u = np.array([float(player_u(x)) for x in t])
    v = np.array([float(player_v(x)) for x in t])
    u[8:] += 0.04 * np.arange(1, 13)  # the ball starts moving away at frame 8
    assert first_departure(t, u, v, player_u, player_v, t_press=0.0) == 8
    assert first_departure(t[:8], u[:8], v[:8], player_u, player_v, t_press=0.0) is None
