import math

import numpy as np
import pytest

from hoops_bot.arc_model import fit_arc
from hoops_bot.hoop_model import SineModel, SineTrack, fit_sine, scan_period
from hoops_bot.window import CLEAN, CONTACT, MISS, HitWindow, classify_outcome

DT = 1 / 60
TWO_PI = 2 * math.pi


# ---- sine fitting -----------------------------------------------------------------------
def test_fit_sine_recovers_parameters_from_unevenly_timed_samples():
    rng = np.random.default_rng(1)
    t = 1000 + np.sort(rng.uniform(0, 3, 40))
    truth = SineModel(0.8, 0.05, -0.03, TWO_PI / 4)
    model, rms = fit_sine(t, truth(t), TWO_PI / 4)
    assert rms < 1e-9
    for name in ("c", "a", "b"):
        assert getattr(model, name) == pytest.approx(getattr(truth, name), abs=1e-9)


def test_scan_period_finds_an_unknown_period():
    rng = np.random.default_rng(2)
    t = 50 + np.sort(rng.uniform(0, 9, 300))
    x = SineModel(0.5, 0.1, 0.05, TWO_PI / 3.3)(t) + rng.normal(0, 0.0005, t.size)
    period, rms = scan_period(t, x)
    assert period == pytest.approx(3.3, rel=0.005)
    assert rms < 0.001


def test_scan_period_asks_for_more_data_when_the_recording_is_short():
    t = np.linspace(0, 0.5, 20)
    with pytest.raises(ValueError, match="longer recording"):
        scan_period(t, np.sin(t))


# ---- live tracking ----------------------------------------------------------------------
def feed(track, model, start, seconds, noise=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    flagged = []
    for t in np.arange(start, start + seconds, DT):
        flagged.append(track.update(float(t), float(model(t) + rng.normal(0, noise))))
    return start + seconds, flagged


def test_track_with_known_period_locks_quickly_and_predicts_ahead():
    hoop = SineModel(0.8, 0.05, 0.02, TWO_PI / 4)
    track = SineTrack(period=4.0)
    now, _ = feed(track, hoop, 500.0, 2.5)  # needs half a cycle: a sliver extrapolates badly
    assert track.ready
    assert abs(float(track.predict(now + 0.5)) - float(hoop(now + 0.5))) < 0.002


def test_track_notices_a_respawn_and_relocks():
    first = SineModel(0.7, 0.05, 0.0, TWO_PI / 4)
    second = SineModel(0.9, 0.05, 0.0, TWO_PI / 4)
    track = SineTrack(period=4.0)
    now, _ = feed(track, first, 100.0, 4.0)
    now, flagged = feed(track, second, now, 3.0, seed=1)
    assert any(flagged) and track.resets == 1
    assert track.ready
    assert abs(float(track.predict(now + 0.4)) - float(second(now + 0.4))) < 0.003


def test_static_hoop_is_a_constant_model():
    track = SineTrack(period=4.0)
    now, _ = feed(track, SineModel.constant(0.84), 10.0, 3.0)
    assert track.ready and track.model.amplitude < 1e-9
    assert float(track.predict(now + 1)) == pytest.approx(0.84, abs=0.001)


def test_unknown_period_is_estimated_once_there_is_enough_data():
    player = SineModel(0.5, 0.08, 0.04, TWO_PI / 2.6)
    track = SineTrack(period=None)
    now, _ = feed(track, player, 20.0, 4.0)
    assert not track.ready and track.period is None
    now, _ = feed(track, player, now, 5.0)
    assert track.ready
    assert track.period == pytest.approx(2.6, rel=0.01)


# ---- hit window -------------------------------------------------------------------------
def test_outcome_comes_from_the_score_with_lives_and_contact_as_a_fallback():
    assert classify_outcome(2, 0, False) == CLEAN
    assert classify_outcome(1, 0, True) == CONTACT
    assert classify_outcome(0, -1, False) == MISS
    assert classify_outcome(None, -1, False) == MISS
    assert classify_outcome(None, 0, True) == CONTACT
    assert classify_outcome(None, 0, False) == CLEAN


def test_probes_spiral_outward_from_the_prior_and_skip_tried_values():
    window = HitWindow(probe_step=0.02, prior=0.0)
    seen = []
    for _ in range(5):
        aim = window.next_probe()
        seen.append(round(aim, 3))
        window.update(aim, MISS)
    assert seen == [0.0, 0.02, -0.02, 0.04, -0.04]


def test_band_is_tightened_by_the_nearest_misses_and_target_is_its_centre():
    window = HitWindow(default_width=0.02)
    assert window.band() is None
    window.update(0.010, CLEAN)
    assert window.band() == pytest.approx((0.0, 0.02))  # default width until neighbours are known
    window.update(0.020, CLEAN)
    window.update(-0.030, MISS)
    window.update(0.060, CONTACT)
    low, high = window.band()
    assert (low, high) == pytest.approx((-0.01, 0.04))
    assert window.target() == pytest.approx(0.015)


# ---- arc model --------------------------------------------------------------------------
TRUE = dict(latency=0.07, d0=0.012, vx=0.9, e0=-0.012, vy=-0.6, g=2.2)
PLAYER_U = SineModel.constant(0.12)
PLAYER_V = SineModel(0.5, 0.10, 0.06, TWO_PI / 2.4)


def make_shot(shot_id, t_press, bounce_after=None, still=False, noise=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    t_r = t_press + TRUE["latency"]
    rows = []
    for i in range(45):
        t = t_press + i * DT
        if t < t_r or still:
            u, v = float(PLAYER_U(t)), float(PLAYER_V(t))
        else:
            tau = t - t_r
            if bounce_after is not None and tau > bounce_after:
                tb = bounce_after
                tau_b = tau - tb
                u0 = float(PLAYER_U(t_r)) + TRUE["d0"] + TRUE["vx"] * tb
                v0 = float(PLAYER_V(t_r)) + TRUE["e0"] + TRUE["vy"] * tb + 0.5 * TRUE["g"] * tb**2
                u = u0 - 0.4 * tau_b
                v = v0 + 0.7 * tau_b + 0.5 * TRUE["g"] * tau_b**2
            else:
                u = float(PLAYER_U(t_r)) + TRUE["d0"] + TRUE["vx"] * tau
                v = float(PLAYER_V(t_r)) + TRUE["e0"] + TRUE["vy"] * tau + 0.5 * TRUE["g"] * tau**2
        rows.append([t, u + rng.normal(0, noise), v + rng.normal(0, noise)])
    return {"id": shot_id, "t_press": t_press, "path": rows,
            "player": {"u": PLAYER_U.to_dict(), "v": PLAYER_V.to_dict()}}


def varied_shots():
    return [make_shot(i, 1000.0 + off, bounce_after=0.33 if i == 2 else None, seed=i)
            for i, off in enumerate([0.0, 0.5, 1.1, 1.9, 2.4])]


def test_arc_and_latency_are_recovered_from_a_handful_of_shots():
    model, notes = fit_arc(varied_shots())
    assert model.n_shots == 5 and not model.latency_uncertain
    assert model.latency == pytest.approx(TRUE["latency"], abs=0.004)
    assert model.vx == pytest.approx(TRUE["vx"], abs=0.01)
    assert model.vy == pytest.approx(TRUE["vy"], abs=0.03)
    assert model.g == pytest.approx(TRUE["g"], abs=0.1)
    assert model.rms < 0.0008


def test_one_shot_cannot_pin_down_latency_and_says_so():
    model, notes = fit_arc(varied_shots()[:1])
    assert model.latency_uncertain
    assert any("latency" in n for n in notes)


def test_shots_that_never_left_the_player_are_skipped_with_a_reason():
    shots = varied_shots() + [make_shot(99, 2000.0, still=True)]
    model, notes = fit_arc(shots)
    assert model.n_shots == 5
    assert any("shot 99 skipped" in n and "never left" in n for n in notes)
