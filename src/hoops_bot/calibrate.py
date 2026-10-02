"""Setup helpers. Everything is saved as fractions of the screen or game area, never pixels.

    hoops-calibrate region    drag a box around the game area
    hoops-calibrate lives     drag a box around the lives display (optional)
    hoops-calibrate score     drag a box around the score
    hoops-calibrate digits    with the score on screen, type it in; learns the digit shapes
    hoops-calibrate scale     finds the size your crops were taken at (assets/reference.json)
    hoops-calibrate motion    watch passively and measure the player's motion periods
    hoops-calibrate preview   live view of what the bot sees
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from .capture import ScreenGrabber
from .config import Config
from .control import TrainingControl
from .detection import TemplateDetector, find_best_scale
from .hoop_model import SineTrack
from .score import learn_digits
from .session import GameSession


def _select(window: str, frame: np.ndarray) -> tuple[float, float, float, float] | None:
    """Drag a box; returns it as fractions of the frame."""
    x, y, w, h = cv2.selectROI(window, frame)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        return None
    fh, fw = frame.shape[:2]
    return (round(x / fw, 4), round(y / fh, 4), round(w / fw, 4), round(h / fh, 4))


def _viewport_frame(cfg: Config) -> np.ndarray:
    return ScreenGrabber(cfg.monitor, cfg.region).grab()


def pick_region() -> None:
    cfg = Config.load()
    with mss.mss() as sct:
        mon = sct.monitors[cfg.monitor]
        frame = cv2.cvtColor(np.asarray(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
    box = _select("Drag a box around the game area, then press Enter", frame)
    if box:
        cfg.region = box
        cfg.save()
        print(f"Saved region (fractions of the monitor): {box}")


def pick_inside(field: str, title: str) -> None:
    cfg = Config.load()
    box = _select(title + ", then press Enter", _viewport_frame(cfg))
    if box:
        setattr(cfg, field, box)
        cfg.save()
        print(f"Saved {field} (fractions of the game area): {box}")


def learn_score_digits() -> None:
    cfg = Config.load()
    if cfg.score_region is None:
        raise SystemExit("Run `hoops-calibrate score` first.")
    text = input("Type the score currently on screen (ideally something with many different digits): ").strip()
    saved = learn_digits(_viewport_frame(cfg), cfg.score_region, text, cfg.digits_dir)
    print(f"Saved digits {''.join(saved)} to {cfg.digits_dir}. Repeat with other scores to cover 0-9.")


def find_scale() -> None:
    """The crops in assets/ were taken at some screen size; find it by matching them to the live game."""
    cfg = Config.load()
    frame = _viewport_frame(cfg)
    width = frame.shape[1]
    scales = []
    for name in (cfg.ball_template, cfg.hoop_template):
        scale, score = find_best_scale(frame, cv2.imread(name))
        print(f"{name}: scale {scale:.3f} (match {score:.2f})")
        if score >= cfg.match_threshold:
            scales.append(scale)
    if not scales:
        raise SystemExit("Neither template matched. Make sure the ball and hoop are on screen.")
    reference = width / float(np.median(scales))
    Path(cfg.reference_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.reference_file).write_text(json.dumps({"reference_width": round(reference, 1)}))
    print(f"Crops were taken at a game width of about {reference:.0f}px. Saved to {cfg.reference_file}.")


def measure_motion(seconds: float = 20.0) -> None:
    """Watch the ball for a while (do not shoot) and find the player's motion periods."""
    cfg = Config.load()
    session, _ = build_session(cfg)
    tracks = {"player_v_period": [], "player_u_period": []}
    end = time.perf_counter() + seconds
    print(f"Watching for {seconds:.0f}s. Do not shoot. Play to a score where the player moves if you want the x period.")
    while time.perf_counter() < end:
        obs = session.sense()
        if obs.ball:
            tracks["player_v_period"].append((obs.t, obs.ball.v))
            tracks["player_u_period"].append((obs.t, obs.ball.u))
    for key, samples in tracks.items():
        track = SineTrack(None, cfg.track_tolerance, unknown_period_span=min(6.0, seconds / 2))
        for t, x in samples:
            track.update(t, x)
        moving = track.model is not None and track.model.amplitude > 0.004
        if moving and track.period:
            setattr(cfg, key, round(track.period, 3))
            print(f"{key}: {track.period:.3f}s (amplitude {track.model.amplitude:.3f})")
        else:
            print(f"{key}: not moving (or not enough data)")
    cfg.save()


def build_session(cfg: Config) -> tuple[GameSession, TrainingControl]:
    from .controller import GameController

    grabber = ScreenGrabber(cfg.monitor, cfg.region)
    control = TrainingControl(cfg.pause_hotkey, cfg.stop_hotkey)
    return GameSession(cfg, grabber, GameController(cfg), control=control), control


def preview() -> None:
    cfg = Config.load()
    session, _ = build_session(cfg)
    print("Boxes: green ball, blue hoop. Press q in the window to quit.")
    while True:
        obs = session.sense()
        frame = obs.frame.copy()
        h, w = frame.shape[:2]
        for det, colour in ((obs.ball, (0, 255, 0)), (obs.hoop, (255, 0, 0))):
            if det is not None:
                x, y = int(det.u * w), int(det.v * h)
                cv2.rectangle(frame, (x - int(det.w * w / 2), y - int(det.h * h / 2)),
                              (x + int(det.w * w / 2), y + int(det.h * h / 2)), colour, 2)
        lives = session.lives_counter.count(obs.frame)
        score = session.score_reader.read(obs.frame) if not session.score_reader.missing else None
        cv2.putText(frame, f"lives {lives}  score {score}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("hoops preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


COMMANDS = {
    "region": pick_region,
    "lives": lambda: pick_inside("lives_region", "Drag a box around the lives"),
    "score": lambda: pick_inside("score_region", "Drag a box around the score"),
    "digits": learn_score_digits,
    "scale": find_scale,
    "motion": measure_motion,
    "preview": preview,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    COMMANDS[parser.parse_args().command]()


if __name__ == "__main__":
    main()
