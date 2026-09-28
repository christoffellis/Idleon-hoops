"""Setup helpers.

    hoops-calibrate region    drag a box around the game area; saves it to config.json
    hoops-calibrate preview   live view with ball/hoop detections and the predicted hoop x
"""
from __future__ import annotations

import argparse
import time

import cv2
import mss
import numpy as np

from .capture import ScreenGrabber
from .config import Config
from .detection import TemplateDetector
from .tracker import HoopTracker


def pick_region() -> None:
    with mss.mss() as sct:
        shot = np.asarray(sct.grab(sct.monitors[1]))
    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    x, y, w, h = cv2.selectROI("Drag around the game area, then press Enter", frame)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("No region selected.")
        return
    cfg = Config.load()
    cfg.region = (int(x), int(y), int(w), int(h))
    cfg.save()
    print(f"Saved region {cfg.region} to config.json")


def preview() -> None:
    cfg = Config.load()
    grabber = ScreenGrabber(cfg.region)
    ball_det = TemplateDetector(cfg.ball_template, cfg.match_threshold)
    hoop_det = TemplateDetector(cfg.hoop_template, cfg.match_threshold)
    tracker = HoopTracker(cfg.hoop_period, cfg.fit_tolerance_px)
    last_sample = -1e9

    while True:
        frame = grabber.grab()
        ball, hoop = ball_det.find(frame), hoop_det.find(frame)
        now = time.monotonic()
        for det, colour, label in ((ball, (0, 200, 255), "ball"), (hoop, (0, 255, 0), "hoop")):
            if det:
                p1 = (int(det.x - det.w / 2), int(det.y - det.h / 2))
                p2 = (int(det.x + det.w / 2), int(det.y + det.h / 2))
                cv2.rectangle(frame, p1, p2, colour, 2)
                cv2.putText(frame, f"{label} {det.score:.2f}", (p1[0], p1[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        if hoop and now - last_sample >= cfg.sample_interval:
            tracker.update(now, hoop.x)
            last_sample = now
        if hoop:
            px = int(tracker.predict(now + cfg.flight_time))
            cv2.line(frame, (px, 0), (px, frame.shape[0]), (255, 0, 255), 1)
        cv2.imshow("hoops-calibrate preview (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["region", "preview"])
    args = parser.parse_args()
    pick_region() if args.command == "region" else preview()


if __name__ == "__main__":
    main()
