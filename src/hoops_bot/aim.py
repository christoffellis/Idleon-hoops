"""`hoops-aim`: play the game by measurement.

Every shot is logged to runs/shots.jsonl and the model is refitted from the whole log, so the same
command works from a cold start and keeps improving:

1. No arc fitted yet: take probe shots at varied phases of the player's motion (any outcome gives
   the arc and the latency).
2. Arc known, no clean hit yet: aim at probe values of the signed miss, stepping outward from the
   panel centre until a clean hit is found (contact hits cost no lives).
3. Clean hit found: aim at the centre of the clean band and keep scoring. Play continues after the
   trophy score unless --stop-at is given.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from .arc_model import ArcModel, fit_arc
from .config import Config
from .control import StopRequested
from .intercept import choose_release, find_releases
from .metrics import GameLog, ShotStats
from .session import GameSession, PlanInvalid
from .shots import load_shots
from .window import HitWindow

GOLDEN = 0.6180339887498949


@dataclass
class Plan:
    t_fire: float
    mode: str            # "probe" (no arc yet) or "aim"
    target: float | None  # signed miss aimed for


class AimLoop:
    def __init__(self, session: GameSession, cfg: Config, stop_at: int | None = None,
                 max_shots: int | None = None, log=print, game_log: GameLog | None = None, horizon: float = 8.0):
        self.s, self.cfg, self.stop_at, self.max_shots, self.log = session, cfg, stop_at, max_shots, log
        self.horizon = horizon
        self.shots = load_shots(cfg.shots_file)
        self.window = HitWindow(cfg.default_window, cfg.probe_step, cfg.aim_prior)
        for shot in self.shots:
            if shot.get("signed_miss") is not None and shot.get("outcome"):
                self.window.update(shot["signed_miss"], shot["outcome"])
        self.arc: ArcModel | None = None
        self._refit()
        self.stats, self.game = ShotStats(), ShotStats()
        self.game_log = game_log if game_log is not None else GameLog()
        self.probes_taken = 0
        self.trophy_announced = False

    # ---- model --------------------------------------------------------------------------
    def _refit(self) -> None:
        self.arc, notes = fit_arc(self.shots)
        self._notes = notes

    # ---- planning -----------------------------------------------------------------------
    def plan(self, now: float) -> Plan | None:
        s = self.s
        lead = max(0.2, 4 * s.frame_dt)
        if self.arc is None:
            period = s.player_v.period or 3.0
            phase = (GOLDEN * self.probes_taken) % 1.0
            return Plan(now + lead + phase * period, "probe", None)
        target = self.window.target()
        releases = find_releases(
            self.arc, target, self.window.width, s.player_u.predict, s.player_v.predict, s.hoop_u.predict,
            s.hoop_v, now + lead, now + lead + self.horizon,
        )
        best = choose_release(releases, now + lead)
        return None if best is None else Plan(best.t_press, "aim", target)

    def step(self) -> dict:
        """Plan and take one shot, replanning if the hoop respawns while waiting."""
        s = self.s
        started = s.clock()
        while True:
            s._control_point()
            s.wait_ready()
            plan = self.plan(s.clock())
            if plan is None:
                if s.clock() - started > self.cfg.max_wait:
                    self.log("No good release found; taking a probe shot to learn more.")
                    arc, self.arc = self.arc, None
                    plan = self.plan(s.clock())
                    self.arc = arc
                else:
                    s.idle(0.25)
                    continue
            try:
                shot = s.play_shot(plan.t_fire, plan.mode, plan.target)
            except PlanInvalid:
                continue
            if plan.mode == "probe":
                self.probes_taken += 1
            return shot

    # ---- the loop -----------------------------------------------------------------------
    def _record(self, shot: dict) -> None:
        self.shots.append(shot)
        if shot.get("signed_miss") is not None:
            self.window.update(shot["signed_miss"], shot["outcome"])
        self._refit()
        for stats in (self.stats, self.game):
            stats.record(shot["outcome"] != "miss")
        miss = shot.get("signed_miss")
        aim = shot.get("target_miss")
        self.log(
            f"shot {shot['id']:>3} {shot['outcome']:<7} "
            f"miss {'n/a' if miss is None else f'{miss:+.4f}'}"
            f"{'' if aim is None else f' (aimed {aim:+.4f})'}  score {shot['score_after']}  lives {shot['lives_after']}"
        )

    def run(self) -> None:
        s = self.s
        s.wait_ready()
        s.lives, s.score = s.read_lives(), s.read_score()
        while True:
            if self.max_shots is not None and self.stats.shots >= self.max_shots:
                break
            shot = self.step()
            self._record(shot)
            if (self.stop_at is not None and shot["score_after"] is not None and shot["score_after"] >= self.stop_at):
                self.log(f"Reached {self.stop_at} points.")
                break
            score = shot["score_after"]
            if not self.trophy_announced and score is not None and score >= self.cfg.trophy_score:
                self.trophy_announced = True
                self.log(f"Trophy score ({self.cfg.trophy_score}) reached. Carrying on.")
            if shot["lives_after"] is not None and shot["lives_after"] <= 0:
                self.game.pause()
                self.stats.pause()
                game = self.game_log.record(score or 0, self.game.shots, self.game.minutes)
                self.log(f"Game {game} over: {score} points from {self.game.shots} shots. Total: {self.stats.summary()}")
                s.wait_for_new_game()
                self.game = ShotStats()
                self.stats.resume()
        self.log(self.stats.summary())


def main() -> None:
    from .calibrate import build_session

    parser = argparse.ArgumentParser(description="Play Idleon hoops from a fitted model of the game.")
    parser.add_argument("--stop-at", type=int, default=None, help="stop once this score is reached (default: keep playing)")
    parser.add_argument("--max-shots", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load()
    session, control = build_session(cfg)
    print(f"Controls: {cfg.pause_hotkey} pause/resume, {cfg.stop_hotkey} stop (Ctrl+C also works).")
    print(f"Focus the Idleon window. Starting in {cfg.countdown}s...")
    time.sleep(cfg.countdown)
    control.start_hotkeys()
    loop = AimLoop(session, cfg, args.stop_at, args.max_shots)
    try:
        loop.run()
    except (StopRequested, KeyboardInterrupt):
        print(f"Stopped. {loop.stats.summary()}")
    finally:
        control.stop_hotkeys()


if __name__ == "__main__":
    main()
