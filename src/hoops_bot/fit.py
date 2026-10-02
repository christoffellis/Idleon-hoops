"""`hoops-fit`: fit the game's arc, latency and hit window from runs/shots.jsonl and report how sure we are."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arc_model import fit_arc
from .config import Config
from .shots import load_shots
from .window import HitWindow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="save an overlay of every fitted arc (needs matplotlib)")
    args = parser.parse_args()
    cfg = Config.load()
    shots = load_shots(cfg.shots_file)
    print(f"{len(shots)} shots in {cfg.shots_file}")
    arc, notes = fit_arc(shots)
    for note in notes:
        print("  skipped:", note)
    if arc is None:
        raise SystemExit("Not enough usable shots yet. Take a few more (hits or misses both count).")
    window = HitWindow(cfg.default_window, cfg.probe_step, cfg.aim_prior)
    for shot in shots:
        if shot.get("signed_miss") is not None and shot.get("outcome"):
            window.update(shot["signed_miss"], shot["outcome"])
    print(f"arc:     vx {arc.vx:.4f}  vy {arc.vy:.4f}  g {arc.g:.4f}  d0 {arc.d0:.4f}  e0 {arc.e0:.4f}")
    print(f"latency: {arc.latency * 1000:.0f} ms (range {arc.latency_low * 1000:.0f}-{arc.latency_high * 1000:.0f})")
    print(f"fit error: {arc.rms:.5f} (fraction of height) over {arc.n_shots} shots, {arc.n_points} points")
    if arc.latency_uncertain:
        print("  latency is not pinned down: take shots with the player at different heights.")
    band = window.band()
    print("clean-hit band (signed miss):", "not found yet" if band is None else f"{band[0]:+.4f} to {band[1]:+.4f}")
    good = arc.rms < 0.003 and not arc.latency_uncertain and band is not None
    print("READY: run hoops-aim." if good else "NOT READY: keep collecting shots (hoops-aim or hoops-record).")
    out = Path(cfg.physics_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arc": arc.__dict__, "band": band}, indent=2, default=float))
    if args.plot:
        _plot(shots, arc)


def _plot(shots, arc) -> None:
    import matplotlib.pyplot as plt

    for shot in shots:
        if shot.get("path"):
            _, u, v = zip(*[(p[0], p[1], p[2]) for p in shot["path"]])
            plt.plot(u, v, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("u (fraction of width)")
    plt.ylabel("v (fraction of height)")
    plt.savefig("runs/arcs.png", dpi=120)
    print("saved runs/arcs.png")


if __name__ == "__main__":
    main()
