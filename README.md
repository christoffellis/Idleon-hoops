# idleon-hoops

A bot that beats the Idleon hoops minigame by **measuring the game's physics** instead of learning
by trial and error. The game is deterministic: the ball's arc never changes, the hoop and player move
as sines, and a clean hit is a narrow band of heights at the hoop. So the bot fits those few numbers
with least squares from a handful of shots, then times each press so the ball lands in the band.

Everything is stored as a **fraction of the screen or game area**, so it works at any resolution.

## Setup

```bash
pip install -e .
```

1. Put your crops in `assets/` (see `assets/README.md`): `ball.png`, `hoop_panel.png`, `life.png`.
2. Open the game, then calibrate:

```bash
hoops-calibrate region    # box around the game area
hoops-calibrate lives     # box around the lives
hoops-calibrate score     # box around the score
hoops-calibrate digits    # type the score on screen; repeat at other scores until 0-9 are covered
hoops-calibrate scale     # works out the size your crops were taken at
hoops-calibrate preview   # check that ball and hoop are boxed and lives/score read correctly
hoops-calibrate motion    # (optional) measure the player's motion periods; saved to config.json
```

## Phase 1: learn the physics

```bash
hoops-aim        # or hoops-record to play by hand while the bot logs
hoops-fit        # prints the arc, latency, hit window, and whether it is READY
```

`hoops-aim` works from a cold start: it takes probe shots, fits the arc from whatever happens (hits or
misses), searches for the clean-hit band (contact hits cost no lives), then aims at the middle of it.
Every shot is appended to `runs/shots.jsonl` and the model is refitted after each one, so progress is
kept between games and cooldowns. F8 pauses, F9 stops. Play continues past the trophy score unless
you pass `--stop-at`.

## Layout

| Module | Job |
| --- | --- |
| `geometry`, `capture`, `detection`, `score` | normalised coordinates, screen grab, template matching, lives and score |
| `trajectory`, `arc_model`, `window` | contact detection, signed miss, arc and latency fit, hit-band learning |
| `hoop_model`, `intercept` | live sine tracking, solving for the press time |
| `session`, `aim`, `record`, `fit`, `calibrate` | the loop and the commands |

Tests run against a simulated game with a fake clock: `pytest`.
Illustrations for the video live in `animations/` (separate extra: `pip install -e .[animations]`).
