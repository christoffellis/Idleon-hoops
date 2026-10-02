# Rewrite plan: identify the game instead of learning it

Status: proposal. Nothing here is built yet. Module tags below are **keep**, **change**, **retire** and **new**.

## 1. Why rewrite

Learning on the live game cannot work with the budget the game allows.

- 3 misses, then a 15 minute cooldown. At about a 50% hit rate that is roughly 25 informative shots an hour.
  Value-based and policy-gradient RL want tens of thousands of samples, so the live game would take hundreds of hours.
- Exploration is the expensive part: an epsilon-greedy "shoot" at 20 Hz burns lives on random shots.
- But the game is simple and deterministic, and we know a lot already:
  - Outcomes are deterministic (confirmed by testing).
  - The hoop moves as a sine with a 4 s period after 10 points. The player gets a sine x movement after 20.
  - Player momentum does **not** affect the ball.
  - A hit with no rim or backboard contact scores double. Contact still scores, at the normal rate.
  - The score is readable on screen.

So we **identify** the game's few unknowns from a handful of shots and then aim exactly. Learning is replaced by
measurement.

## 2. The model (the pick)

A small set of parametric models, each fitted to the data that identifies it, with a Gaussian process as a safety net.
No neural network on the critical path.

| Piece | Model | Fitted by | Data needed |
| --- | --- | --- | --- |
| Ball arc | Projectile, relative to the release point: `x(t) = vx*t`, `y(t) = vy*t + g*t^2/2`, in normalised units | Linear least squares on each shot's tracked path, then pooled | 1 to 3 shots (misses work as well as hits) |
| Hoop | `x(t) = c + a*sin(wt) + b*cos(wt)`, `w = 2*pi/4 s`, `y` fixed per spawn | Linear least squares (the current tracker idea) | 3 frames |
| Player | Same sine model per axis once it moves. Vertical waveform to be confirmed (see section 8) | Same | 3 frames |
| Hit window | An interval on the **signed vertical miss** at the hoop: a clean band inside a wider contact band | Bisection, with the score change as the label (+2 clean, +1 contact, 0 miss) | About 10 to 15 shots |
| Input latency | One number: key press to first ball frame | Median over shots | 5 or more shots |
| Policy | **Analytic.** Solve the intercept, find the release time where the predicted miss sits at the window centre, press then | None | None |

Safety nets:

- If the pooled arc residuals are not at the level of a pixel, or fits from different shots disagree, add terms (drag) or
  fall back to a Gaussian process (Matern 5/2, one length scale per input, noise fixed near zero). It interpolates exactly,
  and the per-input length scales report which inputs matter.
- Bounces never need modelling. Only the pre-contact path is used for the signed miss. A contact hit just widens the window.

Why not DQN or PPO as the main tool: the structure is known, the data is tiny and the answer should be exact. RL stays
as an **optional** module (section 7, M6) trained inside a fitted surrogate of the game, for the video.

### The intercept solver

Release at time `t0` from player position `(px, py)`. Let `tau` be the flight time, found by fixed-point iteration (the ball
is much faster than the hoop, so it converges in a few steps):

```
tau  <-  (hoop_x(t0 + tau) - px) / vx
miss(t0) = py + arc_y(tau) - hoop_y        # signed, in normalised height units
```

Both the player and the hoop are deterministic functions of time, so we can scan the next few seconds of `t0`, pick the
release time where `miss` equals the window centre, and schedule the key press for that time (a precise wait for the last
few milliseconds), minus the measured latency. Because the motions are periodic, good release times recur, so the bot can
simply wait for the best one. This replaces the fixed `flight_time` constant in the current code, which is only right if
flight time never changes with distance.

## 3. Resolution independence

Everything stored or computed is in **normalised units**: `u = x / W`, `v = y / H`, where `W` and `H` are the size of the
**game viewport** (excluding any letterbox bars).

1. **Data and models.** Shots, arcs, windows and hoop fits are stored in `(u, v)`. Time is in seconds, which does not
   depend on resolution.
2. **Distances.** Horizontal and vertical normalisation differ, so a scalar distance is computed with an aspect
   correction: `d = sqrt((du * aspect)^2 + dv^2)` with `aspect = W / H`. The signed miss is purely vertical, so it needs none.
3. **Templates are the catch.** OpenCV template matching is scale-dependent, so normalised coordinates alone do not
   survive a resolution change: the ball, hoop and life crops stop matching. Fix:
   - `assets/reference.json` records the viewport size the crops were taken at.
   - At start-up, templates are resized by `current_W / reference_W`.
   - `hoops-calibrate scale` searches +/-15% on a live frame and stores the best scale.
   - Match positions are refined to sub-pixel with a parabola fit around the peak, so low resolutions stay usable.
4. **Capture region.** Stored as fractions of the chosen monitor plus the monitor index, so a resolution change on the same
   display keeps working. If the window moves, re-run `hoops-calibrate region` (about 10 seconds).
5. **Sub-regions** (lives, score) are fractions of the viewport.
6. **Aspect ratio** is stored with the physics file. A different aspect ratio can change how much of the world is visible,
   so the fit warns and asks for a quick refit (a few shots) instead of silently applying old numbers.
7. **Pixel settings to convert** (all in `config.py` today): `region`, `lives_region`, `fit_tolerance_px`, `norm_x`,
   `norm_y` and `miss_scale_px`.

## 4. Module plan

| Module (`src/hoops_bot/`) | Tag | What happens |
| --- | --- | --- |
| `geometry.py` | new | `Viewport` with `to_norm`, `to_px`, `aspect` and the template scale factor. Every other module uses it |
| `config.py` | change | Normalised units, sections per concern, fractions instead of pixels |
| `capture.py` | change | Region as monitor fractions. Frames carry a `perf_counter` timestamp taken at grab |
| `detection.py` | change | Scale-aware templates, sub-pixel centres. `LivesCounter` kept |
| `score.py` | new | Reads the score from digit templates inside the score region. Reports the change per shot |
| `trajectory.py` | new | Frame-timestamped ball path recorder, contact (velocity kink) detection, signed miss at the hoop |
| `hoop_model.py` | new | Sine fit with a known period for the hoop (and player), respawn detection, parameter readout. Replaces `tracker.py` |
| `tracker.py` | retire | Superseded by `hoop_model.py` |
| `arc_model.py` | new | Projectile fit, pooling, residual report, optional Gaussian process fallback |
| `window.py` | new | Clean and contact band edges, bisection that picks the next shot to take |
| `intercept.py` | new | Fixed-point intercept solver, release scheduling, latency compensation |
| `record.py` | new | `hoops-record`: plays with a probing or aiming rule and logs every shot |
| `fit.py` | new | `hoops-fit`: fits everything from the log, writes `models/physics.json` and a report with plots |
| `aim.py` | new | `hoops-aim`: plays using `physics.json` through all three phases |
| `surrogate.py` | new, optional | Gymnasium environment built from `physics.json`, for training PPO offline |
| `controller.py` | change | `pyautogui.PAUSE = 0` (by default it sleeps 0.1 s after each call, which would hide the start of the flight) and a timestamped key press |
| `calibrate.py` | change | Adds `score` and `scale`. `region` and `lives` store fractions |
| `env.py`, `train.py`, `play.py` | change | Demoted to the optional RL path, using the surrogate and score-based reward |
| `observation.py`, `reward.py` | change | Normalised units. The reward becomes points scored (2, 1, 0) minus a graded miss penalty |
| `control.py`, `metrics.py` | keep | Pause, stop, per-game log. Cooldown handling unchanged |

### Shot log: `runs/shots.jsonl`

One line per shot.

```json
{
  "id": 12, "phase": 1, "t_press": 1042.317, "latency_est": null,
  "viewport": {"w": 1920, "h": 1080},
  "release": {"u": 0.121, "v": 0.534},
  "hoop": {"u": 0.842, "v": 0.402, "fit": {"c": 0.842, "a": 0.0, "b": 0.0}},
  "path": [[0.000, 0.121, 0.534], [0.016, 0.130, 0.521]],
  "contact": {"detected": false, "t": null},
  "signed_miss_v": -0.018,
  "score_before": 12, "score_after": 14,
  "lives_before": 3, "lives_after": 3,
  "outcome": "clean"
}
```

## 5. Workflow

```
hoops-calibrate region|lives|score|scale   # once per setup
hoops-record --mode probe                  # 1 to 3 games of phase 1: arcs, window, latency
hoops-fit                                  # fit, overlay plot, go / no-go
hoops-aim                                  # phase 1 to 3, up to the trophy score
hoops-sim-train                            # optional: PPO inside the surrogate, for the video
```

`hoops-fit` prints a go / no-go: arc residual in pixel-equivalents, whether the shifted flights overlay, window width, and
latency spread. Aim mode refuses to start on a no-go.

## 6. Cost in your time

- **Arc:** solved by the first 3 shots of the first game, hit or miss.
- **Window:** bisection needs about 10 to 15 shots in total. Clean and contact hits cost no lives, so a game can run long
  once the bot is close. Expect 1 to 3 games (a few hours with cooldowns).
- **Phases 2 and 3:** no new learning, only the hoop and player sine fits, which are measured from frames. One check when
  you first reach 20 points confirms the model still overlays.

## 7. Milestones

| Milestone | Deliverable | Acceptance |
| --- | --- | --- |
| M0 | `geometry.py`, normalised config, scale-aware detection | Synthetic scene rendered at 960x540 and 1920x1080 gives the same normalised detections within 0.3% |
| M1 | `score.py`, `trajectory.py` | Reads a synthetic score series exactly. Contact detector flags a synthetic bounce and nothing else |
| M2 | `record.py`, shot log | Fake-game run writes valid `shots.jsonl` |
| M3 | `arc_model.py`, `window.py`, `fit.py` | Recovers the synthetic arc and window edges within 1 px-equivalent. Report renders |
| M4 | `intercept.py`, `aim.py` for phase 1 | 100% of aimed shots hit in the synthetic phase 1, including simulated latency jitter |
| M5 | `hoop_model.py` for phases 2 and 3 | Same, with a moving hoop, then moving player |
| M6 (optional) | `surrogate.py` and PPO | PPO inside the surrogate matches the analytic aimer within a few percent |
| M7 | Docs, `SCRIPT_MAP.md`, manim scenes for the arc overlay, window bisection and intercept | `animations/` renders the new scenes |

Real-game acceptance: phase 1 reaches 10 points with at most one lost life in the validation game, then each later phase
holds the same bar.

## 8. What I need from you

**Blocking (nothing real can be validated until I have these):**

1. **Reference crops at one resolution:** `ball.png`, `hoop_panel.png`, `life.png`, plus a screenshot of the whole game
   viewport at the same size. Tell me that viewport size (for example 1920x1080). The crops are not in this session.
2. **Score display:** a screenshot of the score area at a few different values, and whether its font is a bitmap font
   (the same shape every time). If so, digit templates are trivial.
3. **Movement facts** (short answers are fine):
   - Does the player's vertical motion follow a sine? What is its period?
   - After each score, does the hoop respawn with a new random position, and is the sine amplitude fixed?
   - Does anything else change at 10, 20, 30 or 40 (speeds, amplitudes)?
   - Is the trophy at 40 *points*, so that 20 clean hits would do it?
4. **A recorded session** once M2 is built: 1 to 3 games of phase 1 with `hoops-record`, then push `runs/shots.jsonl`
   (or paste it). This is the only data the fit needs.

**Needed to finish, but not blocking:**

5. **Setup facts:** OS, windowed or fullscreen, one or several monitors, and whether the game letterboxes when you resize
   the window. These decide hotkey and capture details.
6. **Game-over and cooldown screen:** what the bot should see to know a new game is ready. I currently assume a full set of lives.
7. **DQN experiment:** I treat it as retired. If you want it kept for reference, push it to a branch such as `dqn-experiment`.

**Decisions for you:**

8. Keep PPO as a video segment (trained in the surrogate, compared with the analytic aimer), or drop it and tell the
   "wrong tool for the right job" story differently? I can build either.
9. Merge each milestone to `main` as it lands, or keep everything on a long-lived rewrite branch until M4 works?

## 9. Risks

- **Capture frame rate.** If the screen grab gives fewer than about 8 frames per flight, one shot gives a coarse arc.
  Mitigation: pool several shots, aligned on the release point.
- **Clock offset** between the screen grab and the game's frame. A constant offset is absorbed by the latency number. A
  jittery one shows up as spread in the latency estimate, and `hoops-fit` reports that.
- **Aspect-ratio change.** Handled by the warning and refit above, but it is a real source of silent error if ignored.
- **Score reading** depends on the font. Fallback: infer the outcome from the contact flag and lives alone.
- **Window edges are fuzzy** where the rim is touched. Deterministic, but the band is found empirically, not derived.


## Status and decisions (supersedes anything above that disagrees)

- **Regression only.** No RL, no surrogate PPO, no gymnasium. The whole model is a few least-squares fits.
- **Scripts and animations are untouched.**
- **Hit window learned online** from every shot, not a separate bisection phase (contact hits are free probes).
- **Latency** is fitted from shots at different player phases; `hoops-fit` reports if it is still uncertain.
- **Periods**: hoop 4 s; player periods are found by a period scan (fit a sine at each candidate period, keep the best), from `hoops-calibrate motion` or live.
- **Hoop respawn** box is not needed: each spawn is fitted live, and a jump marks a respawn.
- **Reference size** is found by `hoops-calibrate scale`.
- **Play continues** past 40 points; `--stop-at` is optional.
- **Long-lived branch**: `plan/physics-rewrite` until phase 1 works on the real game.
- Built and tested against a simulated game (`tests/synthetic_game.py`); real-game assumptions are listed in the PR notes.
