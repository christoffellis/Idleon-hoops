# idleon-hoops-bot

Train a PPO agent to beat the hoop-shooting minigame in [Legends of Idleon](https://www.legendsofidleon.com/),
using OpenCV to read the screen. Companion code for the video (script in [`docs/script.md`](docs/script.md),
section-by-section map in [`docs/SCRIPT_MAP.md`](docs/SCRIPT_MAP.md)).

## How it works

```
screen -> capture -> detect ball + hoop -> track hoop motion -> [dy, dx] -> PPO -> wait / shoot
```

1. **Capture** the game area with `mss`.
2. **Detect** the ball and the hoop's back panel with OpenCV template matching.
3. **Track** the hoop. It cycles horizontally every 4 s, so a fixed-frequency sine fit over samples taken
   every 0.25 s predicts where it will be when the ball arrives.
4. **Observe** two relative numbers: vertical gap to the hoop, and horizontal gap to the *predicted* hoop position.
5. **Act.** The agent chooses to wait or shoot. Reward is +1 for a score, and a penalty that grows with how far
   a shot missed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

1. Save two cropped screenshots as `assets/ball.png` and `assets/hoop_panel.png` (see `assets/README.md`).
2. Set the capture region: `hoops-calibrate region`.
3. Check detection: `hoops-calibrate preview` (boxes on the ball and hoop, magenta line at the predicted hoop x).

## Usage

```bash
hoops-train --steps 20000     # trains against the live game, saves models/hoops_ppo.zip
hoops-play                    # plays until the trophy score, prints shots and minutes
```

Focus the Idleon window during the countdown. Tuning lives in `src/hoops_bot/config.py`; put overrides in
`config.json` (git-ignored). Start with `flight_time`, `hit_tolerance_px` and `shoot_key`.

## Assumptions to verify

The game's exact controls were inferred from the script, so check these first:

- The player moves up and down on its own and a key press throws the ball. If the controls differ,
  adapt `GameController` and the action space in `env.py`.
- A miss ends the run (`terminate_on_miss`). If it does not, set that to `false`.
- A shot counts as a score when the ball's closest approach to the hoop centre is within `hit_tolerance_px`.
  Tune it against what the game actually awards.

## Development

```bash
pytest                          # detection, tracker, reward and observation tests (no game needed)
python scripts/nyquist_demo.py  # regenerates assets/nyquist.png for the video
```

## Layout

```
src/hoops_bot/
  config.py       settings
  capture.py      screen grabbing
  detection.py    template matching
  tracker.py      hoop motion prediction
  observation.py  detections -> [dy, dx]
  reward.py       reward shaping
  env.py          Gymnasium environment
  train.py        PPO training
  play.py         run a trained model
  calibrate.py    region picker and live preview
  metrics.py      shot and time counters
```
