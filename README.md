# idleon-hoops

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
6. **Judge.** The lives display decides the outcome: a lost life is a miss, an unchanged count is a hit. Losing the
   third life ends the game.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

1. Save three cropped screenshots as `assets/ball.png`, `assets/hoop_panel.png` and `assets/life.png` (see `assets/README.md`).
2. Set the capture region: `hoops-calibrate region`.
3. Optional, but steadier: `hoops-calibrate lives` to box the lives display.
4. Check detection: `hoops-calibrate preview` (boxes on the ball and hoop, magenta line at the predicted hoop x,
   and the lives count, which should read 3 at the start of a game).

## Usage

```bash
hoops-train                   # trains against the live game, one game at a time
hoops-play                    # plays until the trophy score, prints shots and minutes
```

Focus the Idleon window during the countdown.

### Training across games and cooldowns

The game has a cooldown after the third lost life, so training is built around games:

- **After every game** the model is saved to `models/latest.zip` (plus a numbered copy in `models/games/` every
  5 games), and a row goes to `runs/games.csv` with the score, shots and minutes.
- **Between games** the bot waits. Start the next game in Idleon when the cooldown ends and training resumes by
  itself as soon as it sees a fresh game (ball, hoop and full lives).
- **Manual controls:** `F8` pauses and resumes (it holds after the current shot), `F9` saves and stops. `Ctrl+C` in
  the terminal saves and stops too. Hotkeys work while the game window has focus. On macOS, allow the terminal
  under Accessibility and Input Monitoring.
- **Coming back later:** run `hoops-train` again. It resumes from `models/latest.zip`; add `--fresh` to start over.
- Cooldowns and pauses are not counted in the minutes reported for the video. Tuning lives in `src/hoops_bot/config.py`; put overrides in
`config.json` (git-ignored). Start with `flight_time`, `settle_time` and `shoot_key`.

## Assumptions to verify

The game's exact controls were inferred from the script, so check these first:

- The player moves up and down on its own and a key press throws the ball. If the controls differ,
  adapt `GameController` and the action space in `env.py`.
- The lives display shows one icon per remaining life, and the icon disappears (or changes to something that does
  not match `life.png`) when a life is lost.
- The lives display updates within `settle_time` (0.5 s) of a shot. Raise it if misses are being read as hits.

## Development

```bash
pytest                          # detection, tracker, reward, controls and a fake-game environment test
python scripts/nyquist_demo.py  # regenerates assets/nyquist.png for the video
```

## Layout

```
src/hoops_bot/
  config.py       settings
  capture.py      screen grabbing
  detection.py    template matching, lives counter
  tracker.py      hoop motion prediction
  observation.py  detections -> [dy, dx]
  reward.py       reward shaping
  env.py          Gymnasium environment
  train.py        PPO training
  play.py         run a trained model
  calibrate.py    region picker and live preview
  control.py      pause / resume / stop hotkeys
  metrics.py      shot, time and per-game counters
```
