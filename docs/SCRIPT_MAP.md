# Script map

How each section of the video script (`docs/script.md`) lines up with the code.

| Script section | Where it lives |
| --- | --- |
| 1-3 Intro, what is Idleon, game plan | (on camera) |
| 4 Rules, phase 1 | `HoopsEnv` in `src/hoops_bot/env.py` (wait or shoot, lives-based hit and miss detection) |
| 5 Breaking down phase 1 (4 inputs to 2) | `src/hoops_bot/observation.py` |
| 6 PPO and the reward function | `src/hoops_bot/reward.py`, `src/hoops_bot/train.py` |
| 7 Finding the ball and hoop with OpenCV | `src/hoops_bot/detection.py`, `src/hoops_bot/capture.py`, `hoops-calibrate preview` |
| 8 Phase 2, moving hoop and Nyquist | `src/hoops_bot/tracker.py`, `scripts/nyquist_demo.py` |
| 9 Phase 3, moving player | Already handled: `observation.py` measures from the ball's live position |
| 10 Sign-off numbers (shots and minutes) | `src/hoops_bot/metrics.py`, printed by `hoops-play`; cooldowns and pauses are excluded from the minutes |

## Notes for the script

- **Nyquist wording.** The theorem says you need a sampling rate of at least *twice* the signal
  frequency, not half of it. The hoop cycles at 0.25 Hz, so the minimum is 0.5 Hz (a sample every 2 s).
  Sampling every 0.25 s is 4 Hz, which is 16x the signal frequency and 8x the minimum.
- **Phase 3 and time.** The observation is built from the ball's current position each step, so a
  moving player needs no extra handling. Only the hoop needs prediction, because it moves during
  the ball's flight.
- **Tune `flight_time`.** The tracker predicts the hoop position `flight_time` seconds ahead. Measure
  the real flight time once and set it in `config.json`.
