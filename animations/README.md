# Video illustrations

Animated graphics for the video, drawn with [manim](https://www.manim.community/). Colours, font and
sizes come from a theme file, so the whole look can be changed without touching a scene.

## Setup

```bash
pip install -e ".[animations]"
```

Manim also needs FFmpeg. LaTeX is optional: with it, formulas render as proper maths, without it they
fall back to plain text.

## Rendering

Run from the repo root:

```bash
python -m animations.render --list                 # scenes, script sections and themes
python -m animations.render nyquist                # one scene, 1080p
python -m animations.render --all -q m             # everything, 720p (faster previews)
python -m animations.render reward --theme forest
python -m animations.render reward --transparent   # transparent background, to overlay in your editor
```

Output lands in `media/videos/` (git-ignored). `-q` is `l` 480p, `m` 720p, `h` 1080p, `p` 1440p, `k` 4K.
You can also call manim directly: `HOOPS_THEME=forest manim -qm animations/scenes/plan.py GamePlan`.

## Scenes

| Key | Script section | What it shows |
| --- | --- | --- |
| `plan` | 3 The game plan | Three steps revealed one by one |
| `phases` | 4 Rules | Score line: phase 1, phase 2 at 10, phase 3 at 20, trophy at 40 |
| `inputs` | 5 Breaking down phase 1 | Four positions become `dx` and `dy` |
| `reward` | 6 The AI | Reward against miss distance, with the penalty cap |
| `detection` | 7 Finding the ball | A template sliding across the screen, match score peaking at the ball |
| `nyquist` | 8 Phase 2 | The 0.25 Hz hoop, 4 Hz samples, the 2x minimum and `f_s >= 2f` |
| `prediction` | 8 Phase 2 | Sine fit over the last cycle, read off at impact time |
| `moving-player` | 9 Phase 3 | Player and hoop both moving; `dx` to the hoop's position at impact |

Numbers come from the bot's own settings (`hoops_bot.config`, including your `config.json`): the hoop
period, sample interval, flight time, reward scale and match threshold. The prediction scene runs the
real `HoopTracker`. If you retune the bot, re-render and the graphics follow.

## Themes

Two ship with the repo: `earthy` (warm paper background, terracotta, olive and ochre) and `forest`
(the same palette on a dark background). Pick one with `--theme` or the `HOOPS_THEME` environment variable.

To make your own, copy `themes/earthy.json` anywhere and edit it:

```bash
python -m animations.render --all --theme ./my-theme.json
```

Only the keys you include change; the rest fall back to the defaults. To tweak a value without a file:

```bash
python -m animations.render nyquist --set primary=#aa5533 --set title_size=52
```

| Key | Used for |
| --- | --- |
| `background`, `surface`, `grid` | Video background, cards and the mini game screen, chart grid |
| `text`, `muted` | Main text; axes and captions |
| `primary` | The hoop and the signal |
| `secondary` | The player and the samples |
| `accent` | Highlights and the value being measured |
| `hit`, `miss` | Reward chart colours |
| `font` | Any font installed on your machine (default `sans-serif`) |
| `stroke_width`, `title_size`, `label_size`, `small_size` | Line weight and text sizes |

Colours must be hex (`#RRGGBB`). A typo in a key or a colour gives a clear error rather than a silent default.

## Adding a scene

Subclass `ThemedScene` from `animations/base.py`, then register it in `SCENES` in `animations/render.py`.
Use `self.theme`, `self.cfg` and the helpers (`text`, `title`, `caption`, `chip`, `formula`, `chart`) so it
picks up the theme automatically.
