"""Script section 8: sampling the hoop's motion (Nyquist) and predicting where it will be."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from animations.base import *  # noqa: F401,F403,E402
from hoops_bot.tracker import HoopTracker  # noqa: E402


class NyquistSampling(ThemedScene):
    """A 4 s cycle is 0.25 Hz. Sampling every 0.25 s is 16x that, far above the Nyquist minimum."""

    def construct(self):
        t, cfg = self.theme, self.cfg
        period, dt = cfg.hoop_period, cfg.sample_interval
        freq, rate = 1 / period, 1 / dt
        span = 2 * period

        def signal(time):
            return math.sin(math.tau * time / period)

        title = self.title("Sampling the hoop")
        axes, decor = self.chart(
            [0, span, 1], [-1.4, 1.4, 1], 11, 3.8,
            x_ticks=list(range(0, int(span) + 1)), y_ticks=[-1, 0, 1],
        )
        chart = VGroup(decor, axes).move_to(DOWN * 0.7)
        titles = self.axis_titles(decor, "time (s)", "hoop x")
        curve = axes.plot(signal, x_range=[0, span], color=t.primary, stroke_width=t.stroke_width)

        times = np.arange(0, span + dt / 2, dt)
        stems = VGroup(*[Line(axes.c2p(s, 0), axes.c2p(s, signal(s)), color=t.secondary, stroke_width=2) for s in times])
        dots = VGroup(*[Dot(axes.c2p(s, signal(s)), radius=0.06, color=t.secondary) for s in times])

        minimum_times = [period / 4 + k * period / 2 for k in range(int(span / (period / 2)))]
        rings = VGroup(
            *[Circle(radius=0.15).set_fill(opacity=0).set_stroke(t.accent, 4).move_to(axes.c2p(s, signal(s))) for s in minimum_times]
        )

        signal_note = self.text(f"hoop cycle: {period:g} s  =  {freq:g} Hz", "small", t.primary, "BOLD")
        sample_note = self.text(f"our samples: every {dt:g} s  =  {rate:g} Hz", "small", t.secondary, "BOLD")
        minimum_note = self.text(f"Nyquist minimum: 2 per cycle (every {period / 2:g} s)", "small", t.accent, "BOLD")
        notes = VGroup(signal_note, sample_note, minimum_note).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        notes.next_to(title, DOWN, buff=0.3).to_edge(LEFT, buff=0.8)

        rule = self.formula(r"f_s \geq 2f", "sample rate ≥ 2 × signal frequency")
        numbers = self.formula(
            rf"{rate:g}\,\text{{Hz}} \geq 2 \times {freq:g}\,\text{{Hz}}",
            f"{rate:g} Hz ≥ 2 × {freq:g} Hz",
        )
        maths = VGroup(rule, numbers).arrange(DOWN, buff=0.25).to_corner(UR, buff=0.7).shift(DOWN * 1.1)
        ratio = self.text(f"{rate / freq:g}× the signal frequency", "small", t.secondary, "BOLD").next_to(maths, DOWN, buff=0.25)

        self.play(Write(title))
        self.play(FadeIn(decor), Create(axes), FadeIn(titles))
        self.play(Create(curve), FadeIn(signal_note), run_time=2)
        self.play(FadeIn(sample_note), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.04), run_time=2.5)
        self.play(FadeIn(stems, lag_ratio=0.05), run_time=1)
        self.wait(0.5)
        self.play(FadeIn(minimum_note), Create(rings))
        self.wait(0.8)
        self.play(FadeIn(rule, shift=UP * 0.2))
        self.play(FadeIn(numbers, shift=UP * 0.2), FadeIn(ratio))
        self.wait(2)


class HoopPrediction(ThemedScene):
    """Fit the last cycle of samples, then read off where the hoop will be when the ball arrives.
    The fit is computed by hoops_bot.tracker.HoopTracker, the same code the bot runs."""

    def construct(self):
        t, cfg = self.theme, self.cfg
        period, dt, lead = cfg.hoop_period, cfg.sample_interval, cfg.flight_time
        centre, amplitude, noise = 640.0, 180.0, 5.0
        rng = np.random.default_rng(7)

        def truth(time):
            return centre + amplitude * math.sin(math.tau * time / period + 0.6)

        tracker = HoopTracker(period, cfg.fit_tolerance_px)
        sample_times = np.arange(0, period + dt / 2, dt)
        sample_values = []
        for s in sample_times:
            value = truth(s) + rng.normal(0, noise)
            tracker.update(float(s), float(value))
            sample_values.append(value)
        now = float(sample_times[-1])
        horizon = now + 1.5

        title = self.title("Where will the hoop be?")
        axes, decor = self.chart(
            [0, horizon, 1], [420, 880, 115], 10.5, 3.9,
            x_ticks=list(range(0, int(horizon) + 1)), y_ticks=[460, 640, 820],
        )
        chart = VGroup(decor, axes).move_to(DOWN * 0.9)
        titles = self.axis_titles(decor, "time (s)", "hoop x (px)")

        dots = VGroup(*[Dot(axes.c2p(s, v), radius=0.07, color=t.secondary) for s, v in zip(sample_times, sample_values)])
        fit_past = axes.plot(tracker.predict, x_range=[0, now], color=t.primary, stroke_width=t.stroke_width)
        fit_future = DashedVMobject(
            axes.plot(tracker.predict, x_range=[now, horizon], color=t.primary, stroke_width=t.stroke_width),
            num_dashes=18,
            color=t.primary,
        )
        now_line = DashedLine(axes.c2p(now, 420), axes.c2p(now, 880), color=t.muted, stroke_width=3)
        now_label = self.text("now (shoot)", "small", t.muted).next_to(now_line, UP, buff=0.1)

        hit_time = now + lead
        hit_point = axes.c2p(hit_time, tracker.predict(hit_time))
        impact_line = DashedLine(axes.c2p(hit_time, 420), hit_point, color=t.accent, stroke_width=3)
        impact_dot = Dot(hit_point, radius=0.13, color=t.accent)
        impact_ring = Circle(radius=0.3).set_fill(opacity=0).set_stroke(t.accent, 4).move_to(hit_point)
        impact_label = self.text("hoop at impact", "small", t.accent, "BOLD").next_to(impact_ring, UP, buff=0.15)

        model = self.formula(
            r"x(t) = a + b\sin\omega t + c\cos\omega t",
            "x(t) = a + b sin ωt + c cos ωt",
        ).next_to(title, DOWN, buff=0.3)
        model_note = self.text(f"ω = 2π / {period:g} s is known, so a, b, c come from a simple least-squares fit", "small", t.muted)
        model_note.next_to(model, DOWN, buff=0.15)

        self.play(Write(title))
        self.play(FadeIn(decor), Create(axes), FadeIn(titles))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.05), run_time=2.5)
        self.play(FadeIn(model), FadeIn(model_note))
        self.play(Create(fit_past), run_time=2)
        self.play(Create(now_line), FadeIn(now_label))
        self.play(Create(fit_future), run_time=1.5)
        self.play(Create(impact_line), FadeIn(impact_dot, scale=0.5), Create(impact_ring), FadeIn(impact_label))
        self.play(FadeIn(self.caption(f"The AI gets the distance to this point, {lead:g} s ahead")))
        self.wait(2)
