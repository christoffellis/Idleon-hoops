"""Script sections 6 and 7: the reward function and finding the ball with template matching."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from animations.base import *  # noqa: F401,F403,E402


class RewardFunction(ThemedScene):
    """Section 6: +1 for a score, a penalty that grows with the miss distance and then caps.
    Uses the same numbers as hoops_bot.reward (reward scale, penalty and hit reward)."""

    def construct(self):
        t, cfg = self.theme, self.cfg
        scale, penalty, hit_reward = cfg.miss_scale_px, cfg.miss_penalty, cfg.hit_reward
        x_max = scale * 4 / 3

        def reward(miss):
            return -penalty * min(miss / scale, 1.0)

        title = self.title("Rewarding the AI")
        x_ticks = [round(x_max * k / 4) for k in range(5)]
        axes, decor = self.chart(
            [0, x_max, x_max / 4], [-penalty * 1.25, hit_reward * 1.25, penalty], 10.5, 4.2,
            x_ticks=x_ticks, y_ticks=[-penalty, 0, hit_reward],
        )
        chart = VGroup(decor, axes).move_to(DOWN * 0.6)
        titles = self.axis_titles(decor, "how far the shot missed (px)", "reward")

        hit_dot = Dot(axes.c2p(0, hit_reward), radius=0.12, color=t.hit)
        hit_label = self.text("score: +1", "small", t.hit, "BOLD").next_to(hit_dot, RIGHT, buff=0.25)
        curve = axes.plot(reward, x_range=[0, x_max], color=t.miss, stroke_width=t.stroke_width, use_smoothing=False)
        cap_label = self.text("penalty caps out", "small", t.miss).next_to(axes.c2p(scale * 1.15, -penalty), UP, buff=0.2)

        miss = ValueTracker(0)
        marker = always_redraw(lambda: Dot(axes.c2p(miss.get_value(), reward(miss.get_value())), radius=0.11, color=t.accent))
        guide = always_redraw(
            lambda: DashedLine(
                axes.c2p(miss.get_value(), 0), axes.c2p(miss.get_value(), reward(miss.get_value())),
                color=t.accent, stroke_width=3,
            )
        )
        readout = always_redraw(
            lambda: self.text(
                f"missed by {miss.get_value():.0f} px  →  reward {reward(miss.get_value()):+.2f}", "label", t.text
            ).next_to(chart, UP, buff=0.3)
        )

        self.play(Write(title))
        self.play(FadeIn(decor), Create(axes), FadeIn(titles))
        self.play(FadeIn(hit_dot, scale=0.5), FadeIn(hit_label))
        self.play(Create(curve), run_time=1.5)
        self.add(guide, marker, readout)
        self.play(miss.animate.set_value(x_max), run_time=5, rate_func=linear)
        self.play(FadeIn(cap_label))
        self.play(FadeIn(self.caption("The further the miss, the bigger the punishment")))
        self.wait(1.5)


class DetectionDemo(ThemedScene):
    """Section 7: slide the ball's cropped image across the screen and watch the match score peak."""

    def construct(self):
        t, cfg = self.theme, self.cfg
        title = self.title("Finding the ball")
        frame = RoundedRectangle(corner_radius=0.2, width=8.8, height=3.0)
        frame.set_fill(t.surface, 1).set_stroke(t.muted, 3).move_to(UP * 0.9)
        left, width = frame.get_left()[0], frame.width
        mid_y = frame.get_center()[1]

        ball_u = 0.32
        ball = Circle(radius=0.28).set_fill(t.secondary, 1).set_stroke(width=0)
        ball.move_to([left + width * ball_u, mid_y - 0.15, 0])
        hoop = Rectangle(width=0.14, height=0.9).set_fill(t.primary, 1).set_stroke(width=0)
        hoop.move_to([left + width * 0.86, mid_y + 0.5, 0])
        clutter = VGroup(
            *[
                Rectangle(width=w, height=h).set_fill(t.grid, 1).set_stroke(width=0).move_to([left + width * u, mid_y + dy, 0])
                for u, dy, w, h in ((0.12, 0.7, 0.9, 0.35), (0.55, -0.7, 1.2, 0.3), (0.7, 0.6, 0.6, 0.5), (0.45, 0.65, 0.8, 0.25))
            ]
        )

        template = Square(side_length=0.8).set_fill(t.accent, 0.15).set_stroke(t.accent, 4)
        scan = ValueTracker(0.02)
        template.add_updater(lambda m: m.move_to([left + width * scan.get_value(), ball.get_center()[1], 0]))

        def score(x):
            return 0.12 + 0.88 * math.exp(-(((x - ball_u) / 0.045) ** 2))

        axes, decor = self.chart([0, 1, 0.25], [0, 1, 0.5], width, 1.7, y_ticks=[0, 0.5, 1])
        plot = VGroup(decor, axes).next_to(frame, DOWN, buff=0.7)
        plot.shift(RIGHT * (left - axes.c2p(0, 0)[0]))
        threshold = cfg.match_threshold
        threshold_line = DashedLine(axes.c2p(0, threshold), axes.c2p(1, threshold), color=t.miss, stroke_width=3)
        threshold_label = self.text("match threshold", "small", t.miss).next_to(threshold_line, UP, buff=0.1).align_to(threshold_line, RIGHT)
        score_label = self.text("match score", "small", t.muted).rotate(PI / 2).next_to(decor, LEFT, buff=0.25)
        curve = axes.plot(score, x_range=[0.02, 0.98], color=t.accent, stroke_width=t.stroke_width)
        follower = always_redraw(
            lambda: Dot(axes.c2p(scan.get_value(), score(scan.get_value())), radius=0.09, color=t.text)
        )

        self.play(Write(title), FadeIn(frame))
        self.play(FadeIn(clutter), FadeIn(hoop), FadeIn(ball))
        self.play(FadeIn(decor), Create(axes), FadeIn(score_label), Create(threshold_line), FadeIn(threshold_label))
        self.add(template, follower)
        self.play(scan.animate.set_value(0.98), Create(curve), run_time=5, rate_func=linear)
        template.clear_updaters()
        self.play(template.animate.move_to(ball), run_time=0.8)
        found = SurroundingRectangle(ball, color=t.secondary, buff=0.2, corner_radius=0.1)
        found_label = self.text("ball found: (x, y)", "small", t.secondary, "BOLD").next_to(found, UP, buff=0.15)
        self.play(Create(found), FadeIn(found_label))
        self.play(FadeIn(self.caption("OpenCV template matching, run every frame")))
        self.wait(1.5)
