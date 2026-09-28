"""Script sections 5 and 9: turning positions into relative inputs, and the moving player."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from animations.base import *  # noqa: F401,F403,E402


def make_field(theme, center, width=7.0, height=4.2):
    """A mini game screen. Returns (frame, at) where at(u, v) maps u, v in [-1, 1] onto it."""
    frame = RoundedRectangle(corner_radius=0.25, width=width, height=height)
    frame.set_fill(theme.surface, 1).set_stroke(theme.muted, 3).move_to(center)
    cx, cy = frame.get_center()[0], frame.get_center()[1]

    def at(u, v):
        return np.array([cx + u * width / 2 * 0.9, cy + v * height / 2 * 0.85, 0.0])

    return frame, at


def hoop_shape(theme, position):
    hoop = Rectangle(width=0.16, height=0.9).set_fill(theme.primary, 1).set_stroke(width=0)
    return hoop.move_to(position)


class RelativeInputs(ThemedScene):
    """Section 5: four positions become two differences."""

    def construct(self):
        t = self.theme
        title = self.title("Four inputs to track")
        frame, at = make_field(t, LEFT * 3.3 + DOWN * 0.5)
        player_pos, hoop_pos = at(-0.65, -0.45), at(0.6, 0.4)
        player = Dot(player_pos, radius=0.2, color=t.secondary)
        hoop = hoop_shape(t, hoop_pos)
        player_label = self.text("player", "small", t.secondary, "BOLD").next_to(player, DOWN, buff=0.2)
        hoop_label = self.text("hoop", "small", t.primary, "BOLD").next_to(hoop, UP, buff=0.2)

        chips = VGroup(
            self.chip("player x", t.secondary, 3.4),
            self.chip("player y", t.secondary, 3.4),
            self.chip("hoop x", t.primary, 3.4),
            self.chip("hoop y", t.primary, 3.4),
        ).arrange(DOWN, buff=0.3).move_to(RIGHT * 4.4 + DOWN * 0.5)

        self.play(Write(title))
        self.play(FadeIn(frame), FadeIn(player), FadeIn(hoop), FadeIn(player_label), FadeIn(hoop_label))
        self.play(LaggedStart(*[FadeIn(chip, shift=LEFT * 0.3) for chip in chips], lag_ratio=0.25))
        self.wait(0.8)

        corner = np.array([hoop_pos[0], player_pos[1], 0.0])
        dx_line = DashedLine(player_pos, corner, color=t.accent, stroke_width=4)
        dy_line = DashedLine(corner, hoop_pos, color=t.accent, stroke_width=4)
        dx_label = self.text("dx", "label", t.accent, "BOLD").next_to(dx_line, DOWN, buff=0.15)
        dy_label = self.text("dy", "label", t.accent, "BOLD").next_to(dy_line, LEFT, buff=0.15)
        self.play(Create(dx_line), Create(dy_line), FadeIn(dx_label), FadeIn(dy_label))

        results = VGroup(
            self.chip("dy = hoop y − player y", t.accent, 5.2),
            self.chip("dx = hoop x − player x", t.accent, 5.2),
        ).arrange(DOWN, buff=0.5).move_to(RIGHT * 4.4 + DOWN * 0.5)
        new_title = self.title("Four inputs, two numbers")
        self.play(ReplacementTransform(chips, results), Transform(title, new_title))
        self.play(FadeIn(self.caption("Relative values: the same lesson applies wherever the hoop spawns")))
        self.wait(1.5)


class MovingPlayer(ThemedScene):
    """Section 9: the player moves too, but only the hoop needs predicting."""

    def construct(self):
        t, cfg = self.theme, self.cfg
        period, lead = cfg.hoop_period, cfg.flight_time
        title = self.title("Phase 3: both move")
        frame, at = make_field(t, DOWN * 0.3, width=10.5, height=4.6)
        clock = ValueTracker(0)

        def hoop_at(time):
            return at(0.55 + 0.3 * math.sin(math.tau * time / period), 0.45)

        def player_now():
            time = clock.get_value()
            return at(-0.5 + 0.25 * math.sin(math.tau * time / 6.0), -0.1 + 0.55 * math.sin(math.tau * time / 2.4))

        def ghost_pos():
            return hoop_at(clock.get_value() + lead)

        hoop = always_redraw(lambda: hoop_shape(t, hoop_at(clock.get_value())))
        player = always_redraw(lambda: Dot(player_now(), radius=0.2, color=t.secondary))
        ghost = always_redraw(
            lambda: Rectangle(width=0.16, height=0.9).set_fill(opacity=0).set_stroke(t.accent, 3).move_to(ghost_pos())
        )

        def dx_line():
            p, g = player_now(), ghost_pos()
            return DashedLine(p, np.array([g[0], p[1], 0.0]), color=t.accent, stroke_width=4)

        def dx_label():
            p, g = player_now(), ghost_pos()
            return self.text("dx", "label", t.accent, "BOLD").move_to([(p[0] + g[0]) / 2, p[1] - 0.35, 0])

        legend = VGroup(
            VGroup(Dot(radius=0.1, color=t.secondary), self.text("player (now)", "small")).arrange(RIGHT, buff=0.2),
            VGroup(Rectangle(width=0.1, height=0.3).set_fill(t.primary, 1).set_stroke(width=0), self.text("hoop (now)", "small")).arrange(RIGHT, buff=0.2),
            VGroup(Rectangle(width=0.1, height=0.3).set_fill(opacity=0).set_stroke(t.accent, 2), self.text("hoop at impact", "small")).arrange(RIGHT, buff=0.2),
        ).arrange(RIGHT, buff=0.7).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(frame))
        self.play(FadeIn(legend))
        self.add(hoop, player, ghost, always_redraw(dx_line), always_redraw(dx_label))
        self.play(FadeIn(self.caption("The ball leaves from the player's position, so only the hoop needs predicting")))
        self.play(clock.animate.set_value(8), run_time=8, rate_func=linear)
        self.wait(1)
