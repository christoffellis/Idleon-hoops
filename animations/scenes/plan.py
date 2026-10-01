"""Script sections 3 and 4: the game plan and how the game escalates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from animations.base import *  # noqa: F401,F403,E402

# Scores where the rules change (10, 20) and the trophy score (40), from the script.
PHASE_THRESHOLDS = (10, 20, 40)


class GamePlan(ThemedScene):
    """Section 3: three steps, revealed one at a time."""

    def step_card(self, number, heading, sub, color):
        t = self.theme
        box = RoundedRectangle(corner_radius=0.25, width=3.9, height=3.3)
        box.set_fill(t.surface, 1).set_stroke(color, 4)
        badge = Circle(radius=0.4).set_fill(color, 1).set_stroke(width=0)
        badge_number = self.text(number, "label", t.background, "BOLD").move_to(badge)
        head = self.text(heading, "label", weight="BOLD")
        detail = self.text(sub, "small", t.muted)
        for item in (head, detail):
            if item.width > 3.4:
                item.scale_to_fit_width(3.4)
        content = VGroup(VGroup(badge, badge_number), head, detail).arrange(DOWN, buff=0.35)
        content.move_to(box)
        return VGroup(box, content)

    def construct(self):
        t = self.theme
        title = self.title("The game plan")
        steps = [
            ("1", "Break down the rules", "figure out the mechanics", t.primary),
            ("2", "Make it AI-readable", "turn the screen into numbers", t.secondary),
            ("3", "Test and iterate", "until it works", t.accent),
        ]
        cards = VGroup(*[self.step_card(*step) for step in steps]).arrange(RIGHT, buff=0.7)
        cards.next_to(title, DOWN, buff=1.0)
        arrows = [
            Arrow(cards[i].get_right(), cards[i + 1].get_left(), buff=0.08, color=t.muted, stroke_width=4, tip_length=0.2)
            for i in range(len(cards) - 1)
        ]

        self.play(Write(title))
        for i, card in enumerate(cards):
            animations = [FadeIn(card, shift=UP * 0.3)]
            if i > 0:
                animations.append(Create(arrows[i - 1]))
            self.play(*animations, run_time=0.8)
            self.wait(0.6)
        self.wait(1)


class GamePhases(ThemedScene):
    """Section 4 onwards: a score line showing where the rules change."""

    def construct(self):
        t = self.theme
        title = self.title("How the game escalates")
        left, right, y = -5.5, 5.5, -0.4
        top = PHASE_THRESHOLDS[-1]

        def x_at(score):
            return left + (right - left) * score / top

        bounds = (0, *PHASE_THRESHOLDS)
        colors = (t.secondary, t.accent, t.primary)
        descriptions = (
            "Phase 1\nplayer moves\nvertically",
            "Phase 2\nhoop moves\nhorizontally",
            "Phase 3\nplayer also moves\nhorizontally",
        )
        self.play(Write(title))
        for i in range(3):
            a, b = x_at(bounds[i]), x_at(bounds[i + 1])
            segment = Line([a, y, 0], [b, y, 0], color=colors[i], stroke_width=14)
            label = self.text(descriptions[i], "small", t.text).move_to([(a + b) / 2, y + 1.15, 0])
            self.play(Create(segment), FadeIn(label, shift=UP * 0.2), run_time=1.0)

        ticks = VGroup()
        for score in bounds:
            tick_label = str(score) if score != top else f"{score}  trophy"
            ticks.add(
                Line([x_at(score), y - 0.18, 0], [x_at(score), y + 0.18, 0], color=t.muted, stroke_width=3),
                self.text(tick_label, "small", t.muted).move_to([x_at(score), y - 0.55, 0]),
            )
        star = Star(n=5, outer_radius=0.32, color=t.accent).set_fill(t.accent, 1)
        star.move_to([x_at(top), y - 1.25, 0])
        score_label = self.text("score", "small", t.muted).move_to([left - 0.9, y, 0])
        self.play(FadeIn(ticks), FadeIn(score_label))
        self.play(FadeIn(star, scale=0.5))
        self.wait(1.5)
