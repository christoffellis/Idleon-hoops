"""Shared building blocks for the video illustrations.

Scenes subclass ThemedScene and get the theme (colours, font, sizes) plus a few helpers, so the
look of every illustration can be changed in one place (animations/themes/*.json).

Numbers such as the hoop period, sample interval and reward scale come from hoops_bot.config, so
the graphics always match what the bot actually does.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from manim import *  # noqa: F401,F403,E402

from animations.theme import Theme, load_theme  # noqa: E402
from hoops_bot.config import Config as HoopsConfig  # noqa: E402


class ThemedScene(Scene):
    def setup(self):
        super().setup()
        self.theme: Theme = load_theme()
        self.cfg = HoopsConfig.load()
        self.camera.background_color = self.theme.background

    # ---- text ----------------------------------------------------------
    def text(self, content: str, size: str = "label", color: str | None = None, weight: str = "NORMAL"):
        t = self.theme
        font_size = {"title": t.title_size, "label": t.label_size, "small": t.small_size}[size]
        return Text(content, font=t.font, font_size=font_size, color=color or t.text, weight=weight)

    def title(self, content: str):
        return self.text(content, "title", weight="BOLD").to_edge(UP, buff=0.5)

    def caption(self, content: str, color: str | None = None):
        return self.text(content, "small", color or self.theme.muted).to_edge(DOWN, buff=0.5)

    def formula(self, tex: str, plain: str, size: int | None = None, color: str | None = None):
        """LaTeX maths when LaTeX is installed, otherwise the plain-text version."""
        t = self.theme
        if shutil.which("latex"):
            return MathTex(tex, color=color or t.text, font_size=size or t.label_size + 8)
        return Text(plain, font=t.font, font_size=size or t.label_size, color=color or t.text)

    # ---- shapes --------------------------------------------------------
    def chip(self, content: str, color: str | None = None, width: float | None = None):
        """A rounded label with a coloured outline."""
        t = self.theme
        label = self.text(content, "label")
        box = RoundedRectangle(
            corner_radius=0.18,
            width=max(label.width + 0.6, width or 0),
            height=label.height + 0.45,
        )
        box.set_fill(t.surface, 1).set_stroke(color or t.muted, 3)
        label.move_to(box)
        return VGroup(box, label)

    # ---- charts --------------------------------------------------------
    def chart(self, x_range, y_range, width, height, x_ticks=(), y_ticks=(), x_fmt="{:g}", y_fmt="{:g}"):
        """Axes plus faint grid and hand-made tick labels (so no LaTeX is needed for numbers).
        Returns (axes, decor). Add decor first, then axes, then the plotted mobjects."""
        t = self.theme
        axes = Axes(
            x_range=list(x_range),
            y_range=list(y_range),
            x_length=width,
            y_length=height,
            axis_config={"color": t.muted, "stroke_width": 3, "include_tip": False, "include_ticks": False},
        )
        x0, x1 = x_range[0], x_range[1]
        y0, y1 = y_range[0], y_range[1]
        decor = VGroup()
        for x in x_ticks:
            decor.add(Line(axes.c2p(x, y0), axes.c2p(x, y1), color=t.grid, stroke_width=1.5))
        for y in y_ticks:
            decor.add(Line(axes.c2p(x0, y), axes.c2p(x1, y), color=t.grid, stroke_width=1.5))
        for x in x_ticks:
            decor.add(self.text(x_fmt.format(x), "small", t.muted).next_to(axes.c2p(x, y0), DOWN, buff=0.15))
        for y in y_ticks:
            decor.add(self.text(y_fmt.format(y), "small", t.muted).next_to(axes.c2p(x0, y), LEFT, buff=0.15))
        return axes, decor

    def axis_titles(self, decor, x_title: str, y_title: str):
        t = self.theme
        x_label = self.text(x_title, "small", t.muted).next_to(decor, DOWN, buff=0.2)
        y_label = self.text(y_title, "small", t.muted).rotate(PI / 2).next_to(decor, LEFT, buff=0.25)
        return VGroup(x_label, y_label)
