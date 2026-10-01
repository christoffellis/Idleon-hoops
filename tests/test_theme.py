import json
import sys
import tempfile
from pathlib import Path

import pytest

from animations.render import SCENES, SCENES_DIR, build_command, parse_overrides
from animations.theme import COLOUR_FIELDS, Theme, builtin_themes, load_theme

ROOT = Path(__file__).resolve().parents[1]


def with_env(**values):
    """Tiny helper: set env vars, return a function that restores them."""
    import os

    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def restore():
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return restore


def test_default_theme_is_earthy():
    restore = with_env(HOOPS_THEME=None, HOOPS_THEME_OVERRIDES=None)
    try:
        theme = load_theme()
    finally:
        restore()
    assert theme.name == "earthy"
    assert theme.background == Theme().background


def test_every_builtin_theme_loads_and_is_complete():
    restore = with_env(HOOPS_THEME=None, HOOPS_THEME_OVERRIDES=None)
    try:
        assert {"earthy", "forest"} <= set(builtin_themes())
        for name in builtin_themes():
            theme = load_theme(name)
            assert theme.name == name
            for field in COLOUR_FIELDS:
                assert getattr(theme, field).startswith("#")
    finally:
        restore()


def test_custom_theme_file_overrides_only_what_it_sets():
    restore = with_env(HOOPS_THEME=None, HOOPS_THEME_OVERRIDES=None)
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mine.json"
            path.write_text(json.dumps({"name": "mine", "primary": "#112233", "title_size": 60}))
            theme = load_theme(str(path))
        assert (theme.primary, theme.title_size) == ("#112233", 60)
        assert theme.accent == Theme().accent
    finally:
        restore()


def test_env_var_selects_theme_and_overrides_apply():
    restore = with_env(HOOPS_THEME="forest", HOOPS_THEME_OVERRIDES=json.dumps({"accent": "#abcdef"}))
    try:
        theme = load_theme()
    finally:
        restore()
    assert theme.name == "forest"
    assert theme.accent == "#abcdef"


def test_bad_values_are_rejected_with_clear_messages():
    restore = with_env(HOOPS_THEME=None, HOOPS_THEME_OVERRIDES=json.dumps({"primary": "red"}))
    try:
        with pytest.raises(ValueError, match="primary"):
            load_theme("earthy")
    finally:
        restore()
    restore = with_env(HOOPS_THEME=None, HOOPS_THEME_OVERRIDES=json.dumps({"colour": "#ffffff"}))
    try:
        with pytest.raises(ValueError, match="Unknown theme keys"):
            load_theme("earthy")
    finally:
        restore()
    with pytest.raises(FileNotFoundError, match="Built-in themes"):
        load_theme("does-not-exist")


def test_every_scene_is_registered_with_a_real_file_and_class():
    for key, (filename, class_name, _) in SCENES.items():
        source = (SCENES_DIR / filename).read_text()
        assert f"class {class_name}(ThemedScene)" in source, key


def test_every_scene_file_compiles():
    import py_compile

    for path in [*SCENES_DIR.glob("*.py"), ROOT / "animations" / "base.py"]:
        py_compile.compile(str(path), doraise=True)


def test_build_command_targets_the_scene_file():
    command = build_command("nyquist", quality="m", transparent=True)
    assert command[:3] == [sys.executable, "-m", "manim"]
    assert "-qm" in command and "--transparent" in command
    assert command[-2:] == [str(SCENES_DIR / "hoop_motion.py"), "NyquistSampling"]


def test_parse_overrides():
    assert parse_overrides(["primary=#aa5533", "font=Inter"]) == {"primary": "#aa5533", "font": "Inter"}
    with pytest.raises(SystemExit):
        parse_overrides(["nonsense"])
