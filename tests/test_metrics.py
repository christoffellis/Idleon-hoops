import tempfile
import time
from pathlib import Path

from hoops_bot.metrics import GameLog, ShotStats


def test_stats_track_shots_and_accuracy():
    stats = ShotStats()
    for hit in (True, False, True, True):
        stats.record(hit)
    assert (stats.shots, stats.hits, stats.accuracy) == (4, 3, 0.75)


def test_paused_time_is_not_counted():
    stats = ShotStats()
    stats.pause()
    time.sleep(0.2)
    frozen = stats.minutes
    stats.resume()
    assert frozen * 60 < 0.1
    assert stats.minutes * 60 < 0.15


def test_pause_and_resume_are_idempotent():
    stats = ShotStats()
    stats.resume()
    stats.pause()
    stats.pause()
    stats.resume()
    assert stats.minutes >= 0


def test_game_log_appends_and_numbers_games_across_sessions():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "runs" / "games.csv"
        assert GameLog(path).record(12, 30, 2.5) == 1
        assert GameLog(path).record(5, 9, 1.0) == 2
        rows = path.read_text().strip().splitlines()
        assert rows[0].startswith("game,") and len(rows) == 3
