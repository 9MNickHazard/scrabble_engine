"""Helpers for the app tests of training from saved games (test_app_gamesets.py,
test_app_games_runs.py): game set folders made for real with `games.generate`, or by hand."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scrabble_ai import engine as engine_module
from scrabble_ai import games as games_module

SET_KEYS = {"name", "status", "games", "target", "progress", "bytes", "lexicon_id", "leaves_hash", "opening_plies",
            "opening_temperature", "mean_turns", "mean_score", "first_player_win_rate", "bingos_per_game",
            "games_per_second", "seconds", "created", "updated", "error"}


def loaded_for(engine, leaves) -> engine_module.Loaded:
    return engine_module.Loaded(engine, leaves, engine.lexicon_id, None, engine_module.leaves_hash(leaves))


def make_set(games_dir: Path, name: str, loaded, games: int = 40, *, target: int | None = None, block: int = 20,
             pid: int | None = None, stop: bool = False, request: dict | None = None,
             console: list[str] | None = None, **settings) -> Path:
    """A game set folder: `games` games made for real (a target above that leaves it partial),
    optionally with the files a running or requested build leaves."""
    directory = games_dir / name
    if games or target:
        made = iter([False] * (games // block) + [True])
        games_module.generate(loaded, directory, target or games, block=block, say=lambda _: None,
                              should_stop=(lambda: next(made)) if target else None, **settings)
    directory.mkdir(parents=True, exist_ok=True)
    if pid is not None:
        (directory / "running.json").write_text(json.dumps({"pid": pid, "started": time.time()}), encoding="utf-8")
    if stop:
        (directory / "STOP").write_text("", encoding="utf-8")
    if request is not None:
        (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    if console is not None:
        (directory / "console.log").write_text("".join(line + "\n" for line in console), encoding="utf-8")
    return directory
