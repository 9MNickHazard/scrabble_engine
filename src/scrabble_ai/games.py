"""Sets of saved games to train from.

A game set is a folder under `data/games/` holding games the static evaluator (highest score +
leave value) played against itself, their openings varied so they are not all alike:

    games.bin     the games, each as its moves (a few hundred bytes; `scrabble_core::record`)
    index.bin     where each game starts in games.bin, and last the file's length (uint64 each)
    meta.json     what the set is: word list, leave values, how many games, how they were made
    running.json  present while a process is adding games (its pid and start time)
    STOP          create this file to make that process stop after its current block

Games are added in blocks and the three files are brought up to date after each one, so a set
can be stopped and continued, and a crash loses at most one block. A set is tied to the word
list and leave values it was played with: `open_games` refuses to train a network for other ones.
"""

from __future__ import annotations

import json
import os
import re
import time
import zlib
from pathlib import Path
from typing import Callable

import numpy as np

from scrabble_ai import _core
from scrabble_ai import engine as engine_module

GAMES_DIR = engine_module.DATA_DIR / "games"
FORMAT = 1
NAME = re.compile(r"[A-Za-z0-9_-]+")
# Seeds for recorded games start far from the ranges self-play, evaluation and the app use, and
# every set gets a billion of its own (chosen from its name) so two sets never hold the same game.
SEED_BASE = 30_000_000_000_000
BLOCK = 5_000


class GamesError(ValueError):
    """Something the user can fix: a bad name, a set made with another word list, and so on."""


def set_dir(name: str, root: Path | None = None) -> Path:
    """The folder of the set called `name`. A value with a path separator is taken as a folder."""
    if any(separator in name for separator in ("/", "\\")):
        return Path(name)
    if not NAME.fullmatch(name):
        raise GamesError(f"a game set's name may only use letters, digits, - and _: {name!r}")
    return (root or GAMES_DIR) / name


def default_first_seed(name: str) -> int:
    return SEED_BASE + (zlib.crc32(name.encode("utf-8")) % 100_000) * 1_000_000_000


def read_meta(directory: Path) -> dict | None:
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and meta.get("format") == FORMAT else None


def _write_meta(directory: Path, meta: dict) -> None:
    temporary = directory / "meta.json.tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, directory / "meta.json")


def _append(path: Path, data: bytes) -> None:
    with open(path, "ab") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


OPENING_PLIES, OPENING_TEMPERATURE = 4, 6.0


def generate(loaded: engine_module.Loaded, directory: Path, games: int, *, opening_plies: int | None = None,
             opening_temperature: float | None = None, first_seed: int | None = None, block: int = BLOCK,
             should_stop: Callable[[], bool] | None = None, say: Callable[[str], None] = print) -> dict:
    """Bring the set in `directory` up to `games` games, creating it if need be, and return its
    meta. The opening settings (how many of the first plies may be varied, and how freely; 4 and
    6.0 points unless given) are fixed when a set is created: a set that exists keeps its own, and
    asking for different ones is an error. Stops early, cleanly, when `should_stop()` or a STOP
    file in the folder says so."""
    if games < 1:
        raise GamesError("a game set needs at least one game")
    if opening_plies is not None and not 0 <= opening_plies <= 20:
        raise GamesError("opening_plies must be between 0 and 20")
    if opening_temperature is not None and not opening_temperature >= 0:
        raise GamesError("opening_temperature cannot be negative")
    directory.mkdir(parents=True, exist_ok=True)
    meta = read_meta(directory)
    if meta is None:
        opening_plies = OPENING_PLIES if opening_plies is None else opening_plies
        opening_temperature = OPENING_TEMPERATURE if opening_temperature is None else float(opening_temperature)
        if (directory / "games.bin").exists() and (directory / "games.bin").stat().st_size > 0:
            raise GamesError(f"{directory} holds games but no readable meta.json; move it away or delete it")
        meta = {
            "format": FORMAT, "name": directory.name, "lexicon_id": loaded.lexicon_id, "leaves_hash": loaded.leaves_hash,
            "player": "greedy-leave", "opening_plies": opening_plies, "opening_temperature": opening_temperature,
            "first_seed": default_first_seed(directory.name) if first_seed is None else int(first_seed),
            "games": 0, "target": games, "bytes": 0, "moves": 0, "score_sum": 0, "first_player_wins": 0.0, "bingos": 0,
            "seconds": 0.0, "created": time.time(), "updated": time.time(),
        }
        (directory / "games.bin").write_bytes(b"")
        (directory / "index.bin").write_bytes(np.zeros(1, "<u8").tobytes())
        _write_meta(directory, meta)
    else:
        if (meta["lexicon_id"], meta["leaves_hash"]) != (loaded.lexicon_id, loaded.leaves_hash):
            raise GamesError(f"the set {directory.name} was played with another word list or other leave values; "
                             "its games cannot be mixed with new ones")
        asked = {"opening_plies": opening_plies, "opening_temperature": opening_temperature, "first_seed": first_seed}
        if any(value is not None and value != meta[key] for key, value in asked.items()):
            raise GamesError(f"the set {directory.name} was made with other settings (opening plies {meta['opening_plies']}, "
                             f"temperature {meta['opening_temperature']}, first seed {meta['first_seed']})")
        # Anything past what meta.json records was written by a block that did not finish.
        with open(directory / "games.bin", "r+b") as fh:
            fh.truncate(meta["bytes"])
        with open(directory / "index.bin", "r+b") as fh:
            fh.truncate(8 * (meta["games"] + 1))
        meta["target"] = max(games, meta["games"])
        _write_meta(directory, meta)

    from scrabble_ai.app.procs import clear_stale_stop

    stop_file, running = directory / "STOP", directory / "running.json"
    clear_stale_stop(stop_file)
    running.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}), encoding="utf-8")
    say(f"game set {directory.name}: {meta['games']:,} of {games:,} games, word list {loaded.lexicon_id}")
    started, at_start, last = time.time(), meta["games"], time.time()
    seconds_before = meta["seconds"]
    try:
        while meta["games"] < games:
            if stop_file.exists() or (should_stop is not None and should_stop()):
                say("stopping: asked to")
                break
            count = min(block, games - meta["games"])
            data, lengths, totals = loaded.engine.generate_games(loaded.leaves, meta["first_seed"] + meta["games"], count,
                                                                 meta["opening_plies"], meta["opening_temperature"])
            offsets = meta["bytes"] + np.cumsum(lengths.astype(np.uint64))
            _append(directory / "games.bin", data)
            _append(directory / "index.bin", offsets.astype("<u8").tobytes())
            meta["games"] += count
            meta["bytes"] = int(offsets[-1])
            for key in ("moves", "score_sum", "bingos"):
                meta[key] += int(totals[key])
            meta["first_player_wins"] += float(totals["first_player_wins"])
            meta["seconds"] = seconds_before + time.time() - started
            meta["updated"] = time.time()
            _write_meta(directory, meta)
            if time.time() - last >= 10 or meta["games"] >= games:
                last = time.time()
                rate = (meta["games"] - at_start) / max(1e-9, time.time() - started)
                say(f"  {meta['games']:,} games ({rate:,.0f}/s), {meta['bytes'] / 2**20:,.0f} MB")
    finally:
        running.unlink(missing_ok=True)
        stop_file.unlink(missing_ok=True)
    return meta


def open_games(directory: Path, loaded: engine_module.Loaded) -> tuple[_core.GameSet, dict]:
    """The set in `directory`, ready to read positions from, with its meta. Raises `GamesError`
    if it is missing, empty, or was played with another word list or other leave values."""
    meta = read_meta(directory)
    if meta is None:
        raise GamesError(f"there is no game set at {directory}")
    if meta["games"] < 1:
        raise GamesError(f"the game set {meta['name']} has no games yet")
    if meta["lexicon_id"] != loaded.lexicon_id:
        raise GamesError(f"the game set {meta['name']} was played with the word list {meta['lexicon_id']}, not {loaded.lexicon_id}")
    if meta["leaves_hash"] != loaded.leaves_hash:
        raise GamesError(f"the game set {meta['name']} was played with other leave values than the ones loaded")
    try:
        # Only the games meta.json records: anything past them was written by a block that did
        # not finish (a kill), and the next `generate` cuts it off.
        games = _core.GameSet(directory, meta["games"])
    except ValueError as error:
        raise GamesError(f"the game set {meta['name']} is damaged: {error}") from None
    return games, meta


def summary(directory: Path) -> dict | None:
    """What the app lists about a set, or None if the folder is not one."""
    meta = read_meta(directory)
    if meta is None:
        return None
    games = meta["games"]
    return {
        "name": directory.name,
        "games": games,
        "target": meta.get("target", games),
        "bytes": meta["bytes"],
        "lexicon_id": meta["lexicon_id"],
        "leaves_hash": meta["leaves_hash"],
        "opening_plies": meta["opening_plies"],
        "opening_temperature": meta["opening_temperature"],
        "mean_turns": meta["moves"] / games if games else None,
        "mean_score": meta["score_sum"] / (2 * games) if games else None,
        "first_player_win_rate": meta["first_player_wins"] / games if games else None,
        "bingos_per_game": meta["bingos"] / games if games else None,
        "games_per_second": games / meta["seconds"] if meta.get("seconds") else None,
        "seconds": meta.get("seconds", 0.0),
        "created": meta["created"],
        "updated": meta["updated"],
    }
