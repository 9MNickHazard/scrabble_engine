"""Run one match in its own process: `python -m scrabble_ai.app.match_job <match folder>`."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

from scrabble_ai import engine as engine_module
from scrabble_ai.app import players as players_module
from scrabble_ai.app.matches import play_match_detailed, quick_stats, write_json
from scrabble_ai.app.players import Players
from scrabble_ai.app.runs import RUNS_DIR, read_json


def main(folder: str) -> int:
    match_dir = Path(folder)
    spec = read_json(match_dir / "spec.json")
    started = time.time()

    def status(state: str, games_done: int, games: dict | None = None, error: str | None = None) -> None:
        record = {"status": state, "games_done": games_done, "started": started, "seconds": round(time.time() - started, 2),
                  "error": error}
        if games is not None:
            record.update(quick_stats(games))
        write_json(match_dir / "status.json", record)

    try:
        loaded = engine_module.load(allow_missing_leaves=True)  # as the app does: bots can play without them
        players = Players(loaded, Path(spec.get("runs_dir") or RUNS_DIR))
        scorers = {s: players.scorer(s) for s in {spec["a"], spec["b"]} if players_module.kind(s) == "model"}
        seeds = np.arange(spec["pairs"], dtype=np.uint64) + np.uint64(spec["first_seed"])

        def progress(games: dict, done: int) -> None:
            write_json(match_dir / "games.json", games)
            status("running", done, games)

        cancelled = lambda: (match_dir / "CANCEL").exists()  # noqa: E731
        games = play_match_detailed(loaded.engine, loaded.leaves, scorers, spec["a"], spec["b"], seeds,
                                    players.device if scorers else None, spec.get("candidates", 64),
                                    progress=progress, cancelled=cancelled)
        write_json(match_dir / "games.json", games)
        done = len(games["score_a"])
        status("cancelled" if cancelled() and done < 2 * spec["pairs"] else "finished", done, games)
        return 0
    except Exception as error:  # the app shows this instead of a silent stall
        traceback.print_exc()
        status("failed", 0, error=f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
