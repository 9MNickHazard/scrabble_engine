"""Fast matches: two players, many games, no board.

A match runs as its own process (`python -m scrabble_ai.app.match_job <folder>`) so a long one
neither blocks the app nor dies with its window. Its folder, `data/app/matches/<id>/`, holds

    spec.json     who plays, how many pairs, which seeds
    status.json   running / finished / cancelled / failed, progress, the result so far
    games.json    one column per statistic, one entry per finished game
    CANCEL        asks the job to stop
    job.log       the job's console output
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from scrabble_ai import _core
from scrabble_ai.app import players as players_module
from scrabble_ai.app.players import NotFound, PlayerError, Players
from scrabble_ai.app.procs import marker_pid, marker_time, started_after
from scrabble_ai.app.runs import pid_alive, python_executable, read_json, spawn_detached

# scrabble_ai.selfplay and scrabble_ai.evaluate import PyTorch, which takes seconds to load, so
# they are imported inside the functions that need them: the app's window opens without it.

COLUMNS = ["seed", "a_first", "score_a", "score_b", "turns", "bingos_a", "bingos_b", "exchanges_a", "exchanges_b",
           "passes_a", "passes_b", "best_play_a", "best_play_b", "best_word_a", "best_word_b", "ended_by"]


def write_json(path: Path, data) -> None:
    """Replace a JSON file in one step. Windows refuses the rename while a reader has the file
    open, so retry briefly instead of failing the job."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    os.replace(temporary, path)


def play_match_detailed(engine, leaves, scorers: dict, a: str, b: str, seeds: np.ndarray, device, candidates: int = 64,
                        exchanges: int = 4, slots: int = 512, progress=None, cancelled=None) -> dict[str, list]:
    """Play every seed twice (A first, then B first) and return per-game statistics.

    `scorers` maps a model spec to its `Scorer`; bots are played inside the engine. `progress`
    is called with the columns for the pairs complete so far; `cancelled()` stops the match.
    Returns the columns for all complete pairs.
    """
    from scrabble_ai.selfplay import CandidateBuffers, choose

    games = 2 * len(seeds)  # game 2i: A moves first; game 2i+1: B moves first
    slots = max(1, min(slots, games))
    belief_samples = max((getattr(scorer, "belief_samples", 0) for scorer in scorers.values()), default=0)
    env = _core.BatchEnv(engine, leaves, slots, candidates, min(exchanges, candidates - 1), belief_samples)
    buffers = CandidateBuffers(slots, candidates)
    generator = None
    if scorers:
        import torch

        generator = torch.Generator(device=device)
    rows: list[dict | None] = [None] * games
    game_of_slot = np.full(slots, -1, np.int64)
    next_game = 0

    def controller(spec: str) -> str:
        return "external" if spec in scorers else players_module.bot_name(spec)

    def start(slot: int) -> None:
        nonlocal next_game
        if next_game >= games:
            env.clear(slot)
            game_of_slot[slot] = -1
            return
        first, second = (a, b) if next_game % 2 == 0 else (b, a)
        env.reset(slot, int(seeds[next_game // 2]), controller(first), controller(second))
        game_of_slot[slot] = next_game
        next_game += 1

    def columns() -> dict[str, list]:
        out: dict[str, list] = {name: [] for name in COLUMNS}
        for pair in range(len(seeds)):
            if rows[2 * pair] is None or rows[2 * pair + 1] is None:
                continue
            for row in (rows[2 * pair], rows[2 * pair + 1]):
                for name in COLUMNS:
                    out[name].append(row[name])
        return out

    for slot in range(slots):
        start(slot)
    finished, last_report = 0, time.time()
    while finished < games:
        if cancelled is not None and cancelled():
            break
        buffers.prepare(env)
        for slot in np.flatnonzero(buffers.status == _core.STATUS_FINISHED):
            slot = int(slot)
            game = int(game_of_slot[slot])
            a_first = game % 2 == 0
            seat_a, seat_b = (0, 1) if a_first else (1, 0)
            stats = env.game_stats(slot)
            scores = buffers.scores[slot]
            rows[game] = {
                "seed": int(seeds[game // 2]), "a_first": a_first,
                "score_a": int(scores[seat_a]), "score_b": int(scores[seat_b]), "turns": int(env.turn(slot)),
                "bingos_a": stats["bingos"][seat_a], "bingos_b": stats["bingos"][seat_b],
                "exchanges_a": stats["exchanges"][seat_a], "exchanges_b": stats["exchanges"][seat_b],
                "passes_a": stats["passes"][seat_a], "passes_b": stats["passes"][seat_b],
                "best_play_a": stats["best_score"][seat_a], "best_play_b": stats["best_score"][seat_b],
                "best_word_a": stats["best_word"][seat_a], "best_word_b": stats["best_word"][seat_b],
                "ended_by": stats["ended_by"],
            }
            finished += 1
            start(slot)
        deciding = np.flatnonzero(buffers.status == _core.STATUS_DECISION)
        if len(deciding):
            a_to_move = (buffers.mover[deciding] == 0) == (game_of_slot[deciding] % 2 == 0)
            actions = np.zeros(slots, np.int32)
            for spec, mask in ((a, a_to_move), (b, ~a_to_move)):
                if spec in scorers and mask.any():
                    scorer = scorers[spec]
                    choice = choose(buffers, scorer, 0.0, getattr(scorer, "win_weight", 0.0), getattr(scorer, "win_scale", 100.0),
                                    device, generator, only=deciding[mask])
                    actions[choice.slots] = choice.actions
            env.apply(actions)
        if progress is not None and time.time() - last_report > 0.75:
            last_report = time.time()
            progress(columns(), finished)
    result = columns()
    if progress is not None:
        progress(result, finished)
    return result


def quick_stats(games: dict[str, list]) -> dict:
    """Spread and win rate so far, for the match list."""
    from scrabble_ai.evaluate import summarize

    pairs = len(games["score_a"]) // 2
    if pairs < 2:
        return {"spread": None, "spread_ci": None, "win_rate": None}
    spread = np.array(games["score_a"], np.float64) - np.array(games["score_b"], np.float64)
    first = np.array(games["a_first"], bool)
    result = summarize("a", "b", spread[first], spread[~first], np.array(games["score_a"], np.float64),
                       np.array(games["score_b"], np.float64), np.array(games["turns"], np.float64))
    clean = lambda x: None if isinstance(x, float) and math.isnan(x) else x  # noqa: E731
    return {"spread": clean(result.spread), "spread_ci": clean(result.spread_ci), "win_rate": clean(result.win_rate)}


def of_run(spec, run: str) -> bool:
    """Is `spec` a model of the run `run`? (The run is compared without regard to case, as
    Windows compares folder names.)"""
    return isinstance(spec, str) and spec.startswith("model:") and spec[6:].partition("/")[0].lower() == run.lower()


def renamed_spec(spec, old: str, new: str):
    """`spec` pointed at the run's new name if it is a model of the run `old`; any other spec as it is."""
    return f"model:{new}/{spec[6:].partition('/')[2]}" if of_run(spec, old) else spec


class Matches:
    def __init__(self, app_dir: Path, players: Players):
        self.dir = app_dir / "matches"
        self.players = players

    def _folder(self, match_id: str) -> Path:
        if not match_id.startswith("m_") or not match_id.replace("_", "").isalnum() or not match_id.isascii():
            raise NotFound(f"there is no match {match_id}")
        folder = self.dir / match_id
        if not folder.is_dir():
            raise NotFound(f"there is no match {match_id}")
        return folder

    def start(self, a: str, b: str, pairs: int, candidates: int = 64) -> str:
        for value, what in ((pairs, "pairs"), (candidates, "candidates")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise PlayerError(f"{what} must be a whole number")
        for spec in (a, b):
            if players_module.kind(spec) == "human":
                raise PlayerError("a match is between bots and models")
            if players_module.kind(spec) == "model":
                self.players.model_path(spec)
        if not 1 <= pairs <= 100_000:
            raise PlayerError("pairs must be between 1 and 100,000")
        if not 2 <= candidates <= 512:
            raise PlayerError("candidates must be between 2 and 512")
        match_id = "m_" + time.strftime("%Y%m%d_%H%M%S")
        folder = self.dir / match_id
        suffix = 0
        while folder.exists():
            suffix += 1
            folder = self.dir / f"{match_id}_{suffix}"
        folder.mkdir(parents=True)
        spec = {"id": folder.name, "a": a, "b": b, "a_name": players_module.name(a), "b_name": players_module.name(b),
                "pairs": int(pairs), "candidates": int(candidates), "created": time.time(),
                "runs_dir": str(self.players.runs_dir),
                # Fresh seeds for every match, away from the ones training and its evaluations use.
                "first_seed": 8_000_000_000_000 + int(time.time() * 1000) % 1_000_000_000 * 1000}
        write_json(folder / "spec.json", spec)
        write_json(folder / "status.json", {"status": "running", "games_done": 0, "started": time.time(), "seconds": 0.0})
        pid = spawn_detached([python_executable(), "-u", "-m", "scrabble_ai.app.match_job", str(folder)], folder / "job.log")
        write_json(folder / "pid.json", {"pid": pid})
        return folder.name

    def _summary(self, folder: Path) -> dict | None:
        spec, status = read_json(folder / "spec.json"), read_json(folder / "status.json")
        if spec is None or status is None:
            return None
        state = status.get("status", "failed")
        if state == "running" and time.time() - spec.get("created", 0) > 20:
            marker = read_json(folder / "pid.json")
            pid = marker_pid(marker)
            if not pid_alive(pid) or started_after(pid, marker_time(marker, folder / "pid.json")):
                state = "failed"
        total = 2 * spec["pairs"]
        return {
            "id": spec["id"], "a": spec["a"], "b": spec["b"], "a_name": spec["a_name"], "b_name": spec["b_name"],
            "pairs": spec["pairs"], "status": state, "progress": min(1.0, status.get("games_done", 0) / total),
            "games_done": status.get("games_done", 0), "created": spec["created"], "seconds": status.get("seconds", 0.0),
            "spread": status.get("spread"), "spread_ci": status.get("spread_ci"), "win_rate": status.get("win_rate"),
            "error": status.get("error"),
        }

    def list(self) -> list[dict]:
        if not self.dir.is_dir():
            return []
        summaries = [s for s in (self._summary(p) for p in self.dir.iterdir() if p.is_dir()) if s]
        summaries.sort(key=lambda s: -s["created"])
        return summaries

    def active(self) -> list[str]:
        return [s["id"] for s in self.list() if s["status"] == "running"]

    def get(self, match_id: str) -> dict:
        from scrabble_ai.app import insights

        folder = self._folder(match_id)
        summary = self._summary(folder)
        if summary is None:
            raise NotFound(f"there is no match {match_id}")
        games = read_json(folder / "games.json") or {name: [] for name in COLUMNS}
        stats, highlights = insights.match_summary(games, summary["a_name"], summary["b_name"])
        return {"match": summary, "summary": stats, "games": games, "highlights": highlights}

    def cancel(self, match_id: str) -> None:
        (self._folder(match_id) / "CANCEL").write_text("", encoding="utf-8")

    def _of_run(self, run: str) -> list[tuple[Path, dict, bool]]:
        """(folder, spec, still running) for every match in which a model of the run `run` plays."""
        if not self.dir.is_dir():
            return []
        found = []
        for folder in sorted(self.dir.iterdir()):
            spec = read_json(folder / "spec.json") if folder.is_dir() else None
            if spec is None or not (of_run(spec.get("a"), run) or of_run(spec.get("b"), run)):
                continue
            try:
                summary = self._summary(folder)
            except (KeyError, TypeError, ValueError, ArithmeticError):  # a spec.json `start` did not write
                summary = None
            found.append((folder, spec, summary is not None and summary["status"] == "running"))
        return found

    def running_with(self, run: str) -> list[str]:
        """The ids of the matches still running in which a model of the run `run` plays. Their
        job holds the model's path, so the run must not be renamed under them."""
        return [folder.name for folder, _, running in self._of_run(run) if running]

    def rename_run(self, old: str, new: str) -> list[str]:
        """Point the matches played by models of the run `old` at its new name `new`: the player
        specs in their spec.json and the names shown for them, so that they list under the new
        name and their games can still be replayed. Running matches are left alone (the server
        refuses the rename while there are any). Returns the ids of the matches changed."""
        changed = []
        for folder, spec, running in self._of_run(old):
            if running:
                continue
            for side in ("a", "b"):
                if of_run(spec.get(side), old):
                    spec[side] = renamed_spec(spec[side], old, new)
                    try:
                        spec[f"{side}_name"] = players_module.name(spec[side])
                    except PlayerError:
                        pass  # not a model id a match could have started with: keep the name it has
            write_json(folder / "spec.json", spec)
            changed.append(folder.name)
        return changed
