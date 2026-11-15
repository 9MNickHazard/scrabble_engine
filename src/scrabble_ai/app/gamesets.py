"""Game sets for the app: list them, start and stop the process that plays and saves the games,
delete them. The games themselves are `scrabble_ai.games`' business; this module only looks at
the files a set's folder holds and starts `scrabble-ai games` as a detached process, the way
training runs are started.

A set's folder may also hold, besides what `games.py` describes:

    request.json   what the app last asked for (games, opening settings), so that a set whose
                   process has not yet written meta.json can still be listed
    console.log    the output of the processes that built it
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from scrabble_ai import games as games_module
from scrabble_ai.app.players import NotFound
from scrabble_ai.app.procs import kill_process, marker_pid, marker_time, pid_alive, read_json, started_after
from scrabble_ai.app.runs import RunError, python_executable, spawn_detached

MAX_GAMES = 500_000_000
# Measured on 15 November 2026 over 3,000 games: 856,491 bytes. Used for sizing before a set has
# any games of its own to go by.
BYTES_PER_GAME = 286


def _alive(marker_path: Path) -> bool:
    marker = read_json(marker_path)
    pid = marker_pid(marker)
    return marker is not None and pid_alive(pid) and not started_after(pid, marker_time(marker, marker_path))


class GameSets:
    def __init__(self, games_dir: Path):
        self.games_dir = games_dir

    def path(self, name: str) -> Path:
        if not isinstance(name, str) or not games_module.NAME.fullmatch(name) or len(name) > 64:
            raise RunError("a game set's name may only use letters, digits, - and _ (at most 64)")
        return self.games_dir / name

    def names(self) -> list[str]:
        if not self.games_dir.is_dir():
            return []
        return sorted(p.name for p in self.games_dir.iterdir()
                      if p.is_dir() and games_module.NAME.fullmatch(p.name)
                      and ((p / "meta.json").exists() or (p / "request.json").exists()))

    def exists(self, name: str) -> bool:
        return isinstance(name, str) and name in self.names()

    def _existing(self, name: str) -> Path:
        """The folder of a set that exists; a name that is not one (or could not be one) is not found."""
        if not self.exists(name):
            raise NotFound(f"there is no game set named {name!r}")
        return self.games_dir / name

    def summary(self, name: str) -> dict:
        directory = self._existing(name)
        request = read_json(directory / "request.json") or {}
        made = games_module.summary(directory)
        if made is None and not request:
            raise NotFound(f"there is no game set named {name!r}")
        out = made or {
            "name": name, "games": 0, "target": 0, "bytes": 0, "lexicon_id": None, "leaves_hash": None,
            "opening_plies": request.get("opening_plies"), "opening_temperature": request.get("opening_temperature"),
            "mean_turns": None, "mean_score": None, "first_player_win_rate": None, "bingos_per_game": None,
            "games_per_second": None, "seconds": 0.0, "created": request.get("requested"), "updated": request.get("requested"),
        }
        out["target"] = max(out["target"], int(request.get("games", 0) or 0)) if _alive(directory / "running.json") else out["target"]
        running = directory / "running.json"
        error = None
        if running.exists():
            if _alive(running):
                status = "stopping" if (directory / "STOP").exists() else "building"
            else:
                status = "failed"
                error = "\n".join(self.log(name, 8)) or "The process that was playing the games ended unexpectedly."
        elif out["games"] == 0:
            status = "empty"
        else:
            status = "complete" if out["games"] >= out["target"] else "partial"
        out.update(status=status, error=error, progress=(out["games"] / out["target"]) if out["target"] else None)
        return out

    def list(self) -> dict:
        sets = []
        for name in self.names():
            try:
                sets.append(self.summary(name))
            except Exception as error:  # one unreadable folder must not hide the others
                sets.append({"name": name, "status": "failed", "error": f"could not read this game set: {error}", "games": 0,
                             "target": 0, "bytes": 0})
        order = {"building": 0, "stopping": 0}
        sets.sort(key=lambda s: (order.get(s["status"], 1), -(s.get("updated") or 0)))
        measured = [s for s in sets if s.get("games") and s.get("bytes")]
        rates = [s["games_per_second"] for s in sets if s.get("games_per_second")]
        return {"sets": sets, "estimate": {
            "bytes_per_game": (sum(s["bytes"] for s in measured) / sum(s["games"] for s in measured)) if measured else BYTES_PER_GAME,
            "games_per_second": rates[0] if rates else None,
        }}

    def log(self, name: str, tail: int = 200) -> list[str]:
        try:
            lines = (self._existing(name) / "console.log").read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-tail:] if tail > 0 else []

    def start(self, name: str, games, opening_plies=None, opening_temperature=None) -> dict:
        """Create the set, or bring one that exists up to `games` games."""
        directory = self.path(name)
        if isinstance(games, bool) or not isinstance(games, int) or not 1 <= games <= MAX_GAMES:
            raise RunError(f"games must be a whole number between 1 and {MAX_GAMES:,}")
        if opening_plies is not None and (isinstance(opening_plies, bool) or not isinstance(opening_plies, int)
                                          or not 0 <= opening_plies <= 20):
            raise RunError("opening_plies must be a whole number between 0 and 20")
        if opening_temperature is not None and (isinstance(opening_temperature, bool)
                                                or not isinstance(opening_temperature, (int, float))
                                                or not 0 <= opening_temperature <= 100):
            raise RunError("opening_temperature must be a number between 0 and 100")
        # A name that differs from an existing set's only in its capitals is the same folder on
        # Windows; and a folder that holds something else is not ours to fill.
        if any(existing != name and existing.lower() == name.lower() for existing in self.names()) or (
                name not in self.names() and directory.exists() and any(directory.iterdir())):
            raise RunError(f"a game set named {name!r} already exists")
        meta = games_module.read_meta(directory)
        if directory.exists() and _alive(directory / "running.json"):
            raise RunError(f"{name} is already being built")
        if meta is not None:
            if games <= meta["games"]:
                raise RunError(f"{name} already has {meta['games']:,} games; ask for more than that to add to it")
            for key, value in (("opening_plies", opening_plies), ("opening_temperature", opening_temperature)):
                if value is not None and value != meta[key]:
                    raise RunError(f"{name} was made with {key.replace('_', ' ')} {meta[key]}; a set keeps the settings it was made with")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "STOP").unlink(missing_ok=True)
        (directory / "request.json").write_text(json.dumps(
            {"games": games, "opening_plies": opening_plies, "opening_temperature": opening_temperature, "requested": time.time()}),
            encoding="utf-8")
        command = [python_executable(), "-u", "-m", "scrabble_ai.cli", "games", "--name", name, "--games", str(games),
                   "--dir", str(self.games_dir)]
        if meta is None:
            if opening_plies is not None:
                command += ["--opening-plies", str(opening_plies)]
            if opening_temperature is not None:
                command += ["--opening-temperature", str(opening_temperature)]
        pid = spawn_detached(command, directory / "console.log")
        # The generator rewrites this itself once it is up; until then it marks the set as building.
        (directory / "running.json").write_text(json.dumps({"pid": pid, "started": time.time()}), encoding="utf-8")
        return self.summary(name)

    def stop(self, name: str, force: bool = False) -> None:
        directory = self._existing(name)
        running = directory / "running.json"
        if not _alive(running):
            raise RunError(f"{name} is not being built")
        if force:
            kill_process(marker_pid(read_json(running)))
            running.unlink(missing_ok=True)
            (directory / "STOP").unlink(missing_ok=True)
        else:
            (directory / "STOP").write_text("", encoding="utf-8")

    def delete(self, name: str, used_by: list[str] | None = None) -> None:
        """Delete the set's folder. `used_by` names the training runs that are learning from it
        right now: while there are any, it stays."""
        directory = self._existing(name)
        if _alive(directory / "running.json"):
            raise RunError(f"{name} is being built; stop it first")
        if used_by:
            raise RunError(f"{name} is being trained from by {', '.join(used_by)}; stop that first")
        if directory.resolve().parent != self.games_dir.resolve():
            raise RunError("refusing to delete outside the game sets folder")
        for attempt in range(20):
            try:
                shutil.rmtree(directory)
                return
            except FileNotFoundError:
                return
            except OSError:
                if attempt == 19:
                    raise RunError(f"{name} still has files open; try again in a moment") from None
                time.sleep(0.25)
