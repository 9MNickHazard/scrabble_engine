"""Training runs as the app sees them: finding them, starting and stopping them, reading
their metrics.

A run is a folder `runs/<name>/` written by `scrabble_ai.train`. Runs are separate processes
started detached, so they outlive the app window; everything here works from the files a run
leaves behind:

    running.json     present while the training process is alive (holds its pid)
    queued.json      present while the run waits in a queue (holds its settings)
    STOP             asks the run to stop cleanly
    config.toml      the settings in force
    metrics.jsonl    one record per log line (kinds are listed in docs/APP_API.md)
    console.log      the process's console output, when started from the app
    checkpoints/     step_*.pt, best.pt, index.json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

from scrabble_ai import engine as engine_module
from scrabble_ai.app import insights
from scrabble_ai.app.procs import (  # noqa: F401  (pid_alive, kill_process and read_json are re-exported)
    kill_process, marker_pid, marker_time, pid_alive, read_json, started_after,
)
from scrabble_ai.config import Config, ConfigError, load_config
from scrabble_ai.lexicon import REPO_ROOT

RUNS_DIR = REPO_ROOT / "runs"
APP_DIR = REPO_ROOT / "data" / "app"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")  # used with fullmatch
# Settings the app decides itself: where a run lives and what it is called.
RESERVED_SETTINGS = ("run.name", "run.out_dir")
# Settings a checkpoint fixes for good (the network's shape, the word list, the value scale).
LOCKED_ON_RESUME = ("run.lexicon", "run.leaves", "model.", "train.value_scale")

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


class RunError(ValueError):
    """A request that cannot be carried out; the message is shown to the user."""


def python_executable() -> str:
    """The console Python next to the running interpreter (the app itself runs under pythonw)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.with_name("python.exe")).exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def spawn_detached(command: list[str], log_path: Path) -> int:
    """Start a process that keeps running after the app exits, its output appended to a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
    return process.pid


def override_arguments(overrides: dict) -> list[str]:
    arguments = []
    for key, value in overrides.items():
        text = str(value).lower() if isinstance(value, bool) else str(value)
        arguments += ["--set", f"{key}={text}"]
    return arguments


def flatten(config: dict) -> dict:
    return {f"{section}.{key}": value for section, values in config.items() for key, value in values.items()}


def normalize_eval(record: dict) -> dict | None:
    """An eval record's numbers under one layout (early runs keyed them by opponent)."""
    opponent = record.get("opponent")
    if not opponent:
        return None
    spread = record.get("spread", record.get(f"spread_vs_{opponent}"))
    if spread is None:
        return None
    return {"spread": spread, "ci": record.get("spread_ci", record.get(f"spread_ci_vs_{opponent}")),
            "win_rate": record.get("win_rate", record.get(f"win_rate_vs_{opponent}"))}


class _Metrics:
    """One run's metrics.jsonl, read incrementally."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.records: list[dict] = []
        self._overview: tuple[int, int, dict] | None = None

    def overview(self) -> dict:
        """How training went (insights.run_overview), worked out again only when the log has changed."""
        key = (self.offset, len(self.records))
        if self._overview is None or self._overview[:2] != key:
            self._overview = (*key, insights.run_overview(self.records))
        return self._overview[2]

    def refresh(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            self.offset, self.records = 0, []
            return self.records
        if size < self.offset:  # the file was replaced
            self.offset, self.records = 0, []
        if size > self.offset:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
            # Only consume whole lines: the trainer may be mid-write.
            end = chunk.rfind(b"\n") + 1
            for line in chunk[:end].splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("kind") == "eval":
                    numbers = normalize_eval(record)
                    if numbers:
                        record.setdefault("spread", numbers["spread"])
                        record.setdefault("spread_ci", numbers["ci"])
                        record.setdefault("win_rate", numbers["win_rate"])
                self.records.append(record)
            self.offset += end
        return self.records


class RunManager:
    def __init__(self, runs_dir: Path = RUNS_DIR, app_dir: Path = APP_DIR):
        self.runs_dir = runs_dir
        self.app_dir = app_dir
        self.lock = threading.RLock()
        self._metrics: dict[str, _Metrics] = {}
        self._defaults = flatten(Config().to_dict())

    # ---- reading ---------------------------------------------------------------------------

    def path(self, name: str) -> Path:
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise RunError("a run name uses letters, digits, - and _ only, and starts with a letter or digit")
        return self.runs_dir / name

    def exists(self, name: str) -> bool:
        return isinstance(name, str) and NAME_PATTERN.fullmatch(name) is not None and (self.runs_dir / name).is_dir()

    def names(self) -> list[str]:
        if not self.runs_dir.is_dir():
            return []
        return [p.name for p in self.runs_dir.iterdir() if p.is_dir() and NAME_PATTERN.fullmatch(p.name)]

    def _alive(self, marker_path: Path) -> bool:
        """Is the process this marker file names still running, and still that process? (Windows
        hands a finished process's number to a later one.)"""
        marker = read_json(marker_path)
        pid = marker_pid(marker)
        return marker is not None and pid_alive(pid) and not started_after(pid, marker_time(marker, marker_path))

    def _stored_overrides(self, run_dir: Path) -> dict:
        """The settings a run that has not started yet was given."""
        for file in ("queued.json", "unstarted.json"):
            stored = (read_json(run_dir / file) or {}).get("overrides")
            if isinstance(stored, dict):
                return stored
        return {}

    def records(self, name: str) -> list[dict]:
        with self.lock:
            metrics = self._metrics.get(name)
            if metrics is None:
                metrics = self._metrics[name] = _Metrics(self.path(name) / "metrics.jsonl")
            return metrics.refresh()

    def metrics(self, name: str, since: int) -> tuple[list[dict], int]:
        records = self.records(name)
        since = max(0, min(since, len(records)))
        return records[since:], len(records)

    def log(self, name: str, tail: int = 200) -> list[str]:
        """The last `tail` lines of the run's console output."""
        if tail <= 0:
            return []
        path = self.path(name) / "console.log"
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                want = 400 * tail
                while True:
                    start = max(0, size - want)
                    fh.seek(start)
                    data = fh.read()
                    # Enough once the chunk holds `tail` whole lines after its (possibly cut) first one.
                    if start == 0 or data.count(b"\n") > tail or want >= 8_000_000:
                        break
                    want *= 4
        except OSError:
            return []
        lines = data.decode("utf-8", errors="replace").splitlines()
        if start > 0:
            lines = lines[1:]  # the chunk began mid-line
        return lines[-tail:]

    def config(self, name: str) -> dict:
        """The run's settings: from its config.toml, or for a run that has not started, from
        the settings it was queued with."""
        run_dir = self.path(name)
        try:
            with open(run_dir / "config.toml", "rb") as fh:
                return tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            pass
        try:
            return load_config(None, [f"{k}={str(v).lower() if isinstance(v, bool) else v}"
                                      for k, v in self._stored_overrides(run_dir).items()]).to_dict()
        except ConfigError:
            return Config().to_dict()

    def _status(self, run_dir: Path, records: list[dict]) -> tuple[str, str | None]:
        """(status, error)."""
        if (run_dir / "running.json").exists():
            if self._alive(run_dir / "running.json"):
                return ("stopping" if (run_dir / "STOP").exists() else "running"), None
            return "failed", "\n".join(self.log(run_dir.name, 12)) or "The training process ended unexpectedly."
        queued = read_json(run_dir / "queued.json")
        if queued is not None:
            # The queue runner records its pid when it starts; give it a moment to do so.
            try:
                fresh = time.time() - float(queued.get("created", 0) or 0) < 30
            except (TypeError, ValueError):
                fresh = False
            if fresh or self._alive(self.app_dir / "queues" / f"{queued.get('queue', '')}.pid"):
                return "queued", None
            return "stopped", "The queue was interrupted before this run started. Resume starts it."
        stops = [r for r in records if r.get("kind") == "session" and r.get("event") == "stop"]
        if stops and stops[-1].get("finished"):
            return "finished", None
        return "stopped", None

    def summary(self, name: str) -> dict:
        run_dir = self.path(name)
        with self.lock:
            records = self.records(name)
            overview = self._metrics[name].overview()
        config = self.config(name)
        flat = flatten(config)
        status, error = self._status(run_dir, records)
        last = records[-1] if records else {}
        selfplay = [r for r in records if r.get("kind") == "selfplay"]
        starts = [r for r in records if r.get("kind") == "session" and r.get("event") == "start"]

        phase = None
        if selfplay or any(r.get("kind") == "session" and r.get("event") == "selfplay" for r in records):
            phase = "selfplay"
        elif any(r.get("kind") == "warmstart" for r in records):
            phase = "pretraining"
        elif starts or status in ("running", "stopping"):
            phase = "warmstart"

        # The latest evaluation is the last one in the log: its records share a step and game count. (Not the one
        # with the most games: after a forced stop and resume the game counter steps back.)
        latest_eval, latest_at = None, None
        for record in records:
            if record.get("kind") == "eval" and record.get("spread") is not None and record.get("opponent"):
                at = (record.get("step"), record.get("games"))
                if latest_eval is None or at != latest_at:
                    latest_eval, latest_at = {"games": record.get("games", 0)}, at
                latest_eval[record["opponent"]] = {"spread": record["spread"], "ci": record.get("spread_ci"),
                                                   "win_rate": record.get("win_rate")}

        hours_limit, games_limit = flat.get("run.hours", 0.0), flat.get("run.max_games", 0)
        progress = None
        if status == "finished":
            progress = 1.0
        elif status in ("running", "stopping") and starts:
            parts = []
            if hours_limit:
                parts.append((time.time() - starts[-1]["time"]) / (hours_limit * 3600))
            if games_limit:
                parts.append(last.get("games", 0) / games_limit)
            progress = min(1.0, max(parts)) if parts else None

        index = read_json(run_dir / "checkpoints" / "index.json") or {}
        best = index.get("best")
        checkpoints = len(list((run_dir / "checkpoints").glob("step_*.pt"))) if (run_dir / "checkpoints").is_dir() else 0
        parameters = next((r.get("parameters") for r in reversed(starts) if r.get("parameters")), index.get("parameters"))
        created = records[0]["time"] if records else _mtime(run_dir)
        changed = {key: value for key, value in flat.items()
                   if key in self._defaults and value != self._defaults[key]
                   and key not in ("run.name", "run.out_dir", "run.hours", "run.max_games")}
        return {
            "name": name,
            "status": status,
            "phase": phase,
            "network": f"{flat.get('model.blocks', '?')}x{flat.get('model.channels', '?')}",
            "parameters": parameters,
            "games": last.get("games", 0),
            "step": last.get("step", 0),
            "positions": last.get("positions", 0),
            "hours": last.get("hours", 0.0),
            "games_per_second": _games_per_second(records, selfplay, status in ("running", "stopping")),
            "created": created,
            "updated": last.get("time", created),
            "limit": {"hours": hours_limit, "max_games": games_limit},
            "progress": progress,
            "latest_eval": latest_eval,
            "best_step": best["step"] if best else None,
            "has_best": (run_dir / "checkpoints" / "best.pt").exists(),
            "checkpoints": checkpoints,
            "changed": changed,
            "error": error,
            **overview,  # trend, best_eval, health
        }

    def list(self) -> list[dict]:
        summaries = []
        for name in self.names():
            try:
                summaries.append(self.summary(name))
            except Exception as error:  # one unreadable run must not hide the others
                summaries.append({"name": name, "status": "failed", "error": f"could not read this run: {error}"})
        order = {"running": 0, "stopping": 0, "queued": 1}
        summaries.sort(key=lambda s: (order.get(s["status"], 2), -(s.get("updated") or 0)))
        return summaries

    def detail(self, name: str) -> dict:
        if not self.exists(name):
            raise RunError(f"there is no run named {name!r}")
        run_dir = self.path(name)
        checkpoint_dir = run_dir / "checkpoints"
        files = sorted(checkpoint_dir.glob("step_*.pt")) if checkpoint_dir.is_dir() else []
        checkpoints = [{"file": p.name, "step": int(p.stem.split("_")[1]), "size_bytes": p.stat().st_size,
                        "modified": p.stat().st_mtime, "model_id": f"{name}/{p.name}"} for p in files]
        summary = self.summary(name)
        best = None
        if summary["has_best"]:
            best_file = checkpoint_dir / "best.pt"
            best = {"file": "best.pt", "step": summary["best_step"], "model_id": f"{name}/best.pt",
                    "size_bytes": best_file.stat().st_size, "modified": best_file.stat().st_mtime}
        return {"run": summary, "config": self.config(name), "checkpoints": checkpoints, "best": best}

    def active(self) -> list[str]:
        return [s["name"] for s in self.list() if s["status"] in ("running", "stopping")]

    def queued(self) -> list[str]:
        return [s["name"] for s in self.list() if s["status"] == "queued"]

    # ---- acting ----------------------------------------------------------------------------

    @staticmethod
    def _check_overrides(overrides, base: dict | None = None) -> Config:
        """Refuse settings that are not settings, values that do not validate, and the two the
        app decides itself. `base` is the configuration the overrides are applied on top of."""
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict) or not all(isinstance(key, str) for key in overrides):
            raise RunError("overrides must be an object of setting: value")
        for key in overrides:
            if key in RESERVED_SETTINGS:
                raise RunError(f"{key} is set by the app and cannot be changed")
        items = [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in overrides.items()]
        try:
            if base is None:
                return load_config(None, items)
            from scrabble_ai.config import apply_overrides, config_from_dict

            config = config_from_dict(base)
            apply_overrides(config, items)
            config.validate()
            return config
        except ConfigError as error:
            raise RunError(str(error)) from None

    def _check_new(self, name: str, overrides: dict) -> None:
        run_dir = self.path(name)
        if run_dir.exists() or name.lower() in {existing.lower() for existing in self.names()}:
            raise RunError(f"a run named {name!r} already exists")
        config = self._check_overrides(overrides)
        try:
            engine_name, _ = engine_module.lexicon_entry(config.run.lexicon)
        except Exception as error:
            raise RunError(str(error)) from None
        del engine_name

    def _train_command(self, name: str, arguments: list[str]) -> list[str]:
        return [python_executable(), "-u", "-m", "scrabble_ai.cli", "train", "--name", name,
                "--set", f"run.out_dir={self.runs_dir}", *arguments]

    def _launch(self, name: str, arguments: list[str], overrides: dict | None = None) -> None:
        run_dir = self.path(name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "STOP").unlink(missing_ok=True)
        if not (run_dir / "config.toml").exists():
            # Until the trainer has written config.toml this is the only record of what was asked
            # for, and a run that dies in its first seconds must still be resumable as itself.
            stored = overrides if overrides is not None else self._stored_overrides(run_dir)
            (run_dir / "unstarted.json").write_text(json.dumps({"overrides": stored}), encoding="utf-8")
        (run_dir / "queued.json").unlink(missing_ok=True)
        pid = spawn_detached(self._train_command(name, arguments), run_dir / "console.log")
        # The trainer rewrites this itself once it is up; until then it marks the run as started.
        (run_dir / "running.json").write_text(json.dumps({"pid": pid, "started": time.time()}), encoding="utf-8")

    def start(self, specs: list[dict], mode: str) -> dict:
        """Start runs now, side by side (`parallel`), or one at a time (`queue`): a queue also
        waits for every run that is already training or queued."""
        if mode not in ("parallel", "queue"):
            raise RunError("mode must be 'parallel' or 'queue'")
        if not specs:
            raise RunError("nothing to start")
        names = [spec.get("name", "") for spec in specs]
        if len({name.lower() for name in names}) != len(names):  # Windows folder names ignore case
            raise RunError("two runs have the same name")
        with self.lock:
            for spec in specs:
                try:
                    self._check_new(spec["name"], spec.get("overrides") or {})
                except RunError as error:
                    # Say which run, so the form can show the message on the right one.
                    raise RunError(f"{spec['name']}: {error}" if len(specs) > 1 else str(error)) from None
            busy = any(s["status"] in ("running", "stopping", "queued") for s in self.list())
            if mode == "parallel" or (len(specs) == 1 and not busy):
                for spec in specs:
                    self._launch(spec["name"], override_arguments(spec.get("overrides") or {}), spec.get("overrides") or {})
                return {"started": names, "queued": []}
            queue_dir = self.app_dir / "queues"
            queue_dir.mkdir(parents=True, exist_ok=True)
            stem = f"queue_{time.strftime('%Y%m%d_%H%M%S')}"
            queue_file, extra = queue_dir / f"{stem}.json", 1
            while queue_file.exists():  # later queues in the same second sort after the first
                extra += 1
                queue_file = queue_dir / f"{stem}_{extra:03d}.json"
            entries = [{"name": spec["name"], "command": self._train_command(spec["name"], override_arguments(spec.get("overrides") or {})),
                        "log": str(self.path(spec["name"]) / "console.log")} for spec in specs]
            queue_file.write_text(json.dumps({"runs": entries, "runs_dir": str(self.runs_dir)}), encoding="utf-8")
            # Markers first, then the runner: it skips any run whose marker is missing.
            for spec in specs:
                run_dir = self.path(spec["name"])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "queued.json").write_text(json.dumps(
                    {"queue": queue_file.stem, "overrides": spec.get("overrides") or {}, "created": time.time()}), encoding="utf-8")
            spawn_detached([python_executable(), "-u", "-m", "scrabble_ai.app.queue_runner", str(queue_file)],
                           queue_dir / (queue_file.stem + ".log"))
            # With nothing else going the runner starts the first one at once.
            return {"started": [] if busy else names[:1], "queued": names if busy else names[1:]}

    def stop(self, name: str, force: bool = False) -> None:
        run_dir = self.path(name)
        if not run_dir.is_dir():
            raise RunError(f"there is no run named {name!r}")
        running_file = run_dir / "running.json"
        if (run_dir / "queued.json").exists() and not running_file.exists():
            # Not started yet: taking it out of the queue is all there is to do. Its settings are
            # kept, so that resuming it later starts the run that was asked for.
            (run_dir / "unstarted.json").write_text(json.dumps({"overrides": self._stored_overrides(run_dir)}), encoding="utf-8")
            (run_dir / "queued.json").unlink(missing_ok=True)
            return
        if not self._alive(running_file):
            raise RunError(f"{name} is not running")
        if force:
            kill_process(marker_pid(read_json(running_file)))
            running_file.unlink(missing_ok=True)
            (run_dir / "STOP").unlink(missing_ok=True)
            last = (self.records(name) or [{}])[-1]
            metrics_file = run_dir / "metrics.jsonl"
            try:
                # The killed trainer may have left half a line: start the stop record on its own.
                with open(metrics_file, "rb") as fh:
                    fh.seek(-1, os.SEEK_END)
                    lead = "" if fh.read(1) == b"\n" else "\n"
            except OSError:
                lead = ""
            with open(metrics_file, "a", encoding="utf-8") as fh:
                fh.write(lead + json.dumps({"kind": "session", "event": "stop", "reason": "killed from the app", "finished": False,
                                            "time": time.time(), "step": last.get("step", 0), "games": last.get("games", 0),
                                            "positions": last.get("positions", 0), "hours": last.get("hours", 0.0)}) + "\n")
        else:
            (run_dir / "STOP").write_text("", encoding="utf-8")

    def resume(self, name: str, hours: float | None = None, max_games: int | None = None, overrides: dict | None = None) -> dict:
        run_dir = self.path(name)
        if not run_dir.is_dir():
            raise RunError(f"there is no run named {name!r}")
        overrides = {} if overrides is None else overrides
        if not isinstance(overrides, dict):
            raise RunError("overrides must be an object of setting: value")
        if hours is not None and (isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 <= hours <= 100_000):
            raise RunError("hours must be a number, 0 or more (0 means no time limit)")
        if max_games is not None and (isinstance(max_games, bool) or not isinstance(max_games, int) or max_games < 0):
            raise RunError("max_games must be a whole number, 0 or more (0 means no game limit)")
        with self.lock:
            status, _ = self._status(run_dir, self.records(name))
            if status in ("running", "stopping", "queued"):
                raise RunError(f"{name} is already {status}")
            has_checkpoint = (run_dir / "checkpoints" / "latest.txt").exists()
            if has_checkpoint:
                self._check_overrides(overrides, self.config(name))
                locked = sorted(k for k in overrides if k.startswith(LOCKED_ON_RESUME))
                if locked:
                    raise RunError(f"{locked[0]} is fixed by the run's checkpoints and cannot change on resume")
                # Without hours the session gets the run's own time limit again (it is in the checkpoint).
                arguments = override_arguments(overrides)
                if hours is not None:
                    arguments += ["--hours", str(hours)]
                if max_games is not None:
                    arguments += ["--set", f"run.max_games={int(max_games)}"]
                arguments += ["--resume", "latest"]
                remembered = None
            else:
                # Never got as far as a checkpoint: start it again as the run that was asked for,
                # with anything given now on top, and remember the result in case it dies again.
                stored = {k: v for k, v in (self._stored_overrides(run_dir) or flatten(self.config(name))).items()
                          if k not in RESERVED_SETTINGS}
                remembered = {**stored, **overrides}
                if hours is not None:
                    remembered["run.hours"] = hours
                if max_games is not None:
                    remembered["run.max_games"] = int(max_games)
                self._check_overrides(remembered)
                arguments = override_arguments(remembered)
            (run_dir / "running.json").unlink(missing_ok=True)
            self._launch(name, arguments, remembered)
        return self.summary(name)

    def delete(self, name: str) -> None:
        run_dir = self.path(name)
        if not run_dir.is_dir():
            raise RunError(f"there is no run named {name!r}")
        with self.lock:
            status, _ = self._status(run_dir, self.records(name))
            if status in ("running", "stopping"):
                raise RunError(f"{name} is running; stop it first")
            if run_dir.resolve().parent != self.runs_dir.resolve():
                raise RunError("refusing to delete outside the runs folder")
            # A process that has just finished can hold its log open for a moment longer, and
            # Windows will not delete an open file: give it a few seconds before giving up.
            for attempt in range(20):
                try:
                    shutil.rmtree(run_dir)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    if attempt == 19:
                        raise RunError(f"{name} still has files open (its process may be closing down); "
                                       "try again in a moment") from None
                    time.sleep(0.25)
            self._metrics.pop(name, None)

    def rename(self, name: str, new: str) -> None:
        """Rename a run: its folder, the name in its config.toml, and what is cached under the
        name. The stars, notes and matches of its models are the caller's to carry over (the
        server's `_rename_run` does).

        The checkpoint files keep the old name in the settings saved inside them, and are left
        alone: the app always resumes a run with `--name <run>`, which overrides that name, and
        the trainer then writes the new one into config.toml and every later checkpoint."""
        run_dir = self.path(name)
        if not run_dir.is_dir():
            raise RunError(f"there is no run named {name!r}")
        if not isinstance(new, str):
            raise RunError("the new name must be text")
        target = self.path(new)
        with self.lock:
            # Everything that can refuse comes before the folder moves.
            names = self.names()
            # The folder's own spelling (Windows also finds runs/night1 as runs/Night1).
            current = name if name in names else next((n for n in names if n.lower() == name.lower()), name)
            run_dir = self.runs_dir / current
            if new == current:
                raise RunError(f"{current} already has that name")
            if new.lower() != current.lower():  # a change of case only is allowed
                taken = next((n for n in names if n.lower() == new.lower()), None)
                if taken is not None or target.exists():
                    raise RunError(f"a run named {(taken or new)!r} already exists")
            status, _ = self._status(run_dir, self.records(current))
            if status in ("running", "stopping", "queued"):
                advice = {"running": "stop it first", "stopping": "wait until it has stopped",
                          "queued": "take it out of the queue first"}[status]
                raise RunError(f"{current} is {status}; {advice}")
            if run_dir.resolve().parent != self.runs_dir.resolve() or target.resolve().parent != self.runs_dir.resolve():
                raise RunError("refusing to rename outside the runs folder")
            try:
                config_text = _renamed_config((run_dir / "config.toml").read_bytes().decode("utf-8"), new)
            except (OSError, UnicodeDecodeError):
                config_text = None  # a run that never started has none yet
            _rename_folder(run_dir, target)
            if config_text is not None:
                temporary = target / "config.toml.tmp"
                try:
                    temporary.write_bytes(config_text.encode("utf-8"))
                    os.replace(temporary, target / "config.toml")
                except OSError as error:
                    temporary.unlink(missing_ok=True)
                    _rename_folder(target, run_dir)  # back as it was
                    raise RunError(f"{current} was not renamed: its config.toml could not be written ({error})") from None
            # Neither name may serve the old folder's records from the cache: a new run can take the old name.
            for key in [key for key in self._metrics if key.lower() in (current.lower(), new.lower())]:
                del self._metrics[key]


def _rename_folder(source: Path, target: Path) -> None:
    """Rename a run folder. A process that has just finished can hold a file in it open for a
    moment longer, and Windows will not rename a folder with an open file in it: give it a few
    seconds before giving up. A change of case only goes through a temporary name, since where
    folder names ignore case a direct rename can do nothing or fail."""
    if source.name.lower() == target.name.lower():
        temporary = source.with_name(f".renaming-{os.urandom(4).hex()}")  # not a run name: never listed
        _rename_folder(source, temporary)
        try:
            _rename_folder(temporary, target)
        except RunError:
            _rename_folder(temporary, source)
            raise
        return
    for attempt in range(20):
        try:
            os.rename(source, target)
            return
        except FileExistsError:
            raise RunError(f"a run named {target.name!r} already exists") from None
        except FileNotFoundError:
            raise RunError(f"there is no run named {source.name!r}") from None
        except OSError:
            if attempt == 19:
                raise RunError(f"{source.name} still has files open (its process may be closing down); "
                               "try again in a moment") from None
            time.sleep(0.25)


_TOML_TABLE = re.compile(r"^[ \t]*\[([^\[\]\r\n]+)\][ \t]*(?:#[^\r\n]*)?\r?$", re.M)
_TOML_NAME = re.compile(r"""^([ \t]*name[ \t]*=[ \t]*)("(?:[^"\\\r\n]|\\.)*"|'[^'\r\n]*')""", re.M)


def _renamed_config(text: str, new: str) -> str | None:
    """A config.toml with the `name` under `[run]` set to `new` and every other byte as it was,
    or None when it has no such line or the result would not read back as the same settings
    with the new name (it is then left as it is: the app passes the name to the trainer)."""
    tables = list(_TOML_TABLE.finditer(text))
    run = next((i for i, table in enumerate(tables) if table[1].strip() == "run"), None)
    if run is None:
        return None
    end = tables[run + 1].start() if run + 1 < len(tables) else len(text)
    line = _TOML_NAME.search(text, tables[run].end(), end)
    if line is None:
        return None
    result = text[:line.start(2)] + json.dumps(new) + text[line.end(2):]
    try:
        expected, renamed = tomllib.loads(text), tomllib.loads(result)
    except tomllib.TOMLDecodeError:
        return None
    if not isinstance(expected.get("run"), dict):
        return None
    expected["run"]["name"] = new
    return result if renamed == expected else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _games_per_second(records: list[dict], selfplay: list[dict], live: bool) -> float | None:
    """The latest rate while running, the run's average afterwards. Before the first self-play
    report arrives (they come once a minute) it is worked out from the game counter."""
    if selfplay:
        return selfplay[-1].get("games_per_second") if live else _mean([r.get("games_per_second") for r in selfplay])
    counted = [r for r in records if r.get("games", 0) > 0]
    began = next((r for r in records if r.get("kind") == "session" and r.get("event") == "selfplay"), None)
    if began is None or not counted or counted[-1]["time"] - began["time"] < 5:
        return None
    return (counted[-1]["games"] - began["games"]) / (counted[-1]["time"] - began["time"])


def _mean(values: list) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None
