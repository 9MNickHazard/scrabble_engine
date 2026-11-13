"""Shared fixtures and helpers for the app tests (tests/test_app_*.py).

Test modules load them with `pytest_plugins = ["app_fixtures"]`, so pytest registers them once
and the session-scoped training run below happens once per test session.

Nothing here touches the repo's runs/ or data/ folders, starts a detached process or kills one:
`spawned` replaces `spawn_detached` (in runs.py and in the name matches.py imported) with a
recorder that returns a fake pid, and `kill_process` with one that only records the pid.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scrabble_ai import engine as engine_module
from scrabble_ai.lexicon import LexiconError

# A pid no process can have: Windows pids are multiples of 4 well below this. (A pid that is not
# a multiple of 4 is NOT safe: Windows ignores the low two bits and may open a live process.)
DEAD_PID = 1_999_999_996

TINY_RUN = "tiny"


def lexicon_entry_for(lexicon_dir: Path):
    """A stand-in for `engine.lexicon_entry` that resolves names to the test word lists."""

    def lexicon_entry(name: str = "", config_path=None):
        name = name or "TEST"
        path = lexicon_dir / f"{name}.txt"
        if not path.exists():
            raise LexiconError(f"unknown lexicon {name!r}")
        return name, path

    return lexicon_entry


def tiny_settings(out_dir: Path, name: str, leaves_file: Path) -> dict:
    """The smallest settings that train in seconds on the CPU (as in tests/test_train.py)."""
    return {
        "run.name": name, "run.out_dir": str(out_dir), "run.lexicon": "TEST", "run.leaves": str(leaves_file),
        "run.device": "cpu", "run.seed": 3, "run.max_games": 24,
        "model.blocks": 1, "model.channels": 16, "model.vector_hidden": 16, "model.head_hidden": 16,
        "model.value_channels": 2,
        "selfplay.games": 8, "selfplay.max_candidates": 8, "selfplay.max_exchanges": 2,
        "selfplay.temperature_start": 2.0, "selfplay.temperature_end": 0.5, "selfplay.temperature_decay_games": 100,
        "warmstart.games": 16, "warmstart.max_candidates": 8, "warmstart.steps": 6, "warmstart.temperature": 4.0,
        "train.batch_size": 32, "train.min_replay": 64, "train.replay_capacity": 20_000, "train.replay_ratio": 0.5,
        "train.actor_sync_steps": 4, "train.warmup_steps": 4, "train.eval_pairs": 2, "train.gate_pairs": 2,
        "train.checkpoint_minutes": 60.0, "train.eval_minutes": 60.0, "train.keep_checkpoints": 2,
    }


@pytest.fixture(scope="session")
def trained(lexicon_dir, leaves, tmp_path_factory):
    """A genuine run folder: a 1x16 network trained in-process on the CPU for a few dozen games."""
    from scrabble_ai import train as train_module
    from scrabble_ai.config import load_config
    from scrabble_ai.selfplay import Scorer
    from scrabble_ai.train import Trainer

    root = tmp_path_factory.mktemp("app_trained")
    fitted = root / "fitted.leaves"
    leaves.save(fitted)
    settings = tiny_settings(root, TINY_RUN, fitted)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "lexicon_entry", lexicon_entry_for(lexicon_dir))
        patch.setattr(Scorer, "PAD", 32)
        patch.setattr(train_module, "git_commit", lambda: "test")  # the trainer would run `git rev-parse`
        config = load_config(None, [f"{k}={v}" for k, v in settings.items()])
        trainer = Trainer(config, quiet=True)
        trainer.run()
    return SimpleNamespace(root=root, run_dir=root / TINY_RUN, loaded=trainer.loaded, fitted=fitted, settings=settings,
                           config=config, step=trainer.step, best_step=trainer.best_step, games=trainer.games)


def copy_run(trained, runs_dir: Path, name: str = TINY_RUN) -> Path:
    """Copy the trained run (without its replay buffer and TensorBoard files) into `runs_dir`."""
    target = runs_dir / name
    shutil.copytree(trained.run_dir, target, ignore=shutil.ignore_patterns("replay", "replay.new", "tb"))
    return target


class Spawner:
    """Records what would have been started detached, and what would have been killed."""

    def __init__(self):
        self.calls: list[tuple[list[str], Path]] = []
        self.killed: list[int] = []
        self.pid = os.getpid()  # alive, so a started run reads as running
        self.on_spawn = None

    def __call__(self, command, log_path):
        if self.on_spawn is not None:
            self.on_spawn(list(command), Path(log_path))
        self.calls.append((list(command), Path(log_path)))
        return self.pid

    def kill(self, pid: int) -> None:
        self.killed.append(pid)


@pytest.fixture
def spawned(monkeypatch):
    from scrabble_ai.app import matches as matches_module
    from scrabble_ai.app import runs as runs_module

    spawner = Spawner()
    monkeypatch.setattr(runs_module, "spawn_detached", spawner)
    monkeypatch.setattr(matches_module, "spawn_detached", spawner)
    monkeypatch.setattr(runs_module, "kill_process", spawner.kill)
    return spawner


@pytest.fixture
def test_lexicon(monkeypatch, lexicon_dir):
    """Point `engine.lexicon_entry` at the test word lists (RunManager.start checks the lexicon)."""
    monkeypatch.setattr(engine_module, "lexicon_entry", lexicon_entry_for(lexicon_dir))


@pytest.fixture
def cpu_players(trained):
    """A factory for `Players` pinned to the CPU (the GPU is busy; Players would pick CUDA)."""
    from scrabble_ai.app.players import Players

    def make(runs_dir: Path, cache_size: int = 4) -> Players:
        players = Players(trained.loaded, runs_dir, cache_size)
        players._device = torch.device("cpu")
        return players

    return make


@pytest.fixture
def fast_scorer(monkeypatch):
    """Batches padded to 32 rows instead of 512: the same numbers, much less CPU time."""
    from scrabble_ai.selfplay import Scorer

    monkeypatch.setattr(Scorer, "PAD", 32)


# ---- crafting run folders -----------------------------------------------------------------------


def record(kind: str, t: float, step: int = 0, games: int = 0, positions: int = 0, hours: float = 0.0, **extra) -> dict:
    return {"kind": kind, "time": t, "step": step, "games": games, "positions": positions, "hours": hours, **extra}


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def write_config(run_dir: Path, settings: dict) -> None:
    from scrabble_ai.config import load_config
    from scrabble_ai.train import config_toml

    items = [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in settings.items()]
    (run_dir / "config.toml").write_text(config_toml(load_config(None, items)), encoding="utf-8")


def make_run(runs_dir: Path, name: str, records: list[dict] | None = None, settings: dict | None = None,
             pid: int | None = None, stop: bool = False, queued: dict | None = None, console: list[str] | None = None,
             index: dict | None = None, latest: str | None = None) -> Path:
    """A run folder by hand. `pid` writes running.json, `queued` writes queued.json as given."""
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if records is not None:
        write_records(run_dir / "metrics.jsonl", records)
    if settings is not None:
        write_config(run_dir, settings)
    if pid is not None:
        (run_dir / "running.json").write_text(json.dumps({"pid": pid, "started": time.time()}), encoding="utf-8")
    if stop:
        (run_dir / "STOP").write_text("", encoding="utf-8")
    if queued is not None:
        (run_dir / "queued.json").write_text(json.dumps(queued), encoding="utf-8")
    if console is not None:
        (run_dir / "console.log").write_text("".join(line + "\n" for line in console), encoding="utf-8")
    if index is not None or latest is not None:
        (run_dir / "checkpoints").mkdir(exist_ok=True)
    if index is not None:
        (run_dir / "checkpoints" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    if latest is not None:
        (run_dir / "checkpoints" / "latest.txt").write_text(latest, encoding="utf-8")
        (run_dir / "checkpoints" / latest).write_bytes(b"not really a checkpoint")
    return run_dir
