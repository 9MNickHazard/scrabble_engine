"""Who can play: the built-in bots and the trained models.

A player is named by a spec string: `human`, `bot:greedy-leave`, `bot:greedy`, `bot:random`, or
`model:<run>/<checkpoint file>`. This module turns specs into display names and, for models,
into a loaded network ready to score positions.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from pathlib import Path

from scrabble_ai import engine as engine_module

BOTS = {
    "bot:greedy-leave": ("Greedy + leave", "Plays the move with the highest score plus leave value. The baseline to beat."),
    "bot:greedy": ("Greedy", "Plays the highest-scoring move."),
    "bot:random": ("Random", "Plays a random legal placement."),
}
MODEL_FILE = re.compile(r"(best|step_\d{9})\.pt")  # used with fullmatch
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class PlayerError(ValueError):
    """A request about players, games or matches that cannot be carried out (a 400)."""


class NotFound(KeyError):
    """A run, game, match or model that does not exist (a 404)."""

    def __str__(self) -> str:  # KeyError would put the message in quotes
        return str(self.args[0]) if self.args else "not found"


def kind(spec: str) -> str:
    if not isinstance(spec, str):
        raise PlayerError("a player is named by a string such as bot:greedy or model:<run>/best.pt")
    if spec == "human":
        return "human"
    if spec in BOTS:
        return "bot"
    if spec.startswith("model:"):
        return "model"
    raise PlayerError(f"unknown player {spec!r}")


def bot_name(spec: str) -> str:
    """The engine's name for a bot spec: `bot:greedy-leave` -> `greedy-leave`."""
    return spec.split(":", 1)[1]


def split_model(spec: str) -> tuple[str, str]:
    model_id = spec.split(":", 1)[1] if spec.startswith("model:") else spec
    run, _, file = model_id.partition("/")
    if not RUN_NAME.fullmatch(run) or not MODEL_FILE.fullmatch(file):
        raise PlayerError(f"{model_id!r} is not a model id of the form <run>/best.pt or <run>/step_<n>.pt")
    return run, file


def model_label(run: str, file: str) -> str:
    return f"{run} (best)" if file == "best.pt" else f"{run} step {int(file[5:14]):,}"


def name(spec: str) -> str:
    which = kind(spec)
    if which == "human":
        return "You"
    if which == "bot":
        return BOTS[spec][0]
    return model_label(*split_model(spec))


class Players:
    """Loads and caches the networks behind model specs."""

    def __init__(self, loaded: engine_module.Loaded, runs_dir: Path, cache_size: int = 4):
        self.loaded = loaded
        self.runs_dir = runs_dir
        self.cache_size = cache_size
        self.lock = threading.Lock()
        self._scorers: OrderedDict[tuple[str, float], object] = OrderedDict()
        self._device = None

    @property
    def device(self):
        if self._device is None:
            import torch

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def model_path(self, spec: str) -> Path:
        run, file = split_model(spec)
        path = self.runs_dir / run / "checkpoints" / file
        if not path.exists():
            raise PlayerError(f"{model_label(run, file)} no longer exists")
        return path

    def scorer(self, spec: str):
        """The network behind a model spec as a `Scorer`. Reloaded if the file has changed (a
        run's best.pt is replaced whenever a better network is promoted)."""
        from scrabble_ai.selfplay import Scorer
        from scrabble_ai.train import load_network

        path = self.model_path(spec)
        key = (str(path), path.stat().st_mtime)
        with self.lock:
            if key in self._scorers:
                self._scorers.move_to_end(key)
                return self._scorers[key]
            net, state = load_network(path, self.device)
            if state["lexicon_id"] != self.loaded.lexicon_id:
                raise PlayerError(f"{name(spec)} was trained with the word list {state['lexicon_id']}, "
                                  f"not the installed {self.loaded.lexicon_id}")
            if state["leaves_hash"] != self.loaded.leaves_hash:
                raise PlayerError(f"{name(spec)} was trained with different leave values than the ones installed")
            scorer = Scorer(net, self.loaded.engine, state["config"]["train"]["value_scale"], self.device)
            self._scorers[key] = scorer
            while len(self._scorers) > self.cache_size:
                self._scorers.popitem(last=False)
            return scorer
