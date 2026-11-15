"""Training configuration.

Every setting that affects training lives here as a dataclass field with its default.
`configs/train.toml` lists the same settings with explanations; a config file only needs the
values it changes. Unknown sections or keys are errors, so a typo cannot silently do nothing.

Ways to change a setting, later ones winning:
    1. the default below
    2. the TOML file passed with --config
    3. --set section.key=value on the command line (repeatable)
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass
class RunConfig:
    name: str = "run"  # outputs go to <out_dir>/<name>/
    out_dir: str = "runs"
    lexicon: str = ""  # a name from configs/lexicons.toml; empty = that file's default
    leaves: str = ""  # leave-value file; empty = data/leaves/<lexicon id>.leaves
    seed: int = 1  # fixes self-play game seeds and all sampling
    hours: float = 0.0  # stop after this much wall-clock time (0 = no time limit)
    max_games: int = 0  # stop after this many self-play games (0 = no limit)
    device: str = "cuda"
    vram_fraction: float = 0.9  # cap on PyTorch's share of GPU memory (Windows spills silently otherwise)


@dataclass
class ModelConfig:
    blocks: int = 6  # residual blocks
    channels: int = 64  # channels per block
    norm: str = "none"  # "none" (residual scaling, no normalization) or "group" (GroupNorm)
    vector_hidden: int = 128  # width of the MLP over the non-board features
    head_hidden: int = 256  # width of the layer before the value and win outputs
    value_channels: int = 8  # 1x1 conv channels flattened into the head (keeps positions)
    cross_checks: bool = True  # feed the cross-check planes (which letters fit on each square)
    leave_baseline: bool = True  # network predicts a correction to the static leave value
    # Inputs added on 2026-11-14 (features.py). A checkpoint from before then has none of them.
    extra_scalars: bool = True  # the move's score, the opponent's bingo history, make-up of the unseen pool and the leave
    last_move_plane: bool = True  # where the opponent's previous move put its tiles
    openness: bool = True  # which squares the next player can reach, how easily, and where seven tiles fit
    hooks: bool = True  # where one letter extends a word on the board, and where an S does
    belief_samples: int = 4  # racks sampled per placement to infer what the opponent kept (0 = no such input)
    # Outputs added on 2026-11-15 for training from saved games (offline.py). The defaults are
    # what every earlier network has.
    kind: str = "cnn"  # the body of the network: "cnn", "cnn-global" (with board-wide pooling layers) or "transformer"
    head: str = "flat"  # "flat": the head reads a few channels of every square; "pooled": only board-wide means and maxima
    win_head: str = "binary"  # "binary": one win logit; "wdl": win, draw or loss
    aux_heads: bool = False  # helper outputs for training: the opponent's next score and bingo, where the next moves land


@dataclass
class SelfPlayConfig:
    games: int = 256  # games played in lockstep
    max_candidates: int = 64  # moves per decision sent to the network (top by score + leave value)
    max_exchanges: int = 4  # of which at most this many are exchanges
    temperature_start: float = 4.0  # softmax temperature over move values, in points, at game 0
    temperature_end: float = 0.5  # temperature after temperature_decay_games
    temperature_decay_games: int = 300_000  # self-play games over which the temperature anneals
    win_weight: float = 0.0  # 0 = choose moves by expected spread only, 1 = by win probability only
    win_scale: float = 100.0  # points that certain victory is worth when blending the two


@dataclass
class WarmStartConfig:
    games: int = 100_000  # games played by the static evaluator before any network self-play (0 = skip)
    temperature: float = 6.0  # softmax temperature over static values, so bad moves are seen too
    max_candidates: int = 16
    steps: int = 2_000  # training steps on that data before self-play starts (under one pass over it)


@dataclass
class TrainConfig:
    batch_size: int = 1024
    learning_rate: float = 3e-4
    warmup_steps: int = 500  # linear learning-rate warm-up
    weight_decay: float = 1e-4
    grad_clip: float = 1.0  # max gradient norm (0 = off)
    value_scale: float = 50.0  # points per unit of the value output
    huber_delta: float = 2.0  # Huber loss transition, in value-output units
    win_loss_weight: float = 0.5  # weight of the win-probability loss
    target: str = "td"  # "td": learn from what followed in the game (TD(lambda)); "q": deep Q-learning
    td_lambda: float = 0.7  # 1 = Monte Carlo targets; below 1 mixes in the network's own later estimates
    replay_capacity: int = 3_000_000  # positions kept (about 310 bytes each)
    min_replay: int = 100_000  # positions needed before training starts
    replay_ratio: float = 1.0  # training samples drawn per new self-play position
    actor_sync_steps: int = 100  # training steps between copies of the weights into the self-play network
    guard: bool = True  # go back to the best network and halve the learning rate when training makes it worse
    guard_patience: int = 2  # evaluations in a row significantly behind the best before the guard acts
    augment_transpose: bool = True  # randomly swap rows and columns (the only exact board symmetry)
    amp: bool = True  # bf16 autocast
    checkpoint_minutes: float = 10.0
    eval_minutes: float = 20.0
    eval_pairs: int = 250  # seat-swapped pairs per periodic evaluation against each baseline
    gate_pairs: int = 500  # pairs when testing a checkpoint against the current best
    keep_checkpoints: int = 5


@dataclass
class OfflineConfig:
    """Training from saved games (`scrabble-ai train-games`, offline.py). The self-play trainer
    does not read this section."""

    games: str = ""  # the game set to learn from: a name under data/games, or a folder
    passes: float = 6.0  # times through the games; each pass takes one fresh position from every game
    batch_size: int = 2048
    learning_rate: float = 3e-4  # peak; falls along a cosine to a twentieth of it
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    validation_games: int = 20_000  # games held out of training, for measuring the loss on games never trained on
    validate_steps: int = 500  # training steps between those measurements
    aux_share: float = 0.15  # each helper output's pull on the network as a share of the main output's (0 = fixed weights)
    spread_weight: float = 0.25  # fixed weight of the points output when aux_share is 0
    aux_weight: float = 0.1  # fixed weight of each other helper output when aux_share is 0
    eval_minutes: float = 30.0  # how often to play the static evaluator
    eval_pairs: int = 2000  # seat-swapped pairs in those matches (about +/- 1.3 points of win rate)
    final_pairs: int = 10_000  # pairs in the match at the end of the run (about +/- 0.6)
    checkpoint_minutes: float = 10.0
    keep_checkpoints: int = 3
    augment_transpose: bool = True
    amp: bool = True


# What `scrabble-ai train-games` changes in the other sections unless told otherwise: the network
# it trains predicts win, draw or loss from board-wide summaries, with helper outputs, and picks
# its move by win chance with points breaking ties.
OFFLINE_DEFAULTS: dict[str, Any] = {
    "model.head": "pooled",
    "model.win_head": "wdl",
    "model.aux_heads": True,
    "selfplay.win_weight": 0.9,
    "selfplay.max_candidates": 32,
}


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    warmstart: WarmStartConfig = field(default_factory=WarmStartConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        checks = [
            (self.model.norm in ("none", "group"), "model.norm must be 'none' or 'group'"),
            (self.model.blocks >= 1, "model.blocks must be at least 1"),
            (self.model.channels >= 8, "model.channels must be at least 8"),
            (self.model.channels % 8 == 0, "model.channels must be a multiple of 8"),
            (0 <= self.model.belief_samples <= 64, "model.belief_samples must be between 0 and 64"),
            (self.model.kind in ("cnn", "cnn-global", "transformer"), "model.kind must be 'cnn', 'cnn-global' or 'transformer'"),
            (self.model.head in ("flat", "pooled"), "model.head must be 'flat' or 'pooled'"),
            (self.model.win_head in ("binary", "wdl"), "model.win_head must be 'binary' or 'wdl'"),
            (self.offline.passes > 0, "offline.passes must be positive"),
            (self.offline.batch_size >= 1, "offline.batch_size must be at least 1"),
            (self.offline.learning_rate > 0, "offline.learning_rate must be positive"),
            (self.offline.validation_games >= 0, "offline.validation_games cannot be negative"),
            (self.offline.validate_steps >= 1, "offline.validate_steps must be at least 1"),
            (0.0 <= self.offline.aux_share <= 1.0, "offline.aux_share must be between 0 and 1"),
            (self.offline.eval_pairs >= 2 and self.offline.final_pairs >= 2, "offline.eval_pairs and offline.final_pairs must be at least 2"),
            (self.offline.keep_checkpoints >= 1, "offline.keep_checkpoints must be at least 1"),
            (self.selfplay.games >= 1, "selfplay.games must be at least 1"),
            (2 <= self.selfplay.max_candidates <= 512, "selfplay.max_candidates must be between 2 and 512"),
            (0 <= self.selfplay.max_exchanges < self.selfplay.max_candidates,
             "selfplay.max_exchanges must be at least 0 and below selfplay.max_candidates"),
            (0.0 <= self.selfplay.win_weight <= 1.0, "selfplay.win_weight must be between 0 and 1"),
            (self.selfplay.temperature_start >= 0, "selfplay.temperature_start cannot be negative"),
            (self.selfplay.temperature_end >= 0, "selfplay.temperature_end cannot be negative"),
            (0.0 <= self.train.td_lambda <= 1.0, "train.td_lambda must be between 0 and 1"),
            (self.train.target in ("td", "q"), "train.target must be 'td' or 'q'"),
            (self.train.batch_size >= 1, "train.batch_size must be at least 1"),
            (self.train.learning_rate > 0, "train.learning_rate must be positive"),
            (self.train.min_replay >= self.train.batch_size, "train.min_replay must be at least train.batch_size"),
            (self.train.replay_capacity >= self.train.min_replay, "train.replay_capacity must be at least train.min_replay"),
            (self.train.value_scale > 0, "train.value_scale must be positive"),
            (self.train.huber_delta > 0, "train.huber_delta must be positive"),
            (0.0 < self.run.vram_fraction <= 1.0, "run.vram_fraction must be above 0 and at most 1"),
            (self.run.hours >= 0, "run.hours cannot be negative"),
            (self.run.max_games >= 0, "run.max_games cannot be negative"),
            (self.selfplay.temperature_decay_games >= 1, "selfplay.temperature_decay_games must be at least 1"),
            (self.selfplay.win_scale > 0, "selfplay.win_scale must be positive"),
            (self.warmstart.games >= 0, "warmstart.games cannot be negative"),
            (self.warmstart.steps >= 0, "warmstart.steps cannot be negative"),
            (self.warmstart.max_candidates >= 2, "warmstart.max_candidates must be at least 2"),
            (self.warmstart.temperature >= 0, "warmstart.temperature cannot be negative"),
            (self.train.actor_sync_steps >= 1, "train.actor_sync_steps must be at least 1"),
            (self.train.replay_ratio > 0, "train.replay_ratio must be positive"),
            (self.train.warmup_steps >= 0, "train.warmup_steps cannot be negative"),
            (self.train.weight_decay >= 0, "train.weight_decay cannot be negative"),
            (self.train.grad_clip >= 0, "train.grad_clip cannot be negative"),
            (self.train.win_loss_weight >= 0, "train.win_loss_weight cannot be negative"),
            (self.train.eval_pairs >= 2, "train.eval_pairs must be at least 2 (an interval needs two pairs)"),
            (self.train.gate_pairs >= 2, "train.gate_pairs must be at least 2 (an interval needs two pairs)"),
            (self.train.checkpoint_minutes > 0, "train.checkpoint_minutes must be positive"),
            (self.train.eval_minutes > 0, "train.eval_minutes must be positive"),
            (self.train.keep_checkpoints >= 1, "train.keep_checkpoints must be at least 1"),
            (self.train.guard_patience >= 1, "train.guard_patience must be at least 1"),
        ]
        for ok, message in checks:
            if not ok:
                raise ConfigError(message)


def _coerce(value: Any, target: type, where: str) -> Any:
    """Check a TOML value against a field's type; ints are accepted for float fields."""
    if target is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if target is int and isinstance(value, bool):
        raise ConfigError(f"{where}: expected a whole number, got {value!r}")
    if not isinstance(value, target):
        raise ConfigError(f"{where}: expected {target.__name__}, got {value!r}")
    return value


def _parse_scalar(text: str, target: type, where: str) -> Any:
    """Parse a --set value for a field of the given type."""
    try:
        if target is bool:
            if text.lower() in ("true", "1", "yes", "on"):
                return True
            if text.lower() in ("false", "0", "no", "off"):
                return False
            raise ValueError
        if target is int:
            return int(text.replace("_", ""))
        if target is float:
            return float(text.replace("_", ""))
        return text
    except ValueError:
        raise ConfigError(f"{where}: cannot read {text!r} as {target.__name__}") from None


_FIELD_TYPES = {"int": int, "float": float, "bool": bool, "str": str}


def _field_type(section: Any, key: str) -> type:
    for f in dataclasses.fields(section):
        if f.name == key:
            return _FIELD_TYPES[f.type] if isinstance(f.type, str) else f.type
    raise KeyError(key)


def apply_dict(config: Config, data: dict[str, Any]) -> None:
    for section_name, values in data.items():
        section = getattr(config, section_name, None)
        if section is None or not dataclasses.is_dataclass(section):
            known = ", ".join(f.name for f in dataclasses.fields(config))
            raise ConfigError(f"unknown config section [{section_name}]; sections are: {known}")
        if not isinstance(values, dict):
            raise ConfigError(f"[{section_name}] must be a table of settings")
        for key, value in values.items():
            where = f"{section_name}.{key}"
            try:
                target = _field_type(section, key)
            except KeyError:
                known = ", ".join(f.name for f in dataclasses.fields(section))
                raise ConfigError(f"unknown setting {where}; [{section_name}] has: {known}") from None
            setattr(section, key, _coerce(value, target, where))


def apply_overrides(config: Config, overrides: list[str]) -> None:
    """Apply `section.key=value` strings from the command line."""
    for item in overrides:
        path, separator, text = item.partition("=")
        section_name, dot, key = path.strip().partition(".")
        if not separator or not dot:
            raise ConfigError(f"--set expects section.key=value, got {item!r}")
        section = getattr(config, section_name, None)
        if section is None or not dataclasses.is_dataclass(section):
            raise ConfigError(f"unknown config section {section_name!r} in --set {item!r}")
        try:
            target = _field_type(section, key)
        except KeyError:
            known = ", ".join(f.name for f in dataclasses.fields(section))
            raise ConfigError(f"unknown setting {path}; [{section_name}] has: {known}") from None
        setattr(section, key, _parse_scalar(text.strip(), target, path))


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> Config:
    config = Config()
    if path is not None:
        with open(path, "rb") as fh:
            apply_dict(config, tomllib.load(fh))
    apply_overrides(config, overrides or [])
    config.validate()
    return config


# Settings a checkpoint may not mention because they did not exist when it was written, with the
# value that describes how things were then. (A missing setting cannot mean today's default: a
# network saved before an input existed does not have that input.)
BEFORE_IT_EXISTED: dict[str, dict[str, Any]] = {
    "model": {"extra_scalars": False, "last_move_plane": False, "openness": False, "hooks": False, "belief_samples": 0,
              "kind": "cnn", "head": "flat", "win_head": "binary", "aux_heads": False},
}


def config_from_dict(data: dict[str, Any]) -> Config:
    """Rebuild a config saved in a checkpoint."""
    config = Config()
    for section, earlier in BEFORE_IT_EXISTED.items():
        saved = data.get(section)
        if isinstance(saved, dict):
            data = {**data, section: {**{key: value for key, value in earlier.items() if key not in saved}, **saved}}
    apply_dict(config, data)
    config.validate()
    return config
