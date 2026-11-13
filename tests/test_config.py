"""`config`: defaults, TOML loading, --set parsing and validation."""

from __future__ import annotations

import dataclasses
import tomllib

import pytest

from scrabble_ai.config import Config, ConfigError, apply_dict, apply_overrides, config_from_dict, load_config
from scrabble_ai.lexicon import REPO_ROOT

TRAIN_TOML = REPO_ROOT / "configs" / "train.toml"


def test_train_toml_lists_every_setting_with_its_default():
    with open(TRAIN_TOML, "rb") as fh:
        documented = tomllib.load(fh)
    defaults = Config().to_dict()
    assert set(documented) == set(defaults), "sections"
    for section, values in defaults.items():
        assert set(documented[section]) == set(values), f"keys of [{section}]"
        for key, default in values.items():
            value = documented[section][key]
            assert value == default and type(value) is type(default) or (
                isinstance(default, float) and isinstance(value, int) and value == default
            ), f"{section}.{key}: file says {value!r}, default is {default!r}"
    # And loading the file changes nothing.
    assert load_config(TRAIN_TOML) == Config()


def test_defaults_validate():
    Config().validate()


def test_unknown_section_and_key(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[modle]\nblocks = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown config section \[modle\]"):
        load_config(path)
    path.write_text("[model]\nblokcs = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown setting model.blokcs"):
        load_config(path)
    path.write_text("blocks = 2\n", encoding="utf-8")  # a setting outside any section
    with pytest.raises(ConfigError, match=r"unknown config section \[blocks\]"):
        load_config(path)
    path.write_text("model = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a table"):
        load_config(path)


@pytest.mark.parametrize("text", [
    'blocks = "6"',  # string for an int
    "blocks = true",  # bool for an int
    "blocks = 6.0",  # float for an int
    "cross_checks = 1",  # int for a bool
    'cross_checks = "true"',
    "norm = 3",  # int for a str
])
def test_type_errors_in_files(tmp_path, text):
    path = tmp_path / "c.toml"
    path.write_text(f"[model]\n{text}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="model."):
        load_config(path)


def test_float_fields_reject_bools_and_strings(tmp_path):
    for text in ("learning_rate = true", 'learning_rate = "0.1"'):
        path = tmp_path / "c.toml"
        path.write_text(f"[train]\n{text}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="train.learning_rate"):
            load_config(path)


def test_ints_are_accepted_for_float_fields(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[train]\nlearning_rate = 1\nvalue_scale = 40\n[run]\nhours = 2\n", encoding="utf-8")
    config = load_config(path)
    assert config.train.learning_rate == 1.0 and type(config.train.learning_rate) is float
    assert config.train.value_scale == 40.0 and type(config.train.value_scale) is float
    assert config.run.hours == 2.0 and type(config.run.hours) is float


@pytest.mark.parametrize("item, section, key, expected", [
    ("train.learning_rate=1e-3", "train", "learning_rate", 1e-3),
    ("train.learning_rate=5e-4", "train", "learning_rate", 5e-4),
    ("train.learning_rate=2", "train", "learning_rate", 2.0),
    ("train.replay_capacity=300_000", "train", "replay_capacity", 300_000),
    ("selfplay.temperature_decay_games=1_000_000", "selfplay", "temperature_decay_games", 1_000_000),
    ("model.blocks=10", "model", "blocks", 10),
    ("model.cross_checks=false", "model", "cross_checks", False),
    ("model.cross_checks=False", "model", "cross_checks", False),
    ("model.cross_checks=0", "model", "cross_checks", False),
    ("model.leave_baseline=true", "model", "leave_baseline", True),
    ("model.leave_baseline=on", "model", "leave_baseline", True),
    ("run.name=my-run", "run", "name", "my-run"),
    ("run.leaves=zero", "run", "leaves", "zero"),
    ("run.out_dir=C:/tmp/runs", "run", "out_dir", "C:/tmp/runs"),
    (" train.batch_size = 64 ", "train", "batch_size", 64),
    ("run.lexicon=", "run", "lexicon", ""),
])
def test_set_parsing(item, section, key, expected):
    config = Config()
    apply_overrides(config, [item])
    value = getattr(getattr(config, section), key)
    assert value == expected and type(value) is type(expected)


@pytest.mark.parametrize("item, message", [
    ("model.blocks=1e3", "cannot read"),
    ("model.blocks=6.0", "cannot read"),
    ("model.blocks=six", "cannot read"),
    ("model.cross_checks=maybe", "cannot read"),
    ("train.learning_rate=fast", "cannot read"),
    ("model.blocks", "section.key=value"),
    ("blocks=3", "section.key=value"),
    ("modle.blocks=3", "unknown config section"),
    ("model.blokcs=3", "unknown setting"),
])
def test_set_errors(item, message):
    with pytest.raises(ConfigError, match=message):
        apply_overrides(Config(), [item])


def test_later_sources_win(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[model]\nblocks = 3\nchannels = 32\n", encoding="utf-8")
    config = load_config(path, ["model.blocks=4", "model.blocks=5"])
    assert (config.model.blocks, config.model.channels) == (5, 32)


@pytest.mark.parametrize("item, message", [
    ("model.norm=batch", "model.norm"),
    ("model.blocks=0", "model.blocks must be at least 1"),
    ("model.channels=4", "model.channels must be at least 8"),
    ("model.channels=20", "multiple of 8"),
    ("selfplay.games=0", "selfplay.games"),
    ("selfplay.max_candidates=1", "max_candidates"),
    ("selfplay.max_candidates=513", "max_candidates"),
    ("selfplay.max_exchanges=64", "max_exchanges"),
    ("selfplay.max_exchanges=-1", "max_exchanges"),
    ("selfplay.win_weight=1.5", "win_weight"),
    ("selfplay.win_weight=-0.1", "win_weight"),
    ("selfplay.temperature_start=-1", "selfplay.temperature_start"),
    ("selfplay.temperature_end=-0.5", "selfplay.temperature_end"),
    ("train.td_lambda=1.1", "td_lambda"),
    ("train.td_lambda=-0.1", "td_lambda"),
    ("train.batch_size=0", "batch_size"),
    ("train.learning_rate=0", "learning_rate"),
    ("train.guard_patience=0", "train.guard_patience must be at least 1"),
    ("train.min_replay=10", "train.min_replay must be at least train.batch_size"),
    ("train.replay_capacity=50000", "train.replay_capacity must be at least train.min_replay"),
    ("train.value_scale=0", "value_scale"),
    ("train.huber_delta=-1", "huber_delta"),
    ("run.vram_fraction=0", "vram_fraction"),
    ("run.vram_fraction=1.5", "vram_fraction"),
])
def test_validate_rejections(item, message):
    with pytest.raises(ConfigError, match=message):
        load_config(None, [item])


@pytest.mark.parametrize("item", ["selfplay.max_candidates=2", "selfplay.max_exchanges=0", "train.td_lambda=0",
                                  "selfplay.win_weight=1", "run.vram_fraction=1", "model.channels=8"])
def test_validate_accepts_the_boundaries(item):
    overrides = [item] + (["selfplay.max_exchanges=1"] if item == "selfplay.max_candidates=2" else [])
    load_config(None, overrides)


def test_validate_rejects_zero_actor_sync_steps():
    with pytest.raises(ConfigError):
        load_config(None, ["train.actor_sync_steps=0"])


def test_validate_rejects_a_warm_start_with_fewer_than_two_candidates():
    with pytest.raises(ConfigError):
        load_config(None, ["warmstart.max_candidates=1"])


def test_round_trip_through_a_dict():
    config = load_config(None, ["model.blocks=3", "train.learning_rate=2.5e-4", "run.name=x", "model.norm=group",
                                "selfplay.win_weight=0.25", "train.augment_transpose=false"])
    data = config.to_dict()
    assert config_from_dict(data) == config
    # Dicts from older checkpoints with missing keys get the defaults for them.
    del data["train"]["learning_rate"]
    assert config_from_dict(data).train.learning_rate == Config().train.learning_rate
    with pytest.raises(ConfigError):
        config_from_dict({"train": {"no_such_setting": 1}})


def test_apply_dict_leaves_other_values_alone():
    config = Config()
    apply_dict(config, {"model": {"blocks": 2}})
    expected = Config()
    expected.model.blocks = 2
    assert config == expected
    assert dataclasses.asdict(config)["model"]["blocks"] == 2


@pytest.mark.parametrize("item", [
    "train.eval_pairs=1", "train.gate_pairs=1", "train.keep_checkpoints=0", "train.replay_ratio=0",
    "train.checkpoint_minutes=0", "train.eval_minutes=0", "warmstart.games=-1", "warmstart.steps=-1",
    "selfplay.temperature_decay_games=0", "selfplay.win_scale=0", "run.hours=-1", "run.max_games=-1",
])
def test_validate_rejects_settings_that_would_fail_later(item):
    with pytest.raises(ConfigError):
        load_config(None, [item])


def test_the_training_defaults_are_the_ones_that_do_not_memorise():
    """Changed on 2026-11-13 after the first real runs (docs/PLAN.md): each position is trained on
    about once, targets lean on the next position's estimate, and the guard is on."""
    train = Config().train
    assert (train.replay_ratio, train.td_lambda, train.learning_rate, train.batch_size) == (1.0, 0.7, 3e-4, 1024)
    assert train.guard is True and train.guard_patience == 2
    assert load_config(REPO_ROOT / "configs" / "train.toml") == Config(), "configs/train.toml lists the defaults"
