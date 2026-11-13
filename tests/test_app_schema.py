"""`scrabble_ai.app.schema`: what the "new run" form is told about every training setting."""

from __future__ import annotations

import dataclasses
import re

import pytest

from scrabble_ai.app import schema as schema_module
from scrabble_ai.config import Config, load_config

TYPES = {"int": int, "float": float, "bool": bool, "str": str}
UNITS = {"hours", "games", "points", "steps", "positions", "minutes", "pairs"}


def config_fields() -> dict[str, tuple[type, object]]:
    """key -> (type, default) for every field of every config section."""
    defaults = Config()
    out = {}
    for section_field in dataclasses.fields(defaults):
        section = getattr(defaults, section_field.name)
        for field in dataclasses.fields(section):
            field_type = TYPES[field.type] if isinstance(field.type, str) else field.type
            out[f"{section_field.name}.{field.name}"] = (field_type, getattr(section, field.name))
    return out


@pytest.fixture(scope="module")
def schema():
    return schema_module.schema()


@pytest.fixture(scope="module")
def fields(schema):
    return [field for section in schema["sections"] for field in section["fields"]]


def test_sections_in_order(schema):
    assert [s["name"] for s in schema["sections"]] == ["run", "model", "selfplay", "warmstart", "train"]
    assert all(s["title"] for s in schema["sections"])
    for section in schema["sections"]:
        assert all(field["key"].startswith(section["name"] + ".") for field in section["fields"])


def test_every_setting_appears_exactly_once_except_name_and_out_dir(fields):
    keys = [field["key"] for field in fields]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(config_fields()) - {"run.name", "run.out_dir"}


def test_each_field_describes_its_setting(fields):
    known = config_fields()
    for field in fields:
        key = field["key"]
        field_type, default = known[key]
        assert set(field) == {"key", "label", "type", "default", "min", "max", "step", "multiple_of", "unit", "options",
                              "advanced", "locked_on_resume", "help"}, key
        assert field["unit"] is None or (field["unit"] in UNITS and field_type in (int, float)), key
        assert not re.search(r"\((hours|games|points|steps|positions|minutes|pairs)\)", field["label"]), \
            f"{key}: the unit is its own field now, not part of the label"
        assert field["default"] == default and type(field["default"]) is field_type, key
        if field["options"] is not None:
            assert field["type"] == "choice" and field_type is str, key
            assert field["options"] and all(isinstance(o, str) for o in field["options"]), key
            assert default in field["options"], f"{key}: the default is one of the choices"
        else:
            assert TYPES[field["type"]] is field_type, key
        assert field["label"].strip() and field["help"].strip(), key
        assert isinstance(field["advanced"], bool) and isinstance(field["locked_on_resume"], bool), key
        lo, hi, step = field["min"], field["max"], field["step"]
        if field_type in (int, float):
            if lo is not None:
                assert lo <= default, key
            if hi is not None:
                assert default <= hi, key
            if lo is not None and hi is not None:
                assert lo < hi, key
            if step is not None:
                assert step > 0, key
            if field_type is int:
                assert all(v is None or isinstance(v, int) for v in (lo, hi, step)), f"{key}: whole-number limits"
        else:
            assert lo is None and hi is None and step is None, key


def test_multiple_of(fields):
    """`multiple_of` is a rule the form enforces (training.js refuses an int that breaks it); `step`
    is only how far the arrow keys move."""
    rules = {f["key"]: f["multiple_of"] for f in fields if f["multiple_of"] is not None}
    assert rules == {"model.channels": 8}
    for field in fields:
        if field["multiple_of"] is not None:
            assert field["type"] == "int" and field["default"] % field["multiple_of"] == 0, field["key"]
            assert field["min"] % field["multiple_of"] == 0 and field["max"] % field["multiple_of"] == 0, field["key"]
    # The rule is the one Config.validate applies.
    load_config(None, ["model.channels=24"])
    with pytest.raises(Exception, match="model.channels must be a multiple of 8"):
        load_config(None, ["model.channels=20"])


def test_units(fields):
    units = {f["key"]: f["unit"] for f in fields}
    assert (units["run.hours"], units["run.max_games"], units["train.batch_size"], units["train.eval_pairs"]) == (
        "hours", "games", "positions", "pairs")
    assert units["train.checkpoint_minutes"] == "minutes" and units["selfplay.temperature_start"] == "points"
    assert units["model.blocks"] is None and units["train.learning_rate"] is None and units["run.device"] is None


def test_the_everyday_fields(fields):
    everyday = {f["key"] for f in fields if not f["advanced"]}
    assert {"run.hours", "run.max_games", "model.blocks", "model.channels"} <= everyday
    locked = {f["key"] for f in fields if f["locked_on_resume"]}
    assert {"model.blocks", "model.channels", "run.lexicon", "run.leaves", "train.value_scale"} <= locked


def test_described_keys_cover_only_real_settings():
    assert schema_module.described_keys() <= set(config_fields())
    assert schema_module.described_keys() == set(config_fields()) - schema_module.HIDDEN


def test_presets_are_valid_settings(schema, fields):
    by_key = {f["key"]: f for f in fields}
    names = [p["name"] for p in schema["presets"]]
    assert len(names) == len(set(names)) and schema["presets"][0]["overrides"] == {}
    for preset in schema["presets"]:
        assert preset["name"] and preset["description"]
        overrides = preset["overrides"]
        assert set(overrides) <= set(by_key), preset["name"]
        load_config(None, [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in overrides.items()])
        for key, value in overrides.items():
            field = by_key[key]
            assert type(value) is TYPES.get(field["type"], str) or (field["type"] == "float" and isinstance(value, int)), key
            if field["min"] is not None:
                assert value >= field["min"], (preset["name"], key)
            if field["max"] is not None:
                assert value <= field["max"], (preset["name"], key)


def test_presets_pass_the_forms_own_rules(schema, fields):
    by_key = {f["key"]: f for f in fields}
    for preset in schema["presets"]:
        for key, value in preset["overrides"].items():
            field = by_key[key]
            if field["multiple_of"]:
                assert value % field["multiple_of"] == 0, (preset["name"], key, value)
            if field["options"]:
                assert value in field["options"], (preset["name"], key, value)


def test_speed_rows(schema):
    rows = schema["speed"]
    assert [r["network"] for r in rows][0] == "6x64" and rows[0]["measured"] is True
    for row in rows:
        assert set(row) == {"network", "parameters", "games_per_second", "measured", "vram_gb"}
        blocks, channels = map(int, row["network"].split("x"))
        assert blocks >= 1 and channels % 8 == 0 and row["games_per_second"] > 0 and row["vram_gb"] > 0


def test_speed_parameter_counts_match_the_network(schema):
    """The parameter counts in the benchmark table are those of today's network at default settings."""
    from scrabble_ai.config import ModelConfig
    from scrabble_ai.model import ValueNet, parameter_count

    for row in schema["speed"]:
        blocks, channels = map(int, row["network"].split("x"))
        assert parameter_count(ValueNet(ModelConfig(blocks=blocks, channels=channels))) == row["parameters"], row["network"]
