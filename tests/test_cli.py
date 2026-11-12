"""`cli`: argument handling around `train --resume` (the Trainer itself is replaced by a stub)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scrabble_ai import _core, cli
from scrabble_ai import train as train_module
from scrabble_ai.config import Config


@pytest.fixture
def stub_trainer(monkeypatch):
    captured: dict = {}

    class StubTrainer:
        def __init__(self, config, resume=None, quiet=False):
            captured["config"], captured["resume"] = config, resume

        def run(self):
            captured["ran"] = True

    def checkpoint_config(path: Path) -> Config:
        captured["checkpoint"] = path
        config = Config()
        config.run.name = "from-checkpoint"
        return config

    monkeypatch.setattr(train_module, "Trainer", StubTrainer)
    monkeypatch.setattr(train_module, "load_checkpoint_config", checkpoint_config)
    return captured


def make_run(root: Path, name: str) -> Path:
    checkpoints = root / name / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_000000007.pt").write_bytes(b"")
    (checkpoints / "latest.txt").write_text("step_000000007.pt", encoding="utf-8")
    return checkpoints / "step_000000007.pt"


def run_cli(argv: list[str]) -> None:
    try:
        cli.main(argv)
    except SystemExit as exit_:
        pytest.fail(f"scrabble-ai {' '.join(argv)} exited: {exit_}")


def test_resume_latest_with_name(stub_trainer, tmp_path):
    checkpoint = make_run(tmp_path, "foo")
    run_cli(["train", "--name", "foo", "--set", f"run.out_dir={tmp_path}", "--resume", "latest",
             "--set", "train.learning_rate=5e-4"])
    assert stub_trainer["resume"] == checkpoint
    config = stub_trainer["config"]
    assert config.run.name == "foo" and config.train.learning_rate == 5e-4 and stub_trainer["ran"]


def test_resume_latest_with_set_run_name(stub_trainer, tmp_path):
    checkpoint = make_run(tmp_path, "foo")
    run_cli(["train", "--set", "run.name=foo", "--set", f"run.out_dir={tmp_path}", "--resume", "latest"])
    assert stub_trainer["resume"] == checkpoint


def test_fresh_train_applies_config_and_overrides(stub_trainer, tmp_path):
    config_file = tmp_path / "c.toml"
    config_file.write_text("[model]\nblocks = 3\n", encoding="utf-8")
    run_cli(["train", "--config", str(config_file), "--name", "x", "--hours", "1.5", "--set", "model.channels=32"])
    config = stub_trainer["config"]
    assert (config.model.blocks, config.model.channels, config.run.name, config.run.hours) == (3, 32, "x", 1.5)
    assert stub_trainer["resume"] is None


def test_bad_settings_exit_with_a_message(stub_trainer):
    with pytest.raises(SystemExit, match="unknown setting model.blokcs"):
        cli.main(["train", "--set", "model.blokcs=3"])


def test_bench_module_imports():
    import scrabble_ai.bench  # noqa: F401

    assert hasattr(_core.Engine, "bench_movegen")
