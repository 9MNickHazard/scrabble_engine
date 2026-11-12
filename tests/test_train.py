"""`train.Trainer`: a tiny end-to-end run on CPU, checkpoints, resume, refusal, and STOP.

The trainer resolves lexicons through configs/lexicons.toml, whose word list is not in the repo,
so `engine.lexicon_entry` is pointed at the test word lists for this module. Batches are padded to
32 rather than 512 positions so the CPU runs stay quick (padding is tested in test_scorer.py).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scrabble_ai import _core
from scrabble_ai import engine as engine_module
from scrabble_ai import train as train_module
from scrabble_ai.config import Config, apply_overrides, load_config
from scrabble_ai.lexicon import LexiconError
from scrabble_ai.selfplay import Scorer
from scrabble_ai.train import Trainer, load_checkpoint_config, load_network, resolve_resume

from conftest import cuda_only


@pytest.fixture(scope="module")
def env(lexicon_dir, leaves, tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    fitted = root / "fitted.leaves"
    leaves.save(fitted)
    other = root / "other.leaves"
    _core.LeaveModel.zero().save(other)

    def lexicon_entry(name: str = "", config_path=None):
        name = name or "TEST"
        path = lexicon_dir / f"{name}.txt"
        if not path.exists():
            raise LexiconError(f"unknown lexicon {name!r}")
        return name, path

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "lexicon_entry", lexicon_entry)
        patch.setattr(Scorer, "PAD", 32)
        yield SimpleNamespace(root=root, fitted=fitted, other=other)


def settings(env, name: str, **changes) -> list[str]:
    values = {
        "run.name": name, "run.out_dir": str(env.root), "run.lexicon": "TEST", "run.leaves": str(env.fitted),
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
    values.update({key.replace("__", "."): value for key, value in changes.items()})
    return [f"{key}={value}" for key, value in values.items()]


def metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def fresh(env):
    config = load_config(None, settings(env, "fresh"))
    trainer = Trainer(config, quiet=True)
    trainer.run()
    run_dir = env.root / "fresh"
    latest = resolve_resume("latest", run_dir)
    saved = env.root / "fresh-checkpoint.pt"
    shutil.copy(latest, saved)
    return SimpleNamespace(config=config, trainer=trainer, run_dir=run_dir, latest=latest, saved=saved)


def test_a_fresh_run_writes_its_outputs(fresh):
    trainer, run_dir, config = fresh.trainer, fresh.run_dir, fresh.config
    assert trainer.games >= 24 and trainer.step > 0 and trainer.phase == "selfplay"
    assert trainer.best_step == trainer.step, "the closing evaluation promotes the first network"

    # config.toml holds the full configuration and reads back to the same thing.
    assert load_config(run_dir / "config.toml") == config

    checkpoints = run_dir / "checkpoints"
    assert (checkpoints / "latest.txt").read_text(encoding="utf-8").strip() == fresh.latest.name
    assert fresh.latest.name == f"step_{trainer.step:09d}.pt"
    assert (checkpoints / "best.pt").exists()
    assert len(list(checkpoints.glob("step_*.pt"))) <= 2
    assert not (run_dir / "STOP").exists()
    assert (run_dir / "tb").is_dir()

    records = metrics(run_dir)
    kinds = {r["kind"] for r in records}
    assert {"warmstart", "train", "selfplay", "eval", "checkpoint"} <= kinds
    warm = next(r for r in records if r["kind"] == "warmstart")
    assert warm["games_played"] >= 16 and warm["positions_added"] > 0
    assert records[-1]["kind"] == "checkpoint" and records[-1]["file"] == fresh.latest.name
    evals = [r for r in records if r["kind"] == "eval"]
    assert {r["opponent"] for r in evals} == {"greedy-leave", "greedy"} and all(r["pairs"] == 2 for r in evals)

    state = torch.load(fresh.latest, map_location="cpu", weights_only=False)
    assert (state["step"], state["games"], state["positions"]) == (trainer.step, trainer.games, trainer.positions)
    assert state["replay"] == trainer.replay.state()
    assert state["phase"] == "selfplay"
    assert state["lexicon_id"] == trainer.loaded.lexicon_id and state["lexicon_id"].startswith("TEST-")
    assert state["leaves_hash"] == engine_module.leaves_hash(trainer.leaves)
    assert load_checkpoint_config(fresh.latest) == config

    # The saved replay matches the buffer.
    saved = np.load(run_dir / "replay" / "value_target.npy")
    np.testing.assert_array_equal(saved, trainer.replay.arrays["value_target"][: len(trainer.replay)])
    assert len(trainer.replay) == state["replay"]["size"] > 0
    assert trainer.replay.total_added == trainer.positions + warm["positions_added"]

    # best.pt loads on its own.
    net, info = load_network(checkpoints / "best.pt", torch.device("cpu"))
    assert info["step"] == trainer.best_step
    for (name, a), (_, b) in zip(net.state_dict().items(), trainer.best_net.state_dict().items()):
        assert torch.equal(a, b), name


def test_resume_restores_the_state_and_continues(fresh, env):
    before = fresh.trainer
    path = resolve_resume("latest", fresh.run_dir)
    config = load_checkpoint_config(path)
    apply_overrides(config, ["run.max_games=40"])
    config.validate()
    trainer = Trainer(config, resume=path, quiet=True)

    assert (trainer.step, trainer.games, trainer.positions) == (before.step, before.games, before.positions)
    assert trainer.phase == "selfplay" and trainer.best_step == before.best_step
    assert trainer.samples_owed == before.samples_owed and trainer.elapsed == before.elapsed
    assert len(trainer.replay) == len(before.replay) and trainer.replay.state() == before.replay.state()
    for name in trainer.replay.arrays:
        np.testing.assert_array_equal(trainer.replay.arrays[name][: len(before.replay)],
                                      before.replay.arrays[name][: len(before.replay)])
    for model_name in ("net", "actor_net", "best_net"):
        mine, theirs = getattr(trainer, model_name).state_dict(), getattr(before, model_name).state_dict()
        assert all(torch.equal(mine[k], theirs[k]) for k in mine), model_name
    assert trainer.rng.bit_generator.state == before.rng.bit_generator.state
    assert torch.equal(trainer.generator.get_state(), before.generator.get_state())
    assert trainer.actor.next_seed == before.actor.next_seed + trainer.actor.slots
    assert int(trainer.actor.seeds[0]) == before.actor.next_seed, "fresh games continue the seed sequence"
    assert trainer.optimizer.state_dict()["state"].keys() == before.optimizer.state_dict()["state"].keys()

    trainer.run()
    assert trainer.games >= 40 and trainer.step > before.step
    latest = resolve_resume("latest", fresh.run_dir)
    assert latest.name == f"step_{trainer.step:09d}.pt"
    records = metrics(fresh.run_dir)
    assert any(r["kind"] == "gate" for r in records), "the resumed run gated against the saved best network"
    assert len(list((fresh.run_dir / "checkpoints").glob("step_*.pt"))) <= 2


def test_a_checkpoint_from_another_lexicon_is_refused(fresh, env):
    config = load_config(None, settings(env, "other-lexicon", run__lexicon="OTHER"))
    with pytest.raises(RuntimeError, match="lexicon TEST-.*OTHER-"):
        Trainer(config, resume=fresh.saved, quiet=True)


@pytest.mark.parametrize("which", ["other", "zero"])
def test_a_checkpoint_with_other_leave_values_is_refused(fresh, env, which):
    choice = str(env.other) if which == "other" else "zero"
    config = load_config(None, settings(env, f"other-leaves-{which}", run__leaves=choice))
    with pytest.raises(RuntimeError, match="different leave values"):
        Trainer(config, resume=fresh.saved, quiet=True)


def test_resolve_resume(env, tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to resume"):
        resolve_resume("latest", tmp_path / "missing-run")
    assert resolve_resume("some/file.pt", tmp_path) == Path("some/file.pt")


def test_a_stop_file_stops_the_run(env):
    config = load_config(None, settings(env, "stopper", warmstart__games=0, run__max_games=100_000))
    trainer = Trainer(config, quiet=True)
    assert trainer.phase == "selfplay"
    stop = env.root / "stopper" / "STOP"
    calls = 0
    original = trainer.actor.step

    def step(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 70:
            stop.write_text("", encoding="utf-8")
        return original(*args, **kwargs)

    trainer.actor.step = step
    trainer.run()
    assert calls == 70, "the loop checks for the STOP file before every step"
    assert trainer.games < 100_000
    assert not stop.exists(), "the STOP file is removed once the run has stopped"
    assert resolve_resume("latest", env.root / "stopper").exists()


def test_a_run_stopped_during_the_warm_start_still_owes_its_warm_start(env, monkeypatch):
    config = load_config(None, settings(env, "warm-stop", warmstart__games=400, warmstart__steps=50))
    trainer = Trainer(config, quiet=True)
    stop = env.root / "warm-stop" / "STOP"
    calls = 0
    original = train_module.Actor.step

    def step(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            stop.write_text("", encoding="utf-8")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(train_module.Actor, "step", step)
    trainer.run()
    state = torch.load(resolve_resume("latest", env.root / "warm-stop"), map_location="cpu", weights_only=False)
    assert state["step"] == 0, "no pretraining step was taken"
    assert state["phase"] == "warmstart"


@cuda_only
def test_a_tiny_run_on_the_gpu(env):
    config = load_config(None, settings(env, "gpu", run__device="cuda", warmstart__games=0, run__max_games=16,
                                        run__vram_fraction=0.9))
    trainer = Trainer(config, quiet=True)
    trainer.run()
    assert trainer.games >= 16 and trainer.step > 0
    net, state = load_network(resolve_resume("latest", env.root / "gpu"), torch.device("cpu"))
    assert state["config"]["run"]["device"] == "cuda"
    assert all(torch.isfinite(p).all() for p in net.parameters())


@cuda_only
def test_a_gpu_checkpoint_resumes_on_the_cpu(env):
    config = load_config(None, settings(env, "gpu-to-cpu", run__device="cuda", warmstart__games=0, run__max_games=8))
    Trainer(config, quiet=True).run()
    path = resolve_resume("latest", env.root / "gpu-to-cpu")
    resumed_config = load_checkpoint_config(path)
    apply_overrides(resumed_config, ["run.device=cpu", "run.max_games=16"])
    resumed_config.validate()
    trainer = Trainer(resumed_config, resume=path, quiet=True)
    assert trainer.games >= 8


def test_resuming_an_interrupted_warm_start_finishes_it_without_replaying_seeds(env, monkeypatch):
    """Stop a run part-way through the warm start, resume it, and check nothing was skipped."""
    overrides = settings(env, "warm-resume", warmstart__games=600, warmstart__steps=20, run__max_games=8)
    trainer = Trainer(load_config(None, overrides), quiet=True)
    stop = env.root / "warm-resume" / "STOP"
    calls = 0
    original = train_module.Actor.step

    def step(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            stop.write_text("", encoding="utf-8")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(train_module.Actor, "step", step)
        trainer.run()
    assert trainer.phase == "warmstart" and trainer.step == 0
    assert 0 <= trainer.warm_games < 600
    first_seed = trainer.config.run.seed * 1_000_000_000
    # Every slot of the warm-start actor was given a seed, finished or not.
    assert trainer.warm_next_seed >= first_seed + 512
    positions_before, seed_before = len(trainer.replay), trainer.warm_next_seed

    path = resolve_resume("latest", env.root / "warm-resume")
    resumed = Trainer(load_checkpoint_config(path), resume=path, quiet=True)
    assert (resumed.phase, resumed.warm_games, resumed.warm_next_seed) == ("warmstart", trainer.warm_games, seed_before)
    assert len(resumed.replay) == positions_before
    resumed.run()
    assert resumed.phase == "selfplay"
    assert resumed.warm_games >= 600, "the rest of the warm start was played"
    assert resumed.step >= 20, "and the pretraining steps were taken"
    assert resumed.warm_next_seed > seed_before, "with fresh seeds, not the interrupted ones again"
    assert resumed.games >= 8, "then self-play ran"
