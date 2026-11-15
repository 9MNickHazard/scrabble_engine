"""`offline.GameTrainer`: training from a saved game set, end to end on the CPU with a 1x16 network
and a game set of a few hundred games of the test word list.

What it should do is in the docstring of `scrabble_ai/offline.py` and in docs/APP_API.md
("Training from saved games"): one position per game per pass, batches a pure function of the
step and the seed, win/draw/loss as the target with helper outputs balanced against it, an
averaged network that is what gets evaluated and saved, best.pt by the highest win rate, and
checkpoints that resume and refuse a game set that has changed.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scrabble_ai import _core, cli
from scrabble_ai import engine as engine_module
from scrabble_ai import games as games_module
from scrabble_ai import offline as offline_module
from scrabble_ai import train as train_module
from scrabble_ai.config import OFFLINE_DEFAULTS, ConfigError, load_config
from scrabble_ai.evaluate import MatchResult, NetPlayer, play_match
from scrabble_ai.games import GamesError
from scrabble_ai.lexicon import LexiconError
from scrabble_ai.model import parameter_count
from scrabble_ai.offline import GameTrainer, Prefetcher
from scrabble_ai.selfplay import Scorer
from scrabble_ai.train import load_network, resolve_resume

SET_GAMES = 300  # 30 held out, 270 to train from: 8 steps of 32 per pass
COMMON = {"kind", "time", "step", "games", "positions", "hours"}
HELPER_LOSSES = {"next_score_loss", "next_bingo_loss", "next_squares_loss"}
HELPER_WEIGHTS = {"next_score_weight", "next_bingo_weight", "next_squares_weight"}


@pytest.fixture(scope="module")
def env(lexicon_dir, leaves, tmp_path_factory):
    root = tmp_path_factory.mktemp("offline")
    fitted = root / "fitted.leaves"
    leaves.save(fitted)

    def lexicon_entry(name: str = "", config_path=None):
        name = name or "TEST"
        path = lexicon_dir / f"{name}.txt"
        if not path.exists():
            raise LexiconError(f"unknown lexicon {name!r}")
        return name, path

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "lexicon_entry", lexicon_entry)
        patch.setattr(Scorer, "PAD", 32)
        patch.setattr(offline_module, "git_commit", lambda: "test")  # no `git rev-parse` from the tests
        loaded = engine_module.load("TEST", str(fitted))
        games_dir = root / "games" / "small-set"
        games_module.generate(loaded, games_dir, SET_GAMES, block=100, say=lambda _: None)
        yield SimpleNamespace(root=root, fitted=fitted, loaded=loaded, games_dir=games_dir)


def settings(env, name: str, **changes) -> list[str]:
    values = {
        "run.name": name, "run.out_dir": str(env.root / "runs"), "run.lexicon": "TEST", "run.leaves": str(env.fitted),
        "run.device": "cpu", "run.seed": 3,
        "model.blocks": 1, "model.channels": 16, "model.vector_hidden": 16, "model.head_hidden": 16,
        "model.value_channels": 2, "model.belief_samples": 0,
        **{key: str(value).lower() if isinstance(value, bool) else value for key, value in OFFLINE_DEFAULTS.items()},
        "selfplay.max_candidates": 8, "selfplay.max_exchanges": 2,
        "offline.games": str(env.games_dir), "offline.passes": 2.0, "offline.batch_size": 32,
        "offline.validation_games": 30, "offline.validate_steps": 8, "offline.warmup_steps": 4,
        "offline.eval_minutes": 60.0, "offline.eval_pairs": 2, "offline.final_pairs": 3,
        "offline.checkpoint_minutes": 60.0, "offline.keep_checkpoints": 2,
    }
    values.update({key.replace("__", "."): value for key, value in changes.items()})
    return [f"{key}={value}" for key, value in values.items()]


def make(env, name: str, resume: Path | None = None, **changes) -> GameTrainer:
    return GameTrainer(load_config(None, settings(env, name, **changes)), resume, quiet=True)


def metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def stop_after(trainer: GameTrainer, steps: int) -> None:
    """Write the STOP file once `steps` training steps have been taken."""
    real = trainer.train_step

    def step(rows):
        real(rows)
        if trainer.step + 1 >= steps:
            (trainer.run_dir / "STOP").write_text("", encoding="utf-8")

    trainer.train_step = step


def same_state(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def wake(net) -> None:
    """Give the zero-initialised output layers weights, as training would."""
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in (net.value, net.win):
            layer.weight.copy_(0.5 * torch.randn(layer.weight.shape, generator=generator))
            layer.bias.copy_(0.1 * torch.randn(layer.bias.shape, generator=generator))


@pytest.fixture(scope="module")
def fresh(env):
    trainer = make(env, "fresh")
    trainer.run()
    run_dir = env.root / "runs" / "fresh"
    return SimpleNamespace(trainer=trainer, run_dir=run_dir, records=metrics(run_dir),
                           latest=resolve_resume("latest", run_dir))


# ---- a whole run -------------------------------------------------------------------------------


def test_the_shape_of_a_run(fresh):
    t = fresh.trainer
    assert (t.validation_games, t.train_games, t.batch, t.steps_per_pass, t.total_steps) == (30, 270, 32, 8, 16)
    assert t.step == 16 and t.best_step == 16 and t.best_win_rate is not None
    assert t.amp is False, "no bfloat16 on the CPU"


def test_a_run_writes_its_outputs(fresh, env):
    run_dir, t = fresh.run_dir, fresh.trainer
    assert load_config(run_dir / "config.toml") == t.config
    checkpoints = run_dir / "checkpoints"
    assert (checkpoints / "latest.txt").read_text(encoding="utf-8").strip() == "step_000000016.pt"
    assert fresh.latest == checkpoints / "step_000000016.pt" and fresh.latest.exists()
    assert (checkpoints / "best.pt").exists()
    assert not (run_dir / "STOP").exists() and not (run_dir / "running.json").exists()
    index = json.loads((checkpoints / "index.json").read_text(encoding="utf-8"))
    assert set(index) == {"network", "parameters", "lexicon_id", "leaves_hash", "value_scale", "checkpoints", "best"}
    assert index["network"] == "1x16" and index["parameters"] == parameter_count(t.net)
    assert (index["lexicon_id"], index["leaves_hash"]) == (env.loaded.lexicon_id, env.loaded.leaves_hash)
    assert [c["file"] for c in index["checkpoints"]] == ["step_000000016.pt"]
    assert index["best"]["file"] == "best.pt" and index["best"]["step"] == 16


def test_the_metrics_records(fresh):
    records, t = fresh.records, fresh.trainer
    for record in records:
        assert COMMON <= set(record), record["kind"]
        assert record["games"] == record["positions"] == record["step"] * 32, "one position from each game"
    start, stop = records[0], records[-1]
    assert set(start) == COMMON | {"event", "mode", "parameters", "limit_hours", "game_set", "train_games",
                                   "validation_games", "total_steps", "steps_per_pass", "batch"}
    assert (start["kind"], start["event"], start["mode"], start["game_set"]) == ("session", "start", "games", "small-set")
    assert (start["train_games"], start["validation_games"], start["total_steps"], start["steps_per_pass"], start["batch"]) == (
        270, 30, 16, 8, 32)
    assert start["parameters"] == parameter_count(t.net) and start["limit_hours"] == 0.0
    assert set(stop) == COMMON | {"event", "reason", "finished"}
    assert (stop["kind"], stop["event"], stop["finished"]) == ("session", "stop", True)
    assert stop["reason"] == "all 16 steps done"
    kinds = [r["kind"] for r in records]
    assert {"train", "valid", "eval", "best", "checkpoint"} <= set(kinds)
    assert set(kinds) <= {"session", "train", "valid", "eval", "best", "checkpoint"}

    train = [r for r in records if r["kind"] == "train"]
    assert train
    for record in train:
        assert set(record) == COMMON | {"loss", "win_loss", "spread_loss", *HELPER_LOSSES, "spread_weight", *HELPER_WEIGHTS,
                                        "learning_rate", "positions_per_second", "pass"}
        assert all(math.isfinite(record[k]) for k in record if k.endswith("_loss") or k == "loss")
        assert record["pass"] == round(record["step"] / 8, 3) and record["positions_per_second"] > 0
    valid = [r for r in records if r["kind"] == "valid"]
    assert [r["step"] for r in valid] == [8, 16], "every validate_steps steps; the end adds none when one was just made"
    for record in valid:
        assert set(record) == COMMON | {"win_loss", "spread_loss", *HELPER_LOSSES, "win_accuracy", "win_brier",
                                        "value_rmse_points", "leave_only_rmse_points", "positions_checked"}
        assert record["positions_checked"] == 30
        assert 0 <= record["win_accuracy"] <= 1 and 0 <= record["win_brier"] <= 1
        assert all(math.isfinite(record[k]) for k in record if k != "kind")
    evals = [r for r in records if r["kind"] == "eval"]
    assert len(evals) == 1
    assert set(evals[0]) == COMMON | {"opponent", "spread", "spread_ci", "win_rate", "win_ci", "score_for",
                                      "score_against", "pairs"}
    assert evals[0]["opponent"] == "greedy-leave" and evals[0]["pairs"] == 3, "the final match has final_pairs"
    best = [r for r in records if r["kind"] == "best"]
    assert len(best) == 1 and set(best[0]) == COMMON | {"win_rate", "win_ci", "pairs"}
    assert best[0]["win_rate"] == evals[0]["win_rate"]
    checkpoint = [r for r in records if r["kind"] == "checkpoint"]
    assert [set(r) for r in checkpoint] == [COMMON | {"file"}] and checkpoint[0]["file"] == "step_000000016.pt"
    assert kinds[-2] == "checkpoint"


def test_a_step_file_holds_the_averaged_network_and_the_rest(fresh, env):
    t = fresh.trainer
    state = torch.load(fresh.latest, map_location="cpu", weights_only=False)
    assert state["format"] == train_module.CHECKPOINT_FORMAT
    assert same_state(state["model"], t.average.state_dict()), "`model` is the averaged network"
    assert same_state(state["trained_model"], t.net.state_dict())
    assert not same_state(state["model"], state["trained_model"])
    assert (state["step"], state["games"]) == (16, 16 * 32)
    assert (state["lexicon_id"], state["leaves_hash"]) == (env.loaded.lexicon_id, env.loaded.leaves_hash)
    assert state["games_set"] == {"name": "small-set", "games": SET_GAMES, "first_seed": t.games_meta["first_seed"]}
    assert state["weights"] == t.weights and (state["best_step"], state["best_win_rate"]) == (16, t.best_win_rate)
    assert state["git_commit"] == "test" and state["config"] == t.config.to_dict()
    assert {"optimizer", "elapsed", "torch_generator", "torch_version"} <= set(state)
    best = torch.load(fresh.run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert "trained_model" not in best and "optimizer" not in best
    assert best["step"] == 16 and same_state(best["model"], t.average.state_dict())


def test_saved_networks_load_and_play(fresh, env):
    for path in (fresh.latest, fresh.run_dir / "checkpoints" / "best.pt"):
        net, state = load_network(path, torch.device("cpu"))
        assert same_state(net.state_dict(), fresh.trainer.average.state_dict()), path.name
        assert (net.config.head, net.config.win_head, net.config.aux_heads) == ("pooled", "wdl", True)
        scorer = Scorer(net, env.loaded.engine, state["config"]["train"]["value_scale"], torch.device("cpu"),
                        win_weight=state["config"]["selfplay"]["win_weight"])
        game = _core.Game(env.loaded.engine, env.loaded.leaves, 31)
        _, board, leave, unseen, meta, leave_value = game.candidates(8, 2)
        values, wins = scorer(board, leave, unseen, meta, leave_value)
        assert values.shape == wins.shape == (len(board),)
        assert torch.isfinite(values).all() and ((wins > 0) & (wins < 1)).all()
    player = NetPlayer("best", scorer, 0.9, 100.0)
    result = play_match(env.loaded.engine, env.loaded.leaves, player, "greedy-leave", np.arange(1, dtype=np.uint64) + 5,
                        torch.device("cpu"), 8, 2)
    assert result.pairs == 1 and 0.0 <= result.win_rate <= 1.0


# ---- batches -----------------------------------------------------------------------------------


def test_a_batch_is_a_pure_function_of_the_step(env):
    a, b = make(env, "pure-a"), make(env, "pure-b")
    first = a._batch(5)
    a._batch(13)  # another pass in between
    for again in (a._batch(5), b._batch(5)):
        assert first.keys() == again.keys()
        for key in first:
            assert np.array_equal(first[key], again[key]), key
    assert not np.array_equal(first["board"], a._batch(6)["board"])
    other_seed = make(env, "pure-c", run__seed=4)
    assert not np.array_equal(first["board"], other_seed._batch(5)["board"])
    # What a batch holds.
    assert first["board"].shape == (32, 225) and first["targets"].shape == (32, _core.TARGETS)
    assert first["squares"].shape == (32, 450) and first["cross"] is not None and first["open"] is not None


def test_one_position_per_game_per_pass_and_never_a_held_out_game(env, monkeypatch):
    trainer = make(env, "passes", offline__passes=3.0)
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    monkeypatch.setattr(trainer, "_sample", lambda games, picks: seen.append((games.copy(), picks.copy())) or {})
    per_pass = []
    for number in range(3):
        seen.clear()
        for index in range(trainer.steps_per_pass):
            trainer._batch(number * trainer.steps_per_pass + index)
        games = np.concatenate([g for g, _ in seen])
        picks = np.concatenate([p for _, p in seen])
        assert len(games) == trainer.steps_per_pass * trainer.batch == 256
        assert len(np.unique(games)) == len(games), f"pass {number}: no game twice"
        assert games.max() < trainer.train_games == 270, "held-out games are never trained on"
        assert ((picks >= 0) & (picks < 1)).all()
        per_pass.append((games, picks))
    assert not np.array_equal(per_pass[0][0], per_pass[1][0]), "each pass in its own order"
    seen.clear()
    trainer._validation = None
    trainer._validation_batches()
    held_out = np.concatenate([g for g, _ in seen])
    assert sorted(held_out.tolist()) == list(range(270, 300))
    # Positions of a game differ from pass to pass.
    common = np.intersect1d(per_pass[0][0], per_pass[1][0])
    assert len(common) > 200
    picks0 = dict(zip(per_pass[0][0].tolist(), per_pass[0][1].tolist()))
    picks1 = dict(zip(per_pass[1][0].tolist(), per_pass[1][1].tolist()))
    assert sum(picks0[g] != picks1[g] for g in common.tolist()) > 0.9 * len(common)


def test_validation_positions_are_the_same_every_time(env):
    a, b = make(env, "valid-a"), make(env, "valid-b")
    first, second = a._validation_batches(), b._validation_batches()
    assert len(first) == len(second) == 1
    for key in ("board", "targets", "turns"):
        assert np.array_equal(first[0][key], second[0][key])
    none = make(env, "no-validation", offline__validation_games=0)
    assert none.validation_games == 0 and none.train_games == SET_GAMES
    assert none.validate() is None
    assert not (none.run_dir / "metrics.jsonl").exists() or not any(
        r["kind"] == "valid" for r in metrics(none.run_dir))


def test_the_transpose_augmentation_turns_the_square_targets_with_the_board(env):
    trainer = make(env, "transpose")
    rows = trainer._batch(0)
    captured = []

    class Capture:
        def outputs(self, planes, vector):
            captured.append(planes.detach().clone())
            return trainer.net.outputs(planes, vector)

    _, _, plain, _ = trainer._forward(Capture(), rows, augment=False)
    _, _, augmented, _ = trainer._forward(Capture(), rows, augment=True)
    before, after = captured[0][:, :33], captured[1][:, :33]  # the letter, blank, placed, empty and premium planes
    flipped = 0
    for i in range(len(before)):
        if torch.equal(before[i], before[i].transpose(-1, -2)):
            continue  # a board that reads the same both ways says nothing
        maps = plain[i].view(2, 15, 15)
        if torch.equal(after[i], before[i]):
            assert torch.equal(augmented[i], plain[i]), i
        else:
            assert torch.equal(after[i], before[i].transpose(-1, -2)), i
            assert torch.equal(augmented[i].view(2, 15, 15), maps.transpose(-1, -2)), i
            flipped += 1
    assert 4 <= flipped <= 28, "about half the rows are turned"


# ---- losses ------------------------------------------------------------------------------------


def test_the_losses_and_their_masks(env):
    trainer = make(env, "losses")
    wake(trainer.net)
    rows = trainer._batch(1)
    out, targets, squares, baseline = trainer._forward(trainer.net, rows, augment=False)
    losses = trainer._losses(out, targets, squares, baseline)
    assert set(losses) == {"win", "spread", "next_score", "next_bingo", "next_squares"}
    assert all(torch.isfinite(v) and v.ndim == 0 for v in losses.values())
    scale, delta = trainer.config.train.value_scale, trainer.config.train.huber_delta

    # Win, draw or loss: classes 0, 1, 2 for a result of 1, 0.5, 0.
    result = targets[:, _core.T_WIN]
    classes = torch.where(result == 1.0, 0, torch.where(result == 0.5, 1, 2))
    assert torch.allclose(losses["win"], F.cross_entropy(out["win"], classes))
    assert torch.allclose(losses["spread"], F.huber_loss(out["value"], (targets[:, _core.T_SPREAD] - baseline) / scale,
                                                         delta=delta))
    assert torch.equal(baseline, torch.from_numpy(rows["leave_value"])), "the leave baseline is on by default"

    # Rows with no next move say nothing about it, whatever their targets hold.
    targets = targets.clone()
    squares = squares.clone()
    targets[::2, _core.T_HAS_OPP] = 0.0
    targets[1::3, _core.T_HAS_OWN] = 0.0
    masked = trainer._losses(out, targets, squares, baseline)
    has = targets[:, _core.T_HAS_OPP] == 1
    turn = out["next_turn"]
    assert torch.allclose(masked["next_score"], F.huber_loss(turn[has, 0], targets[has, _core.T_OPP_SCORE] / scale))
    assert torch.allclose(masked["next_bingo"],
                          F.binary_cross_entropy_with_logits(turn[has, 1], targets[has, _core.T_OPP_BINGO]))
    junk_targets, junk_squares = targets.clone(), squares.clone()
    junk_targets[~has, _core.T_OPP_SCORE] = 10_000.0
    junk_targets[~has, _core.T_OPP_BINGO] = 1.0
    junk_squares[~has, 0] = 1.0
    junk_squares[targets[:, _core.T_HAS_OWN] == 0, 1] = 1.0
    junked = trainer._losses(out, junk_targets, junk_squares, baseline)
    for name in ("next_score", "next_bingo", "next_squares"):
        assert torch.equal(junked[name], masked[name]), name
    for name in ("win", "spread"):
        assert torch.equal(junked[name], losses[name]), name
    # With no next move anywhere the helper losses are zero, not NaN.
    none = targets.clone()
    none[:, _core.T_HAS_OPP] = 0.0
    none[:, _core.T_HAS_OWN] = 0.0
    empty = trainer._losses(out, none, squares, baseline)
    assert empty["next_score"] == 0 and empty["next_bingo"] == 0 and empty["next_squares"] == 0


def test_a_binary_network_without_helpers_also_trains(env):
    trainer = make(env, "binary", model__win_head="binary", model__aux_heads="false")
    assert trainer._helpers() == ("spread",) and trainer.weights == {"spread": 0.25}
    rows = trainer._batch(0)
    out, targets, squares, baseline = trainer._forward(trainer.net, rows, augment=False)
    losses = trainer._losses(out, targets, squares, baseline)
    assert set(losses) == {"win", "spread"}
    assert torch.allclose(losses["win"], F.binary_cross_entropy_with_logits(out["win"], targets[:, _core.T_WIN]))
    before = copy.deepcopy(trainer.net.state_dict())
    for step in range(3):
        trainer.train_step(trainer._batch(step))
        trainer.step += 1
    assert all(math.isfinite(v) for entry in trainer.losses for v in entry.values())
    assert not same_state(before, trainer.net.state_dict())
    trainer._report_training(1.0, 3)
    record = metrics(trainer.run_dir)[-1]
    assert set(record) == COMMON | {"loss", "win_loss", "spread_loss", "spread_weight", "learning_rate",
                                    "positions_per_second", "pass"}
    valid = trainer.validate()
    assert set(valid) == {"win_loss", "spread_loss", "win_accuracy", "win_brier", "value_rmse_points",
                          "leave_only_rmse_points", "positions_checked"}


def test_training_with_the_inference_about_the_opponents_rack(env):
    trainer = make(env, "belief", model__belief_samples=4)
    rows = trainer._batch(0)
    assert (rows["meta"][:, _core.M_BELIEF_INFERRED] == 1).any()
    plain = make(env, "no-belief")._batch(0)
    assert np.array_equal(rows["board"], plain["board"]) and np.array_equal(rows["targets"], plain["targets"])
    trainer.train_step(rows)
    assert all(math.isfinite(v) for v in trainer.losses[-1].values())


@pytest.mark.parametrize("kind", ["cnn-global", "transformer"])
def test_the_other_kinds_train(env, kind):
    trainer = make(env, f"kind-{kind}", model__kind=kind, model__blocks=2)
    before = copy.deepcopy(trainer.net.state_dict())
    for step in range(2):
        trainer.train_step(trainer._batch(step))
        trainer.step += 1
    assert all(math.isfinite(v) for entry in trainer.losses for v in entry.values())
    assert not same_state(before, trainer.net.state_dict())


def test_the_logged_loss_is_the_weighted_sum(env):
    trainer = make(env, "weighted")
    trainer.weights = {"spread": 0.5, "next_score": 0.2, "next_bingo": 0.3, "next_squares": 0.7}
    trainer.train_step(trainer._batch(0))
    entry = trainer.losses[-1]
    expected = entry["win"] + sum(trainer.weights[name] * entry[name] for name in trainer.weights)
    assert entry["loss"] == pytest.approx(expected, rel=1e-5)


# ---- balancing the helpers ---------------------------------------------------------------------


def pulls(trainer, out, losses) -> dict:
    def norm(loss):
        (gradient,) = torch.autograd.grad(loss, out["trunk"], retain_graph=True, allow_unused=True)
        return 0.0 if gradient is None else float(gradient.norm())

    return {name: norm(loss) for name, loss in losses.items()}


def test_rebalance_moves_the_helper_weights_toward_their_share(env):
    trainer = make(env, "balance")
    rows = trainer._batch(0)
    # A new network: the main output's gradient is zero, so there is nothing to measure against.
    out, targets, squares, baseline = trainer._forward(trainer.net, rows, augment=False)
    losses = trainer._losses(out, targets, squares, baseline)
    start = dict(trainer.weights)
    assert pulls(trainer, out, losses)["win"] == 0.0
    trainer._rebalance(out, losses)
    assert trainer.weights == start

    wake(trainer.net)
    out, targets, squares, baseline = trainer._forward(trainer.net, rows, augment=False)
    losses = trainer._losses(out, targets, squares, baseline)
    measured = pulls(trainer, out, losses)
    assert measured["win"] > 0 and all(measured[name] > 0 for name in start)
    trainer._rebalance(out, losses)
    for name, before in start.items():
        target = min(100.0, max(0.01, 0.15 * measured["win"] / measured[name]))
        assert trainer.weights[name] == pytest.approx(0.5 * before + 0.5 * target, rel=1e-4), name
    # Again and again: each weight converges on its share.
    for _ in range(30):
        trainer._rebalance(out, losses)
    for name in start:
        assert trainer.weights[name] * measured[name] == pytest.approx(0.15 * measured["win"], rel=1e-3), name


def test_with_no_share_the_fixed_weights_are_used(env):
    trainer = make(env, "fixed", offline__aux_share=0.0, offline__spread_weight=0.4, offline__aux_weight=0.05)
    assert trainer.weights == {"spread": 0.4, "next_score": 0.05, "next_bingo": 0.05, "next_squares": 0.05}
    wake(trainer.net)
    rows = trainer._batch(0)
    out, targets, squares, baseline = trainer._forward(trainer.net, rows, augment=False)
    trainer._rebalance(out, trainer._losses(out, targets, squares, baseline))
    trainer.step = offline_module.BALANCE_AFTER
    trainer.train_step(rows)
    assert trainer.weights == {"spread": 0.4, "next_score": 0.05, "next_bingo": 0.05, "next_squares": 0.05}


def test_train_step_rebalances_on_schedule(env):
    trainer = make(env, "schedule")
    wake(trainer.net)
    rows = trainer._batch(0)
    start = dict(trainer.weights)
    for step in (0, 100, offline_module.BALANCE_AFTER - 1, offline_module.BALANCE_AFTER + 1):
        trainer.step = step
        trainer.train_step(rows)
        assert trainer.weights == start, f"no re-measurement at step {step}"
    trainer.step = offline_module.BALANCE_AFTER
    trainer.train_step(rows)
    assert trainer.weights != start
    changed = dict(trainer.weights)
    trainer.step = offline_module.BALANCE_AFTER + offline_module.BALANCE_EVERY
    trainer.train_step(rows)
    assert trainer.weights != changed


def test_the_learning_rate_schedule(env):
    trainer = make(env, "schedule-lr", offline__learning_rate=1e-3, offline__warmup_steps=4)
    rates = []
    for step in range(trainer.total_steps + 1):
        trainer.step = step
        rates.append(trainer.learning_rate())
    assert rates[0] == pytest.approx(1e-3 * 0.25 * 1.0, rel=1e-6), "a quarter of the way up the warm-up, at the top of the cosine"
    assert rates[-1] == pytest.approx(1e-3 / 20), "a twentieth of the peak by the last step"
    peak = int(np.argmax(rates))
    assert peak == 3 and all(a >= b for a, b in zip(rates[peak:], rates[peak + 1:]))


# ---- evaluation and the best network -----------------------------------------------------------


def test_best_is_saved_only_when_the_win_rate_improves(env, monkeypatch):
    trainer = make(env, "best")
    calls = []
    results = iter([0.40, 0.35, 0.40, 0.55, 0.20])

    def fake_match(engine, leaves, a, b, seeds, device, max_candidates, max_exchanges):
        calls.append(SimpleNamespace(a=a, b=b, seeds=seeds, max_candidates=max_candidates, max_exchanges=max_exchanges))
        rate = next(results)
        return MatchResult(a.name, b, len(seeds), 3.0, 4.0, rate, 0.05, 420.0, 417.0, 24.0)

    monkeypatch.setattr(offline_module, "play_match", fake_match)
    best = trainer.checkpoint_dir / "best.pt"
    history = []
    for step, pairs in ((2, 6), (4, 6), (6, 6), (8, 9), (10, 6)):
        trainer.step = step
        result = trainer.evaluate(pairs)
        assert result.pairs == pairs
        history.append((trainer.best_step, trainer.best_win_rate, best.exists() and torch.load(
            best, map_location="cpu", weights_only=False)["step"]))
    assert history == [(2, 0.40, 2), (2, 0.40, 2), (2, 0.40, 2), (8, 0.55, 8), (8, 0.55, 8)], \
        "a lower or equal win rate keeps the best network"
    records = metrics(trainer.run_dir)
    assert [r["step"] for r in records if r["kind"] == "best"] == [2, 8]
    evals = [r for r in records if r["kind"] == "eval"]
    assert [r["win_rate"] for r in evals] == [0.40, 0.35, 0.40, 0.55, 0.20]
    assert all(r["win_ci"] == 0.05 and r["spread"] == 3.0 and r["spread_ci"] == 4.0 for r in evals)
    assert [r["pairs"] for r in evals] == [6, 6, 6, 9, 6]
    index = json.loads((trainer.checkpoint_dir / "index.json").read_text(encoding="utf-8"))
    assert index["best"]["step"] == 8
    # The match is the averaged network, choosing as it is meant to, against the static evaluator.
    call = calls[0]
    assert call.b == "greedy-leave" and len(call.seeds) == 6
    assert (call.max_candidates, call.max_exchanges) == (8, 2)
    assert call.a.win_weight == 0.9 and call.a.win_scale == trainer.config.selfplay.win_scale
    assert same_state(call.a.scorer.net.state_dict(), trainer.average.state_dict())
    assert call.a.scorer.net is not trainer.average, "a copy, so training on cannot change it mid-match"


# ---- stopping and resuming ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def stopped(env):
    """A run stopped with its STOP file after 5 of its 16 steps."""
    trainer = make(env, "resumable")
    stop_after(trainer, 5)
    trainer.run()
    run_dir = env.root / "runs" / "resumable"
    return SimpleNamespace(trainer=trainer, run_dir=run_dir, records=metrics(run_dir), latest=resolve_resume("latest", run_dir))


def test_a_stop_file_stops_the_run_cleanly(stopped):
    t, records = stopped.trainer, stopped.records
    assert t.step == 5
    stop = records[-1]
    assert (stop["event"], stop["reason"], stop["finished"]) == ("stop", "STOP file found", False)
    assert records[-2]["kind"] == "checkpoint" and records[-2]["file"] == "step_000000005.pt"
    assert [r["pairs"] for r in records if r["kind"] == "eval"] == [2], "eval_pairs, not final_pairs, when stopped early"
    assert [r["step"] for r in records if r["kind"] == "valid"] == [5]
    assert not (stopped.run_dir / "STOP").exists() and not (stopped.run_dir / "running.json").exists()


def test_resume_restores_everything_and_carries_on(stopped, fresh, env):
    t = stopped.trainer
    resumed = make(env, "resumable", resume=stopped.latest)
    assert resumed.step == 5 and resumed.elapsed == pytest.approx(t.elapsed)
    assert same_state(resumed.net.state_dict(), t.net.state_dict())
    assert same_state(resumed.average.state_dict(), t.average.state_dict())
    saved, restored = t.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert saved["param_groups"] == restored["param_groups"]
    for key, values in saved["state"].items():
        for name, value in values.items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(restored["state"][key][name])), (key, name)
    assert resumed.weights == t.weights
    assert (resumed.best_step, resumed.best_win_rate) == (t.best_step, t.best_win_rate) == (5, t.best_win_rate)
    assert torch.equal(resumed.generator.get_state(), t.generator.get_state())
    # The batches it goes on with are the ones it would have had.
    for step in (5, 11):
        mine, straight = resumed._batch(step), fresh.trainer._batch(step)
        assert all(np.array_equal(mine[k], straight[k]) for k in mine), step
    resumed.run()
    assert resumed.step == 16
    records = metrics(stopped.run_dir)
    starts = [r for r in records if r["kind"] == "session" and r["event"] == "start"]
    assert len(starts) == 2 and starts[1]["step"] == 5
    assert records[-1]["event"] == "stop" and records[-1]["finished"] is True
    # Stopped and carried on, the run trains exactly the network of a run that never stopped.
    final = torch.load(resolve_resume("latest", stopped.run_dir), map_location="cpu", weights_only=False)
    straight = torch.load(fresh.latest, map_location="cpu", weights_only=False)
    assert same_state(final["trained_model"], straight["trained_model"])
    assert same_state(final["model"], straight["model"])


def test_resume_is_refused_when_the_game_set_has_changed(env, tmp_path):
    directory = tmp_path / "growing"
    shutil.copytree(env.games_dir, directory)
    trainer = make(env, "grows", offline__games=str(directory))
    stop_after(trainer, 2)
    trainer.run()
    latest = resolve_resume("latest", trainer.run_dir)
    make(env, "grows", resume=latest, offline__games=str(directory))  # unchanged: fine
    games_module.generate(env.loaded, directory, SET_GAMES + 20, say=lambda _: None)
    with pytest.raises(RuntimeError, match="has changed"):
        make(env, "grows", resume=latest, offline__games=str(directory))


def test_resume_refuses_another_batch_size_or_held_out_set(env):
    """Both decide which games a step trains on, and which are never trained on."""
    trainer = make(env, "fixed-shape")
    stop_after(trainer, 2)
    trainer.run()
    latest = resolve_resume("latest", trainer.run_dir)
    for change in ({"offline__batch_size": 16}, {"offline__validation_games": 10}):
        with pytest.raises(RuntimeError, match="cannot change on resume"):
            make(env, "fixed-shape", resume=latest, **change)
    assert make(env, "fixed-shape", resume=latest, offline__passes=3.0).total_steps == 24, "more passes are allowed"


def test_a_stop_asked_for_before_the_first_step_counts_and_nothing_is_measured(env):
    trainer = make(env, "early-stop")
    (trainer.run_dir / "STOP").write_text("", encoding="utf-8")  # written since this process began: a request
    trainer.run()
    records = metrics(trainer.run_dir)
    assert trainer.step == 0 and records[-1]["reason"] == "STOP file found"
    assert [r["kind"] for r in records] == ["session", "checkpoint", "session"], "nothing was trained, so nothing is measured"
    assert not (trainer.run_dir / "STOP").exists()
    # A STOP file older than the process was left by one long gone, and is swept away.
    again = make(env, "early-stop", resume=resolve_resume("latest", trainer.run_dir))
    (again.run_dir / "STOP").write_text("", encoding="utf-8")
    os.utime(again.run_dir / "STOP", (1_000_000_000, 1_000_000_000))
    stop_after(again, 2)
    again.run()
    assert again.step == 2


def test_resume_refuses_a_checkpoint_of_another_kind(env, tmp_path):
    fake = tmp_path / "selfplay.pt"
    torch.save({"format": train_module.CHECKPOINT_FORMAT, "model": {}, "step": 3}, fake)
    with pytest.raises(RuntimeError, match="trained from games"):
        make(env, "other-kind", resume=fake)


def test_a_time_limit_stops_the_run_cleanly(env):
    trainer = make(env, "timed", run__hours=1e-9)
    trainer.run()
    records = metrics(trainer.run_dir)
    stop = records[-1]
    assert stop["event"] == "stop" and stop["finished"] is False and stop["reason"].startswith("time limit of")
    assert trainer.step < trainer.total_steps
    assert resolve_resume("latest", trainer.run_dir).exists()
    assert not (trainer.run_dir / "running.json").exists()


def test_matches_checkpoints_and_reports_on_their_schedule(env, monkeypatch):
    """Every interval at zero: a report, a match and a checkpoint after every step; the best is
    the match with the highest win rate, and the closing match has final_pairs."""
    rates = iter([0.30, 0.50, 0.45, 0.60] + [0.55] * 20)
    pairs = []

    def fake_match(engine, leaves, a, b, seeds, device, max_candidates, max_exchanges):
        pairs.append(len(seeds))
        return MatchResult(a.name, b, len(seeds), 1.0, 2.0, next(rates), 0.05, 420.0, 419.0, 24.0)

    monkeypatch.setattr(offline_module, "play_match", fake_match)
    monkeypatch.setattr(offline_module, "REPORT_SECONDS", 0.0)
    trainer = make(env, "scheduled", offline__eval_minutes=0.0, offline__checkpoint_minutes=0.0)
    trainer.run()
    records = metrics(trainer.run_dir)
    assert pairs == [2] * 16 + [3]
    train = [r for r in records if r["kind"] == "train"]
    assert [r["step"] for r in train] == list(range(1, 17))
    assert [r["pass"] for r in train] == [round(s / 8, 3) for s in range(1, 17)]
    hours = [r["hours"] for r in records]
    assert hours == sorted(hours)
    assert [r["step"] for r in records if r["kind"] == "best"] == [1, 2, 4]
    assert (trainer.best_step, trainer.best_win_rate) == (4, 0.60)
    assert torch.load(trainer.checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)["step"] == 4
    assert sorted(p.name for p in trainer.checkpoint_dir.glob("step_*.pt")) == ["step_000000015.pt", "step_000000016.pt"]
    index = json.loads((trainer.checkpoint_dir / "index.json").read_text(encoding="utf-8"))
    assert index["best"]["step"] == 4 and [c["step"] for c in index["checkpoints"]] == [15, 16]


def test_a_gpu_that_is_not_there_is_refused(env):
    if torch.cuda.is_available():
        pytest.skip("a GPU is visible")
    with pytest.raises(RuntimeError, match="cannot see a GPU"):
        make(env, "no-gpu", run__device="cuda")


def test_old_step_files_are_pruned(env, monkeypatch):
    monkeypatch.setattr(offline_module, "play_match", lambda *a, **k: MatchResult("a", "b", 2, 0.0, 1.0, 0.5, 0.1, 400.0,
                                                                                  400.0, 24.0))
    trainer = make(env, "pruned")
    for step in (1, 2, 3, 4):
        trainer.step = step
        trainer.save()
    files = sorted(p.name for p in trainer.checkpoint_dir.glob("step_*.pt"))
    assert files == ["step_000000003.pt", "step_000000004.pt"]
    index = json.loads((trainer.checkpoint_dir / "index.json").read_text(encoding="utf-8"))
    assert [c["file"] for c in index["checkpoints"]] == files and index["best"] is None


# ---- refusals ----------------------------------------------------------------------------------


def test_it_needs_a_game_set_it_can_use(env, tmp_path):
    with pytest.raises(GamesError, match="offline.games is not set"):
        make(env, "no-set", offline__games="")
    with pytest.raises(GamesError, match="no game set"):
        make(env, "missing-set", offline__games=str(tmp_path / "nowhere"))
    with pytest.raises(GamesError, match="leave values"):
        make(env, "other-leaves", run__leaves="zero")


def test_offline_settings_are_validated():
    for item, message in (("offline.passes=0", "offline.passes"), ("offline.batch_size=0", "offline.batch_size"),
                          ("offline.learning_rate=0", "offline.learning_rate"),
                          ("offline.validation_games=-1", "offline.validation_games"),
                          ("offline.validate_steps=0", "offline.validate_steps"),
                          ("offline.aux_share=1.5", "offline.aux_share"), ("offline.eval_pairs=1", "offline.eval_pairs"),
                          ("offline.final_pairs=1", "offline.final_pairs"),
                          ("offline.keep_checkpoints=0", "offline.keep_checkpoints")):
        with pytest.raises(ConfigError, match=message):
            load_config(None, [item])


# ---- Prefetcher --------------------------------------------------------------------------------


def test_the_prefetcher_hands_over_batches_in_order():
    prefetch = Prefetcher(lambda step: {"step": step}, 3, 9, depth=2)
    try:
        assert [prefetch.get() for _ in range(6)] == [(s, {"step": s}) for s in range(3, 9)]
    finally:
        prefetch.close()
    assert not prefetch.thread.is_alive()


def test_the_prefetcher_passes_errors_on_and_closes_while_full():
    def make_batch(step):
        if step == 2:
            raise ValueError("bad game")
        return step

    prefetch = Prefetcher(make_batch, 0, 10)
    assert prefetch.get() == (0, 0) and prefetch.get() == (1, 1)
    with pytest.raises(ValueError, match="bad game"):
        prefetch.get()
    prefetch.close()

    full = Prefetcher(lambda step: step, 0, 1_000, depth=2)
    deadline = time.time() + 5
    while full.queue.qsize() < 2 and time.time() < deadline:
        time.sleep(0.01)
    started = time.time()
    full.close()
    assert not full.thread.is_alive() and time.time() - started < 3


# ---- scrabble-ai train-games -------------------------------------------------------------------


@pytest.fixture
def stub_game_trainer(monkeypatch):
    captured: dict = {}

    class StubTrainer:
        def __init__(self, config, resume=None, quiet=False):
            captured["config"], captured["resume"] = config, resume

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(offline_module, "GameTrainer", StubTrainer)
    return captured


def run_cli(argv: list[str]) -> None:
    try:
        cli.main(argv)
    except SystemExit as exit_:
        pytest.fail(f"scrabble-ai {' '.join(argv)} exited: {exit_}")


def test_train_games_applies_the_games_defaults_under_the_users_settings(stub_game_trainer):
    run_cli(["train-games", "--games", "big-set", "--name", "g1", "--hours", "2", "--set", "model.head=flat",
             "--set", "selfplay.max_candidates=48"])
    config = stub_game_trainer["config"]
    assert stub_game_trainer["ran"] and stub_game_trainer["resume"] is None
    assert (config.offline.games, config.run.name, config.run.hours) == ("big-set", "g1", 2.0)
    assert config.model.head == "flat" and config.selfplay.max_candidates == 48, "what the user set wins"
    assert (config.model.win_head, config.model.aux_heads, config.selfplay.win_weight) == ("wdl", True, 0.9)


def test_train_games_lets_a_config_file_override_the_games_defaults(stub_game_trainer, tmp_path, capsys):
    """OFFLINE_DEFAULTS come "before anything the user set". A settings file sets one of them by
    giving it a value other than the built-in one; restating the built-in value (as
    configs/train.toml does for every setting) is not a choice, or passing that file would
    quietly undo what training from games needs. `--set` outranks both."""
    config_file = tmp_path / "mine.toml"
    config_file.write_text('[model]\nhead = "flat"\n[selfplay]\nwin_weight = 0.3\nmax_candidates = 48\n', encoding="utf-8")
    run_cli(["train-games", "--config", str(config_file), "--games", "big-set", "--name", "g2", "--set", "model.win_head=binary"])
    config = stub_game_trainer["config"]
    assert (config.selfplay.win_weight, config.selfplay.max_candidates) == (0.3, 48), "the file's own choices stand"
    assert config.model.head == "pooled", "the built-in value restated in a file is not a choice"
    assert config.model.win_head == "binary", "--set outranks the games default"
    assert config.model.aux_heads is True, "a default nobody touched still applies"
    said = capsys.readouterr().out
    assert "model.head=pooled" in said and "model.aux_heads=true" in said
    assert "win_weight" not in said and "win_head" not in said, "only the defaults in force are announced"


def test_train_games_with_the_standard_settings_file_keeps_the_games_defaults(stub_game_trainer):
    from scrabble_ai.config import OFFLINE_DEFAULTS
    from scrabble_ai.lexicon import REPO_ROOT

    run_cli(["train-games", "--config", str(REPO_ROOT / "configs" / "train.toml"), "--games", "big-set", "--name", "g2b"])
    settings = stub_game_trainer["config"].to_dict()
    assert {key: settings[key.split(".")[0]][key.split(".")[1]] for key in OFFLINE_DEFAULTS} == OFFLINE_DEFAULTS


def test_train_games_resume_takes_the_checkpoints_settings(stub_game_trainer, monkeypatch, tmp_path):
    from scrabble_ai.config import Config

    checkpoints = tmp_path / "g3" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_000000004.pt").write_bytes(b"")
    (checkpoints / "latest.txt").write_text("step_000000004.pt", encoding="utf-8")

    def checkpoint_config(path):
        config = Config()
        config.offline.games = "saved-set"
        config.model.head = "flat"  # as the run was trained
        return config

    monkeypatch.setattr(train_module, "load_checkpoint_config", checkpoint_config)
    run_cli(["train-games", "--name", "g3", "--set", f"run.out_dir={tmp_path}", "--resume", "latest",
             "--set", "offline.learning_rate=0.0001"])
    config = stub_game_trainer["config"]
    assert stub_game_trainer["resume"] == checkpoints / "step_000000004.pt"
    assert (config.offline.games, config.model.head, config.offline.learning_rate) == ("saved-set", "flat", 1e-4)
    assert config.run.name == "g3"


def test_train_games_refusals_exit_with_a_message(env):
    with pytest.raises(SystemExit, match="offline.games is not set"):
        cli.main(["train-games", "--name", "x", "--set", f"run.out_dir={env.root / 'runs'}", "--set", "run.lexicon=TEST",
                  "--set", f"run.leaves={env.fitted}", "--set", "run.device=cpu"])
    with pytest.raises(SystemExit, match="unknown setting offline.pases"):
        cli.main(["train-games", "--set", "offline.pases=3"])


def test_scrabble_ai_games_makes_a_set(env, tmp_path, capsys):
    run_cli(["games", "--name", "cli-set", "--games", "30", "--dir", str(tmp_path), "--lexicon", "TEST",
             "--leaves", str(env.fitted), "--opening-plies", "2", "--opening-temperature", "3"])
    meta = games_module.read_meta(tmp_path / "cli-set")
    assert (meta["games"], meta["opening_plies"], meta["opening_temperature"]) == (30, 2, 3.0)
    assert "cli-set: 30 games" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="other settings"):
        cli.main(["games", "--name", "cli-set", "--games", "40", "--dir", str(tmp_path), "--lexicon", "TEST",
                  "--leaves", str(env.fitted), "--opening-plies", "3"])
    with pytest.raises(SystemExit, match="error"):
        cli.main(["games", "--name", "bad name", "--games", "5", "--dir", str(tmp_path), "--lexicon", "TEST",
                  "--leaves", str(env.fitted)])
