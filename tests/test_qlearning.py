"""Deep Q-learning in the self-play trainer: `selfplay.returns(..., best=...)`, `Choice.best`,
`Actor(..., q_learning=True)` and its wiring from `train.target = "q"`.

The rule (docs in `returns`): `best[t]` is the highest score + value among the candidates of turn
t, a going-out candidate valued at 0, and the target of every turn but the last is
`-best[t+1]`; the last keeps its Monte Carlo value, and so does every turn of a game that was too
long to keep whole.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scrabble_ai import _core
from scrabble_ai import engine as engine_module
from scrabble_ai import selfplay
from scrabble_ai.config import load_config
from scrabble_ai.lexicon import LexiconError
from scrabble_ai.replay import Replay
from scrabble_ai.selfplay import Actor, ActorStats, CandidateBuffers, choose, returns

M_SCORE, M_SPREAD, M_TERMINAL = _core.M_SCORE, _core.M_SPREAD, _core.M_TERMINAL
CPU = torch.device("cpu")


class StubScorer:
    """Values an afterstate by a fixed rule of its codes, so every number is known: a few points
    per tile on the board, plus the leave value, plus the move's score to break ties."""

    def __call__(self, board, leave, unseen, meta, leave_value):
        tiles = ((board & _core.CODE_LETTER) != 0).sum(axis=1).astype(np.float32)
        value = 0.37 * tiles - 6.0 + 0.5 * leave_value + 0.01 * meta[:, M_SCORE]
        win = 1.0 / (1.0 + np.exp(-value / 20.0))
        return torch.from_numpy(value.astype(np.float32)), torch.from_numpy(win.astype(np.float32))


class RecordingReplay(Replay):
    def __init__(self):
        super().__init__(200_000)
        self.blocks: list[dict] = []

    def add(self, rows):
        self.blocks.append({name: np.array(array, copy=True) for name, array in rows.items()})
        super().add(rows)


# ---- returns -----------------------------------------------------------------------------------


def test_q_targets_by_hand():
    mover = np.array([0, 1, 0, 1, 0], np.int8)
    spread_after = np.array([12, -5, 20, -8, 30], np.float32)  # each mover's spread right after their move
    score = np.array([12, 7, 25, 3, 40], np.float32)
    value = np.array([1, 2, 3, 4, 5], np.float32)
    best = np.array([10.0, 7.5, -3.0, 4.0, 99.0], np.float32)
    # Seat 0 won by 25: the last mover (seat 0) gained 25 - 30 = -5 after their move.
    expected = np.array([-7.5, 3.0, -4.0, -99.0, -5.0], np.float32)
    for td_lambda in (0.0, 0.7, 1.0):
        g = returns(mover, spread_after, score, value, 25, td_lambda, best)
        assert g.dtype == np.float32 and np.array_equal(g, expected), td_lambda
    # The last turn is the Monte Carlo value, as without `best`.
    mc = returns(mover, spread_after, score, value, 25, 1.0)
    assert g[-1] == mc[-1]
    assert not np.array_equal(g[:-1], mc[:-1])
    # The inputs are not changed.
    assert best.tolist() == [10.0, 7.5, -3.0, 4.0, 99.0] and spread_after.tolist() == [12, -5, 20, -8, 30]


def test_q_targets_of_short_games():
    one = returns(np.array([1], np.int8), np.array([4], np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32),
                  -10, 0.7, np.array([123.0], np.float32))
    assert one.tolist() == [6.0], "seat 1 won by 10 and stood at +4: Monte Carlo"
    two = returns(np.array([0, 1], np.int8), np.array([3, -1], np.float32), np.zeros(2, np.float32),
                  np.zeros(2, np.float32), 8, 0.7, np.array([0.0, 11.0], np.float32))
    assert two.tolist() == [-11.0, -7.0]


# ---- choose: Choice.best -----------------------------------------------------------------------


def crafted_buffers() -> CandidateBuffers:
    """Three slots of four candidates: slot 0 decides among 3, slot 1 is empty, slot 2 decides
    among 4, of which candidate 1 goes out (30 points with the bonus, ahead by 15)."""
    b = CandidateBuffers(3, 4)
    b.status[:] = [_core.STATUS_DECISION, _core.STATUS_EMPTY, _core.STATUS_DECISION]
    b.count[:] = [3, 0, 4]
    b.meta[0, :, M_SCORE] = [10, 20, 5, 77]
    b.leave_value[0] = [1.0, -2.0, 30.0, 999.0]  # the fourth is padding: never counted
    b.meta[2, :, M_SCORE] = [0, 30, 8, 12]
    b.meta[2, 1, M_TERMINAL] = 1
    b.meta[2, 1, M_SPREAD] = 15
    b.leave_value[2] = [2.0, 50.0, -1.0, 0.0]
    return b


class LinearScorer:
    """value = 3 * leave value + 1; win given per candidate."""

    def __init__(self, wins):
        self.wins = np.asarray(wins, np.float32)

    def __call__(self, board, leave, unseen, meta, leave_value):
        return torch.from_numpy(3.0 * leave_value + 1.0), torch.from_numpy(self.wins[: len(leave_value)])


def test_best_is_the_highest_score_plus_value_with_going_out_worth_nothing_more():
    b = crafted_buffers()
    generator = torch.Generator().manual_seed(0)
    scorer = LinearScorer([0.99, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
    choice = choose(b, scorer, 0.0, 0.0, 100.0, CPU, generator)
    assert choice.slots.tolist() == [0, 2]
    # Slot 0: 10 + 4, 20 - 5, 5 + 91. Slot 2: 0 + 7, 30 + 0 (went out), 8 - 2, 12 + 1.
    assert choice.best.tolist() == [96.0, 30.0]
    assert choice.actions.tolist() == [2, 1], "by points alone the best is the one chosen"
    assert choice.value.tolist() == [91.0, 0.0]
    # Choosing by win chance picks other moves; `best` stays the same.
    by_win = choose(b, scorer, 0.0, 1.0, 100.0, CPU, generator)
    assert by_win.actions.tolist()[0] == 0 and by_win.best.tolist() == [96.0, 30.0]
    sampled = choose(b, scorer, 50.0, 0.0, 100.0, CPU, generator)
    assert sampled.best.tolist() == [96.0, 30.0]
    # The static evaluator: value = leave value.
    static = choose(b, None, 0.0, 0.0, 100.0, CPU, generator)
    assert static.best.tolist() == [35.0, 30.0]
    # Only some slots.
    only = choose(b, scorer, 0.0, 0.0, 100.0, CPU, generator, only=np.array([2]))
    assert only.slots.tolist() == [2] and only.best.tolist() == [30.0]
    # Nothing to decide.
    b.status[:] = _core.STATUS_EMPTY
    none = choose(b, scorer, 0.0, 0.0, 100.0, CPU, generator)
    assert none.best.shape == (0,) and none.best.dtype == np.float32


def test_best_on_real_candidates(engine, leaves):
    """Against the candidates the environment writes, best is the maximum over each slot's own."""
    env = _core.BatchEnv(engine, leaves, 6, 10, 2, 0)
    b = CandidateBuffers(6, 10)
    for slot in range(6):
        env.reset(slot, 4_000 + slot, "external", "external")
    generator = torch.Generator().manual_seed(1)
    scorer = StubScorer()
    checked = 0
    for _ in range(30):
        b.prepare(env)
        choice = choose(b, scorer, 1.0, 0.0, 100.0, CPU, generator)
        for j, slot in enumerate(choice.slots):
            k = b.count[slot]
            value, _ = scorer(b.board[slot, :k], b.leave[slot, :k], b.unseen[slot, :k], b.meta[slot, :k], b.leave_value[slot, :k])
            value = np.where(b.meta[slot, :k, M_TERMINAL] == 1, 0.0, value.numpy())
            assert choice.best[j] == pytest.approx(float((b.meta[slot, :k, M_SCORE] + value).max()), abs=1e-4)
            checked += 1
        actions = np.zeros(6, np.int32)
        actions[choice.slots] = choice.actions
        env.apply(actions)
    assert checked > 100


# ---- Actor -------------------------------------------------------------------------------------


def play(engine, leaves, monkeypatch, *, q_learning: bool, games: int = 6, temperature: float = 2.0):
    """Run an Actor until `games` games have finished, recording the `best` of every decision by
    (game seed, turn) and, for every finished game, its turns."""
    replay = RecordingReplay()
    holder = SimpleNamespace(actor=None)
    bests: dict[tuple[int, int], float] = {}
    real_choose = selfplay.choose

    def recording_choose(buffers, *args, **kwargs):
        choice = real_choose(buffers, *args, **kwargs)
        actor = holder.actor
        for j, slot in enumerate(choice.slots):
            bests[(int(actor.seeds[slot]), int(actor.length[slot]))] = float(choice.best[j])
        return choice

    monkeypatch.setattr(selfplay, "choose", recording_choose)
    actor = Actor(engine, leaves, replay, 4, 10, 2, 0.7, CPU, first_seed=2_000, sampling_seed=5, q_learning=q_learning)
    holder.actor = actor
    finished = []
    original = actor._finish

    def finish(slot, stats):
        n = int(actor.length[slot])
        finished.append(SimpleNamespace(seed=int(actor.seeds[slot]), n=n, truncated=bool(actor.truncated[slot]),
                                        scores=tuple(int(s) for s in actor.buffers.scores[slot]),
                                        meta=actor.t_meta[slot, :n].copy(), mover=actor.t_mover[slot, :n].copy()))
        original(slot, stats)

    actor._finish = finish
    total, steps = ActorStats(), 0
    while total.games < games:
        total.merge(actor.step(StubScorer(), temperature))
        steps += 1
        assert steps < 5_000
    assert len(finished) == len(replay.blocks)
    return replay, finished, bests


def monte_carlo(game) -> np.ndarray:
    spread0 = game.scores[0] - game.scores[1]
    final = np.where(game.mover == 0, spread0, -spread0)
    return (final - game.meta[:, M_SPREAD]).astype(np.float32)


def test_the_actor_stores_q_targets(engine, leaves, monkeypatch):
    replay, finished, bests = play(engine, leaves, monkeypatch, q_learning=True)
    differ = 0
    for block, game in zip(replay.blocks, finished):
        assert not game.truncated
        best = np.array([bests[(game.seed, t)] for t in range(game.n)], np.float32)
        expected = monte_carlo(game)
        expected[:-1] = -best[1:]
        keep = game.meta[:, M_TERMINAL] != 1
        assert np.allclose(block["value_target"], expected[keep], atol=1e-4), game.seed
        mc = monte_carlo(game)[keep]
        differ += int((~np.isclose(block["value_target"], mc)).sum())
        # The last turn kept, if it did not go out, keeps its Monte Carlo value.
        if keep[-1]:
            assert block["value_target"][-1] == pytest.approx(mc[-1])
        # The win target is the game's result either way.
        spread0 = game.scores[0] - game.scores[1]
        won = np.where(game.mover == 0, spread0, -spread0)[keep]
        assert np.array_equal(block["win_target"], ((won > 0) + 0.5 * (won == 0)).astype(np.float32))
    assert differ > 20, "Q targets are not the Monte Carlo ones"


def test_without_q_learning_the_actor_keeps_td_targets(engine, leaves, monkeypatch):
    replay, finished, bests = play(engine, leaves, monkeypatch, q_learning=False, games=3)
    for block, game in zip(replay.blocks, finished):
        best = np.array([bests[(game.seed, t)] for t in range(game.n)], np.float32)
        keep = game.meta[:, M_TERMINAL] != 1
        q = monte_carlo(game)
        q[:-1] = -best[1:]
        assert not np.allclose(block["value_target"], q[keep], atol=1e-3)


def test_a_truncated_game_falls_back_to_monte_carlo(engine, leaves, monkeypatch):
    monkeypatch.setattr(selfplay, "MAX_TURNS", 6)  # every game is longer than this
    replay, finished, _ = play(engine, leaves, monkeypatch, q_learning=True, games=4)
    for block, game in zip(replay.blocks, finished):
        assert game.truncated and game.n == 6
        keep = game.meta[:, M_TERMINAL] != 1
        assert np.allclose(block["value_target"], monte_carlo(game)[keep]), game.seed


# ---- wiring from the settings ------------------------------------------------------------------


@pytest.fixture
def train_env(lexicon_dir, leaves, tmp_path, monkeypatch):
    fitted = tmp_path / "fitted.leaves"
    leaves.save(fitted)

    def lexicon_entry(name: str = "", config_path=None):
        name = name or "TEST"
        path = lexicon_dir / f"{name}.txt"
        if not path.exists():
            raise LexiconError(f"unknown lexicon {name!r}")
        return name, path

    monkeypatch.setattr(engine_module, "lexicon_entry", lexicon_entry)
    return SimpleNamespace(root=tmp_path, fitted=fitted)


@pytest.mark.parametrize("model", [
    ["model.kind=transformer", "model.head=pooled", "model.win_head=wdl", "model.aux_heads=true"],
    ["model.kind=cnn-global", "model.blocks=3"],
])
def test_self_play_with_q_targets_trains_the_new_kinds_of_network(train_env, model):
    """The self-play trainer only calls `forward`: every kind of network fits it, and Q targets
    come out finite from a warm start and from the network's own play."""
    from scrabble_ai.train import Trainer

    config = load_config(None, [
        "run.name=q-kinds", f"run.out_dir={train_env.root}", "run.lexicon=TEST", f"run.leaves={train_env.fitted}",
        "run.device=cpu", "model.blocks=1", "model.channels=16", "model.vector_hidden=16", "model.head_hidden=16",
        "model.belief_samples=0", "selfplay.games=4", "selfplay.max_candidates=8", "selfplay.max_exchanges=2",
        "warmstart.games=8", "warmstart.max_candidates=8", "warmstart.steps=2", "train.batch_size=16",
        "train.min_replay=16", "train.target=q", *model])
    trainer = Trainer(config, quiet=True)
    try:
        assert trainer.warm_start(lambda: False)
        assert trainer.step == 2 and all(np.isfinite(loss) for entry in trainer.losses for loss in entry)
        scorer = trainer._scorer(trainer.actor_net)
        games = 0
        for _ in range(400):
            games += trainer.actor.step(scorer, 1.0).games
            if games >= 2:
                break
        assert games >= 2
        replay = trainer.replay.sample(64, np.random.default_rng(0))
        assert np.isfinite(replay["value_target"]).all()
        trainer.train_step()
        assert np.isfinite(trainer.losses[-1]).all()
    finally:
        if trainer.writer is not None:
            trainer.writer.close()


@pytest.mark.parametrize("target,expected", [("td", False), ("q", True)])
def test_train_target_chooses_the_learning_rule(train_env, target, expected):
    from scrabble_ai.train import Trainer

    config = load_config(None, [f"run.name=q-{target}", f"run.out_dir={train_env.root}", "run.lexicon=TEST",
                                f"run.leaves={train_env.fitted}", "run.device=cpu", "model.blocks=1", "model.channels=16",
                                "model.vector_hidden=16", "model.head_hidden=16", "selfplay.games=4",
                                "selfplay.max_candidates=8", "selfplay.max_exchanges=2", f"train.target={target}"])
    trainer = Trainer(config, quiet=True)
    try:
        assert trainer.actor.q_learning is expected
    finally:
        if trainer.writer is not None:
            trainer.writer.close()
