"""`selfplay.Actor` end to end on CPU: what lands in the replay buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scrabble_ai import _core
from scrabble_ai import selfplay
from scrabble_ai.config import ModelConfig
from scrabble_ai.model import ValueNet
from scrabble_ai.replay import Replay
from scrabble_ai.selfplay import Actor, ActorStats, Scorer, returns

M_SCORE, M_SPREAD, M_TERMINAL, M_KIND = _core.M_SCORE, _core.M_SPREAD, _core.M_TERMINAL, _core.M_KIND
M_DRAWS = _core.M_DRAWS
OPP_KIND, OPP_TILES, OPP_SCORE = _core.M_OPPONENT_LAST_KIND, _core.M_OPPONENT_LAST_TILES, _core.M_OPPONENT_LAST_SCORE


class RecordingReplay(Replay):
    """A replay that also keeps a copy of every block added (one block per finished game)."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.blocks: list[dict[str, np.ndarray]] = []

    def add(self, rows):
        self.blocks.append({name: np.array(array, copy=True) for name, array in rows.items()})
        super().add(rows)


def play(engine, leaves, games, *, slots=4, temperature=0.0, td_lambda=1.0, scorer=None, first_seed=1000,
         max_candidates=12, max_exchanges=3):
    """Run an Actor until `games` games have finished. Returns the replay, one record per finished
    game (seed, final scores, turns, the values used during play) and the summed stats."""
    replay = RecordingReplay(200_000)
    actor = Actor(engine, leaves, replay, slots, max_candidates, max_exchanges, td_lambda, torch.device("cpu"),
                  first_seed=first_seed, sampling_seed=123)
    finished = []
    original = actor._finish

    def finish(slot, stats):
        n = int(actor.length[slot])
        finished.append({
            "seed": int(actor.seeds[slot]),
            "scores": tuple(int(s) for s in actor.buffers.scores[slot]),
            "turns": int(actor.env.turn(slot)),
            "values": actor.t_value[slot, :n].copy(),
            "length": n,
            "truncated": bool(actor.truncated[slot]),
        })
        original(slot, stats)

    actor._finish = finish
    total = ActorStats()
    steps = 0
    while total.games < games:
        stats = actor.step(scorer, temperature)
        total.merge(stats)
        steps += 1
        assert steps < 20_000, "games are not finishing"
    assert len(finished) == len(replay.blocks) == total.games
    return replay, finished, total


def mover_final(scores, rows):
    """Final spread from the side of the mover of each stored row (seat 0 moves on even turns)."""
    spread0 = scores[0] - scores[1]
    return np.where(np.arange(rows) % 2 == 0, spread0, -spread0).astype(np.float64)


def check_opponent_last_turn(block, meta):
    """meta[M_OPPONENT_LAST_*] of each stored afterstate describes the previous turn of the same
    game, which is the previous stored afterstate (every turn is stored except a final going-out
    move, which has no successor). The first move of a game has no previous turn."""
    assert meta.shape[1] == _core.META
    assert meta[0, [OPP_KIND, OPP_TILES, OPP_SCORE]].tolist() == [0, 0, 0]
    previous_kind = meta[:-1, M_KIND]
    # Afterstate kind 0 placement / 1 exchange / 2 pass -> last-turn kind 1 / 2 / 3.
    np.testing.assert_array_equal(meta[1:, OPP_KIND], previous_kind + 1)
    placed = ((block["board"] & np.uint8(_core.CODE_PLACED)) != 0).sum(axis=1)
    # Tiles placed (counted on the board, not M_DRAWS, which is capped by the bag) or exchanged.
    previous_tiles = np.where(previous_kind == 0, placed[:-1], np.where(previous_kind == 1, meta[:-1, M_DRAWS], 0))
    np.testing.assert_array_equal(meta[1:, OPP_TILES], previous_tiles)
    np.testing.assert_array_equal(meta[1:, OPP_SCORE], meta[:-1, M_SCORE])
    # Placements mark at least one square as just placed; exchanges and passes mark none.
    np.testing.assert_array_equal(placed == 0, meta[:, M_KIND] != 0)


def check_game(block, record):
    rows = len(block["board"])
    turns = record["turns"]
    assert turns <= selfplay.MAX_TURNS
    meta = block["meta"].astype(np.int64)
    # Every turn is stored except a final move that went out.
    assert rows in (turns, turns - 1)
    assert not (meta[:, M_TERMINAL] == 1).any(), "a going-out afterstate was stored"
    if rows == turns:
        assert meta[-1, M_TERMINAL] == 2, "the game ended without going out, so on the sixth scoreless turn"
        assert (meta[-6:, M_KIND] != 0).all(), "the last six turns were passes or exchanges"
    assert (meta[:-1, M_TERMINAL] == 0).all()
    # Spreads chain from one mover to the next: spread[t+1] = -spread[t] + score[t+1].
    np.testing.assert_array_equal(meta[1:, M_SPREAD], -meta[:-1, M_SPREAD] + meta[1:, M_SCORE])
    check_opponent_last_turn(block, meta)

    final = mover_final(record["scores"], rows)
    np.testing.assert_array_equal(block["win_target"], (final > 0) + 0.5 * (final == 0))
    # Tiles are conserved in every stored afterstate: board + leave + unseen = 100.
    on_board = (block["board"] != 0).sum(axis=1)
    held = block["leave"].astype(np.int64).sum(axis=1)
    hidden = block["unseen"].astype(np.int64).sum(axis=1)
    np.testing.assert_array_equal(on_board + held + hidden, 100)
    # Board codes only use the documented bits.
    assert ((block["board"] & ~np.uint8(_core.CODE_LETTER | _core.CODE_BLANK | _core.CODE_PLACED | _core.CODE_OPPONENT)) == 0).all()
    assert ((block["board"] & _core.CODE_LETTER) <= 26).all()
    return final, meta


def test_monte_carlo_targets_and_greedy_games_match_the_engine(engine, leaves):
    replay, finished, stats = play(engine, leaves, 24, slots=3)
    total_rows = 0
    for block, record in zip(replay.blocks, finished):
        final, meta = check_game(block, record)
        # td_lambda = 1: value_target = final spread (mover's side) - spread right after the move.
        np.testing.assert_array_equal(block["value_target"], (final - meta[:, M_SPREAD]).astype(np.float32))
        total_rows += len(block["board"])
    assert stats.positions == total_rows == replay.total_added == len(replay)

    # Temperature 0 with no network picks candidate 0, the static best: these are greedy-leave games.
    seeds = np.array([r["seed"] for r in finished], np.uint64)
    bot = engine.play_pairs("greedy-leave", leaves, "greedy-leave", leaves, seeds)
    for i, record in enumerate(finished):
        assert record["scores"] == (bot["score_a"][2 * i], bot["score_b"][2 * i]), f"seed {record['seed']}"
        assert record["turns"] == bot["turns"][2 * i]
    assert stats.turns == sum(r["turns"] for r in finished)
    assert stats.score_sum == sum(sum(r["scores"]) for r in finished)
    assert stats.first_player_wins == sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in (r["scores"] for r in finished))
    # Without a network the value used in play is the leave value, so both error sums agree.
    assert stats.value_sq_error == pytest.approx(stats.leave_sq_error)
    # Seeds are consecutive from first_seed, each used once.
    assert sorted(r["seed"] for r in finished) == list(range(1000, 1000 + len(finished)))


def test_sampled_games_keep_the_invariants(engine, leaves):
    replay, finished, stats = play(engine, leaves, 30, slots=8, temperature=8.0, first_seed=5000)
    for block, record in zip(replay.blocks, finished):
        final, meta = check_game(block, record)
        np.testing.assert_array_equal(block["value_target"], (final - meta[:, M_SPREAD]).astype(np.float32))
    assert stats.positions == replay.total_added
    # The sampled games are not all the greedy game.
    seeds = np.array([r["seed"] for r in finished], np.uint64)
    bot = engine.play_pairs("greedy-leave", leaves, "greedy-leave", leaves, seeds)
    differ = sum(r["scores"] != (bot["score_a"][2 * i], bot["score_b"][2 * i]) for i, r in enumerate(finished))
    assert differ > len(finished) // 2


def test_td_zero_targets_bootstrap_from_the_next_leave_value(engine, leaves):
    """With no network, the value of an afterstate during play is its leave value, so with
    lambda = 0 every target but the last of a game is -(score[t+1] + leave_value[t+1])."""
    replay, finished, _ = play(engine, leaves, 20, slots=4, temperature=3.0, td_lambda=0.0, first_seed=7000)
    for block, record in zip(replay.blocks, finished):
        final, meta = check_game(block, record)
        target = block["value_target"]
        expected = -(meta[1:, M_SCORE].astype(np.float32) + block["leave_value"][1:])
        np.testing.assert_allclose(target[:-1], expected, rtol=0, atol=1e-4)
        # The last stored row has nothing stored after it: its target is Monte Carlo.
        assert target[-1] == pytest.approx(final[-1] - meta[-1, M_SPREAD], abs=1e-4)


def test_network_actor_targets_follow_returns(engine, leaves):
    torch.manual_seed(0)
    net = ValueNet(ModelConfig(blocks=1, channels=16, vector_hidden=16, head_hidden=16, value_channels=2))
    torch.nn.init.normal_(net.value.weight, std=0.3)
    torch.nn.init.normal_(net.win.weight, std=0.3)
    scorer = Scorer(net, engine, 50.0, torch.device("cpu"))
    replay, finished, stats = play(engine, leaves, 6, slots=3, temperature=1.0, td_lambda=0.7, scorer=scorer,
                                   first_seed=9000, max_candidates=8, max_exchanges=2)
    for block, record in zip(replay.blocks, finished):
        final, meta = check_game(block, record)
        n = record["length"]
        assert n == record["turns"]
        values = record["values"]
        # Recompute every turn's target from the values used in play, including the dropped
        # going-out row when there was one (its value is 0 and its row is not stored).
        rows = len(block["board"])
        spread_all = meta[:, M_SPREAD].astype(np.float64)
        score_all = meta[:, M_SCORE].astype(np.float64)
        if rows == n - 1:
            # The dropped final move went out: its value was 0, the mover's spread after it is
            # their final spread, and its score is what links that spread to the previous one.
            assert values[-1] == 0.0
            last_spread = mover_final(record["scores"], n)[-1]
            spread_all = np.append(spread_all, last_spread)
            score_all = np.append(score_all, last_spread + spread_all[-2])
        movers = (np.arange(n) % 2).astype(np.int8)
        expected = returns(movers, spread_all.astype(np.float32), score_all.astype(np.float32), values,
                           record["scores"][0] - record["scores"][1], 0.7)
        np.testing.assert_allclose(block["value_target"], expected[:rows], rtol=0, atol=1e-3)
        # The network's values are real numbers, not the leave values.
        assert not np.allclose(values[:rows], block["leave_value"])
    assert stats.value_sq_error != pytest.approx(stats.leave_sq_error)


def test_truncated_games_fall_back_to_monte_carlo(engine, leaves, monkeypatch):
    """Games longer than MAX_TURNS keep only their first MAX_TURNS afterstates, and their targets
    must not bootstrap across the missing tail even when td_lambda < 1."""
    monkeypatch.setattr(selfplay, "MAX_TURNS", 10)
    replay, finished, stats = play(engine, leaves, 8, slots=4, temperature=2.0, td_lambda=0.0, first_seed=11_000)
    assert all(r["turns"] > 10 for r in finished), "the test lexicon's games are longer than 10 turns"
    for block, record in zip(replay.blocks, finished):
        assert record["truncated"]
        assert len(block["board"]) == 10
        final = mover_final(record["scores"], 10)
        np.testing.assert_array_equal(block["value_target"], (final - block["meta"][:, M_SPREAD]).astype(np.float32))
    assert stats.positions == 10 * len(finished)


def test_state_round_trip_resumes_the_seed_counter(engine, leaves):
    replay = Replay(10_000)
    actor = Actor(engine, leaves, replay, 2, 8, 2, 1.0, torch.device("cpu"), first_seed=50, sampling_seed=4)
    for _ in range(5):
        actor.step(None, 5.0)
    state = actor.state()
    assert state["next_seed"] == 52
    other = Actor(engine, leaves, Replay(10_000), 2, 8, 2, 1.0, torch.device("cpu"), first_seed=0, sampling_seed=99)
    other.load_state(state)
    assert other.seeds.tolist() == [52, 53] and other.next_seed == 54
    assert torch.equal(other.generator.get_state(), actor.generator.get_state())
