"""`evaluate`: paired statistics and matches between bots and networks."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scrabble_ai.config import ModelConfig
from scrabble_ai.evaluate import NetPlayer, eval_seeds, play_match, summarize
from scrabble_ai.model import ValueNet
from scrabble_ai.selfplay import Scorer

CPU = torch.device("cpu")


def untrained_baseline_net() -> ValueNet:
    torch.manual_seed(0)
    return ValueNet(ModelConfig(blocks=1, channels=16, vector_hidden=16, head_hidden=16, value_channels=2)).eval()


def fast_scorer(engine) -> Scorer:
    """An untrained leave-baseline network. Batches are padded to 32 rather than 512 to keep the
    CPU tests quick; padding itself is tested in test_scorer.py."""
    scorer = Scorer(untrained_baseline_net(), engine, 50.0, CPU)
    scorer.PAD = 32
    return scorer


def test_summarize_by_hand():
    # Pairs: A's spread moving first / second.
    first = np.array([10.0, -4.0, 6.0])
    second = np.array([2.0, 0.0, -6.0])
    result = summarize("A", "B", first, second, np.array([400.0, 380, 350, 410, 300, 390]),
                       np.array([390.0, 384, 344, 412, 306, 392]), np.array([20.0, 22, 24, 26, 21, 23]))
    # Per-pair means: (10 + 2) / 2 = 6, (-4 + 0) / 2 = -2, (6 - 6) / 2 = 0.  Mean = 4/3.
    assert result.spread == pytest.approx(4 / 3)
    # Deviations 14/3, -10/3, -4/3: squares sum to (196 + 100 + 16) / 9 = 104/3; variance (n - 1 = 2)
    # = 52/3; half-width = 1.96 * sqrt(52/3) / sqrt(3) = 1.96 * sqrt(52) / 3.
    assert result.spread_ci == pytest.approx(1.96 * math.sqrt(52) / 3)
    # Wins: first games 1, 0, 1; second games 1, tie 0.5, 0 -> per pair 1, 0.25, 0.5 -> mean 7/12.
    assert result.win_rate == pytest.approx(7 / 12)
    # Deviations 5/12, -4/12, -1/12: squares sum 42/144 = 7/24; variance 7/48;
    # half-width = 1.96 * sqrt(7/48) / sqrt(3) = 1.96 * sqrt(7) / 12.
    assert result.win_ci == pytest.approx(1.96 * math.sqrt(7) / 12)
    assert result.score_a == pytest.approx(2230 / 6)
    assert result.score_b == pytest.approx(2228 / 6)
    assert result.turns == pytest.approx(136 / 6)
    assert result.pairs == 3
    assert not result.significant  # 1.33 < 4.71
    assert "A vs B: spread +1.3 +/- 4.7" in result.line()


def test_summarize_significance_and_single_pair():
    result = summarize("A", "B", np.array([30.0, 40.0, 35.0]), np.array([20.0, 30.0, 25.0]),
                       np.zeros(6), np.zeros(6), np.zeros(6))
    assert result.spread == 30.0 and result.spread_ci == pytest.approx(1.96 * 5 / math.sqrt(3))
    assert result.significant and result.win_rate == 1.0 and result.win_ci == 0.0
    one = summarize("A", "B", np.array([5.0]), np.array([-5.0]), np.zeros(2), np.zeros(2), np.zeros(2))
    assert one.spread == 0.0 and math.isnan(one.spread_ci) and not one.significant


@pytest.mark.parametrize("bot", ["random", "greedy", "greedy-leave"])
def test_a_bot_against_itself_has_zero_spread_in_every_pair(engine, leaves, bot):
    seeds = eval_seeds(40)
    result = play_match(engine, leaves, bot, bot, seeds)
    assert result.spread == 0.0 and result.spread_ci == 0.0
    assert result.win_rate == 0.5 and result.win_ci == 0.0
    assert result.score_a == result.score_b
    raw = engine.play_pairs(bot, leaves, bot, leaves, seeds)
    np.testing.assert_array_equal(raw["spread_first"], -raw["spread_second"])


def test_bots_differ(engine, leaves):
    result = play_match(engine, leaves, "greedy", "random", eval_seeds(40))
    assert result.spread > 0 and result.significant


def test_untrained_baseline_network_plays_exactly_like_greedy_leave(engine, leaves):
    """With leave_baseline on, an untrained network values every candidate at score + leave value,
    so it picks candidate 0, the static best, which is the greedy-leave bot's move. Both seats then
    play the same game from the same seed, so every pair's spread is exactly 0."""
    net = NetPlayer("net", fast_scorer(engine))
    seeds = eval_seeds(12)
    # Few slots, so slots are reused for later games.
    result = play_match(engine, leaves, net, "greedy-leave", seeds, CPU, slots=5)
    assert result.spread == 0.0 and result.spread_ci == 0.0, result.line()
    bots = play_match(engine, leaves, "greedy-leave", "greedy-leave", seeds)
    assert (result.score_a, result.score_b, result.turns) == (bots.score_a, bots.score_b, bots.turns)
    swapped = play_match(engine, leaves, "greedy-leave", net, seeds, CPU, max_candidates=2, max_exchanges=1)
    assert swapped.spread == 0.0 and swapped.spread_ci == 0.0, swapped.line()
    both = play_match(engine, leaves, net, NetPlayer("net2", net.scorer), seeds, CPU, max_candidates=8, max_exchanges=2)
    assert both.spread == 0.0 and both.spread_ci == 0.0


def test_network_seat_bookkeeping_matches_the_engine(engine, leaves):
    """A network that plays greedy-leave against greedy must give exactly the engine's own
    greedy-leave vs greedy result: same spreads by seat, same scores, same turns."""
    net = NetPlayer("net", fast_scorer(engine))
    seeds = eval_seeds(16, offset=3)
    engine_result = play_match(engine, leaves, "greedy-leave", "greedy", seeds)
    assert engine_result.spread != 0.0
    for a, b, sign in ((net, "greedy", 1), ("greedy", net, -1)):
        result = play_match(engine, leaves, a, b, seeds, CPU, slots=7)
        assert result.spread == pytest.approx(sign * engine_result.spread, abs=1e-12)
        assert result.spread_ci == pytest.approx(engine_result.spread_ci, abs=1e-12)
        assert result.turns == engine_result.turns
        if sign == 1:
            assert (result.score_a, result.score_b) == (engine_result.score_a, engine_result.score_b)
            assert result.win_rate == engine_result.win_rate
        else:
            assert (result.score_a, result.score_b) == (engine_result.score_b, engine_result.score_a)
            assert result.win_rate == pytest.approx(1 - engine_result.win_rate)


def test_a_network_needs_a_device(engine, leaves):
    net = NetPlayer("net", fast_scorer(engine))
    with pytest.raises(ValueError, match="device"):
        play_match(engine, leaves, net, "greedy", eval_seeds(2))


def test_eval_seeds_are_reproducible_and_far_from_self_play():
    seeds = eval_seeds(5, offset=2)
    assert seeds.dtype == np.uint64
    assert seeds.tolist() == [7_000_000_000_002 + i for i in range(5)]
