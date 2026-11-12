"""`selfplay.returns`: the two-player sign convention for value targets.

Every expected number in the hand-built games below is worked out by hand in the comments.
Convention under test: a value belongs to an afterstate and is from the side of the player who
made that move; `spread_after[t]` is that mover's spread right after move t (end-of-game bonus for
going out included in the move's score); `final_spread_seat0` is seat 0's final score minus seat 1's.
"""

from __future__ import annotations

import numpy as np
import pytest

from scrabble_ai.selfplay import returns


def f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def game_arrays(movers, scores, values):
    """Spreads after each move for an alternating game that starts 0-0 and has no end-of-game
    adjustment before the last move: spread_after[t] = -spread_after[t-1] + score[t]."""
    spread = []
    previous = 0.0
    for t, score in enumerate(scores):
        current = (-previous if t > 0 else 0.0) + score
        spread.append(current)
        previous = current
    return np.asarray(movers, np.int8), f32(spread), f32(scores), f32(values)


# ---- hand-built games ------------------------------------------------------------------------


def test_two_moves_second_player_goes_out():
    # t0 seat 0 scores 10 -> seat 0 spread after = 10.
    # t1 seat 1 scores 25 including the going-out bonus -> seat 1 spread after = -10 + 25 = 15.
    # Final: seat 1 leads by 15, so final_spread_seat0 = -15.
    # MC: t0 (seat 0): -15 - 10 = -25.   t1 (seat 1): 15 - 15 = 0 (nothing left after going out).
    movers, spread, score, value = game_arrays([0, 1], [10, 25], [7.0, 0.0])
    assert list(spread) == [10, 15]
    mc = returns(movers, spread, score, value, -15, 1.0)
    assert mc.tolist() == [-25.0, 0.0]
    # lambda = 0: g[0] = -(score[1] + value[1]) = -(25 + 0) = -25; the last target is MC.
    assert returns(movers, spread, score, value, -15, 0.0).tolist() == [-25.0, 0.0]
    assert mc.dtype == np.float32


def test_three_moves_first_player_goes_out_all_lambdas():
    # t0 seat 0 scores 20 -> spread 20.
    # t1 seat 1 scores 30 -> spread -20 + 30 = 10.
    # t2 seat 0 scores 15 + bonus 2 x 4 = 23 and goes out -> spread -10 + 23 = 13. Final seat 0 +13.
    movers, spread, score, value = game_arrays([0, 1, 0], [20, 30, 23], [5.0, -3.0, 0.0])
    assert list(spread) == [20, 10, 13]
    # MC: t0 seat 0: 13 - 20 = -7.  t1 seat 1 (did not make the last move): -13 - 10 = -23.
    #     t2 seat 0: 13 - 13 = 0.
    assert returns(movers, spread, score, value, 13, 1.0).tolist() == [-7.0, -23.0, 0.0]
    # lambda = 0: g[2] = 0;  g[1] = -(23 + value[2]) = -(23 + 0) = -23;
    #             g[0] = -(30 + value[1]) = -(30 - 3) = -27.
    assert returns(movers, spread, score, value, 13, 0.0).tolist() == [-27.0, -23.0, 0.0]
    # lambda = 0.5: g[2] = 0;  g[1] = -(23 + 0.5 * 0 + 0.5 * 0) = -23;
    #               g[0] = -(30 + 0.5 * (-3) + 0.5 * (-23)) = -(30 - 1.5 - 11.5) = -17.
    assert returns(movers, spread, score, value, 13, 0.5).tolist() == [-17.0, -23.0, 0.0]


def test_four_moves_with_an_exchange_and_the_second_player_going_out():
    # t0 seat 0 scores 12 -> spread 12.
    # t1 seat 1 scores 8  -> spread -12 + 8 = -4.
    # t2 seat 0 exchanges (0) -> spread 4.
    # t3 seat 1 scores 22 + bonus 8 = 30 and goes out -> spread -4 + 30 = 26.  Final seat 0: -26.
    movers, spread, score, value = game_arrays([0, 1, 0, 1], [12, 8, 0, 30], [4.0, -6.0, 2.0, 0.0])
    assert list(spread) == [12, -4, 4, 26]
    # MC: t0: -26 - 12 = -38;  t1: 26 - (-4) = 30;  t2: -26 - 4 = -30;  t3: 26 - 26 = 0.
    assert returns(movers, spread, score, value, -26, 1.0).tolist() == [-38.0, 30.0, -30.0, 0.0]
    # lambda = 0: g = [-(8 + (-6)), -(0 + 2), -(30 + 0), 0] = [-2, -2, -30, 0].
    assert returns(movers, spread, score, value, -26, 0.0).tolist() == [-2.0, -2.0, -30.0, 0.0]
    # lambda = 0.8:
    #   g[3] = 0
    #   g[2] = -(30 + 0.2 * 0 + 0.8 * 0)          = -30
    #   g[1] = -(0 + 0.2 * 2 + 0.8 * (-30))       = -(0.4 - 24)          = 23.6
    #   g[0] = -(8 + 0.2 * (-6) + 0.8 * 23.6)     = -(8 - 1.2 + 18.88)   = -25.68
    got = returns(movers, spread, score, value, -26, 0.8)
    np.testing.assert_allclose(got, [-25.68, 23.6, -30.0, 0.0], rtol=0, atol=1e-4)


def test_game_ending_on_the_sixth_scoreless_turn():
    # t0 seat 0 scores 18 (spread 18), then six scoreless turns t1..t6; t6 (seat 0) is the sixth.
    # Spreads after: seat 1 rows -18, seat 0 rows +18.
    # End: each side loses its own rack: seat 0 holds 9 points, seat 1 holds 14.
    # Final: seat 0 = 18 - 9 = 9, seat 1 = 0 - 14 = -14, final_spread_seat0 = 23.
    movers = [0, 1, 0, 1, 0, 1, 0]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.5]  # t6 is not "going out": its value is a real estimate
    movers, spread, score, value = game_arrays(movers, [18, 0, 0, 0, 0, 0, 0], values)
    assert list(spread) == [18, -18, 18, -18, 18, -18, 18]
    # MC: seat 0 rows: 23 - 18 = 5;  seat 1 rows: -23 - (-18) = -5.
    # The last mover (seat 0) gains 5 = opponent's rack 14 - own rack 9.
    assert returns(movers, spread, score, value, 23, 1.0).tolist() == [5, -5, 5, -5, 5, -5, 5]
    # lambda = 0: g[t] = -(0 + value[t+1]) for t < 6 (scores after t0 are all 0); g[6] = MC = 5.
    assert returns(movers, spread, score, value, 23, 0.0).tolist() == [-2, -3, -4, -5, -6, 1.5, 5]
    # lambda = 0.5, from the end:
    #   g[6] = 5
    #   g[5] = -(0 + 0.5 * (-1.5) + 0.5 * 5)   = -(-0.75 + 2.5)  = -1.75
    #   g[4] = -(0.5 * 6 + 0.5 * (-1.75))      = -(3 - 0.875)    = -2.125
    #   g[3] = -(0.5 * 5 + 0.5 * (-2.125))     = -(2.5 - 1.0625) = -1.4375
    #   g[2] = -(0.5 * 4 + 0.5 * (-1.4375))    = -(2 - 0.71875)  = -1.28125
    #   g[1] = -(0.5 * 3 + 0.5 * (-1.28125))   = -(1.5 - 0.640625) = -0.859375
    #   g[0] = -(0.5 * 2 + 0.5 * (-0.859375))  = -(1 - 0.4296875) = -0.5703125
    got = returns(movers, spread, score, value, 23, 0.5)
    expected = [-0.5703125, -0.859375, -1.28125, -1.4375, -2.125, -1.75, 5.0]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_six_passes_from_the_start():
    # Nobody scores; seat 0 holds 12 points of tiles, seat 1 holds 8. Final: -12 vs -8 -> seat 0: -4.
    movers, spread, score, value = game_arrays([0, 1, 0, 1, 0, 1], [0] * 6, [1, 2, 3, 4, 5, 6])
    assert returns(movers, spread, score, value, -4, 1.0).tolist() == [-4, 4, -4, 4, -4, 4]
    # lambda = 0: g[t] = -value[t+1] for t < 5, g[5] = MC = 4.
    assert returns(movers, spread, score, value, -4, 0.0).tolist() == [-2, -3, -4, -5, -6, 4]


def test_degenerate_lengths():
    assert returns(np.zeros(0, np.int8), f32([]), f32([]), f32([]), 0, 0.5).shape == (0,)
    # One move: seat 0 scores 40 going out, final +40 -> MC 0 whatever lambda is.
    assert returns(np.zeros(1, np.int8), f32([40]), f32([40]), f32([0]), 40, 0.3).tolist() == [0.0]


def test_lambda_one_returns_monte_carlo_exactly_and_lambda_near_one_converges():
    movers, spread, score, value = game_arrays([0, 1, 0, 1], [12, 8, 0, 30], [400.0, -600.0, 200.0, 0.0])
    mc = returns(movers, spread, score, value, -26, 1.0)
    # Values do not leak into Monte Carlo targets, however large.
    assert mc.tolist() == [-38.0, 30.0, -30.0, 0.0]
    near = returns(movers, spread, score, value, -26, 1.0 - 1e-7)
    np.testing.assert_allclose(near, mc, rtol=0, atol=1e-3)


# ---- property tests --------------------------------------------------------------------------


def random_game(rng: np.random.Generator):
    """A random alternating game. Ends either by going out (last value 0, no adjustment) or on a
    sixth scoreless turn (the last six scores 0, then a rack adjustment)."""
    length = int(rng.integers(2, 40))
    goes_out = bool(rng.integers(0, 2)) or length < 6
    scores = rng.integers(0, 90, size=length).astype(np.float64)
    if not goes_out:
        scores[-6:] = 0
    movers = np.arange(length) % 2
    spread = np.zeros(length)
    for t in range(length):
        spread[t] = (-spread[t - 1] if t else 0.0) + scores[t]
    # Final spread from seat 0's side: the last mover's spread after, plus a rack adjustment.
    adjustment = 0.0 if goes_out else float(rng.integers(-30, 31))
    last_final = spread[-1] + adjustment
    final_seat0 = last_final if movers[-1] == 0 else -last_final
    values = rng.normal(0, 30, size=length)
    if goes_out:
        values[-1] = 0.0
    return movers.astype(np.int8), spread, scores, values, int(final_seat0)


def forward_view(movers, spread, scores, values, final_seat0, lam):
    """The lambda-return from its definition (forward view), as an independent reference:

        G_t = (1 - lam) * sum_{n=1}^{N-1} lam^(n-1) * G_t^(n)  +  lam^(N-1) * MC_t,   N = T - t

    with the two-player n-step return from mover t's side
        G_t^(n) = sum_{k=1}^{n} (-1)^k score[t+k]  +  (-1)^n value[t+n].
    """
    length = len(scores)
    final = np.where(movers == 0, final_seat0, -final_seat0).astype(np.float64)
    mc = final - spread
    out = np.zeros(length)
    for t in range(length):
        horizon = length - t
        total = 0.0
        running = 0.0
        for n in range(1, horizon):
            running += (-1) ** n * scores[t + n]
            n_step = running + (-1) ** n * values[t + n]
            total += (1 - lam) * lam ** (n - 1) * n_step
        total += lam ** (horizon - 1) * mc[t]
        out[t] = total
    return out, mc


@pytest.mark.parametrize("seed", range(40))
def test_matches_the_forward_view_definition(seed):
    rng = np.random.default_rng(seed)
    movers, spread, scores, values, final_seat0 = random_game(rng)
    for lam in (0.0, 0.3, 0.9, float(rng.uniform()), 1.0):
        expected, mc = forward_view(movers, spread, scores, values, final_seat0, lam)
        got = returns(movers, f32(spread), f32(scores), f32(values), final_seat0, lam)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-3, err_msg=f"lambda {lam}")
        if lam == 1.0:
            assert got.tolist() == mc.tolist()


@pytest.mark.parametrize("seed", range(40, 60))
def test_lambda_one_recursion_equals_direct_monte_carlo(seed):
    """For alternating games with spread_after[t+1] = -spread_after[t] + score[t+1], running the
    recursion g[t] = -(score[t+1] + g[t+1]) from g[T-1] = MC[T-1] reproduces MC exactly. This
    is what justifies `returns` short-circuiting at lambda = 1."""
    rng = np.random.default_rng(seed)
    movers, spread, scores, values, final_seat0 = random_game(rng)
    final = np.where(movers == 0, final_seat0, -final_seat0)
    direct = final - spread
    g = direct.copy()
    for t in range(len(g) - 2, -1, -1):
        g[t] = -(scores[t + 1] + g[t + 1])
    assert g.tolist() == direct.tolist()
    assert returns(movers, f32(spread), f32(scores), f32(values), final_seat0, 1.0).tolist() == direct.tolist()
