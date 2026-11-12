"""Paired evaluation.

Every seed is played twice with the two players swapping seats, so both get the same tiles in the
same order from each side of the board. The unit of analysis is the pair: player A's mean spread
over its two games. Seat advantage and most of the tile luck cancel inside a pair, which makes the
confidence interval much tighter than treating the games as independent.

A player is a built-in bot ("random", "greedy", "greedy-leave") or a network (`NetPlayer`).
Evaluation always plays the best move: no sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from scrabble_ai import _core
from scrabble_ai.selfplay import CandidateBuffers, Scorer, choose

BOTS = ("random", "greedy", "greedy-leave")


@dataclass
class NetPlayer:
    name: str
    scorer: Scorer
    win_weight: float = 0.0
    win_scale: float = 100.0


Player = str | NetPlayer


@dataclass
class MatchResult:
    a: str
    b: str
    pairs: int
    spread: float  # A's mean spread per game
    spread_ci: float  # half-width of the 95% confidence interval, from the per-pair means
    win_rate: float  # A's share of wins, ties counting half
    win_ci: float
    score_a: float
    score_b: float
    turns: float

    @property
    def significant(self) -> bool:
        """True when the interval on A's spread excludes zero."""
        return abs(self.spread) > self.spread_ci

    def line(self) -> str:
        return (f"{self.a} vs {self.b}: spread {self.spread:+.1f} +/- {self.spread_ci:.1f}, "
                f"wins {100 * self.win_rate:.1f}% +/- {100 * self.win_ci:.1f}, "
                f"scores {self.score_a:.0f}-{self.score_b:.0f}, {self.turns:.1f} turns/game, {self.pairs} pairs")


def summarize(a: str, b: str, spread_first: np.ndarray, spread_second: np.ndarray, score_a: np.ndarray,
              score_b: np.ndarray, turns: np.ndarray) -> MatchResult:
    """Statistics over pairs. `spread_first[i]` / `spread_second[i]` are A's spread in the game
    of pair i where A moved first / second."""
    n = len(spread_first)
    per_pair = (spread_first + spread_second) / 2.0

    def half_width(values: np.ndarray) -> float:
        return 1.96 * float(values.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")

    def win(spread: np.ndarray) -> np.ndarray:
        return (spread > 0) + 0.5 * (spread == 0)

    win_per_pair = (win(spread_first) + win(spread_second)) / 2.0
    return MatchResult(a, b, n, float(per_pair.mean()), half_width(per_pair), float(win_per_pair.mean()),
                       half_width(win_per_pair), float(score_a.mean()), float(score_b.mean()), float(turns.mean()))


def _name(player: Player) -> str:
    return player if isinstance(player, str) else player.name


def play_match(engine: _core.Engine, leaves: _core.LeaveModel, a: Player, b: Player, seeds: np.ndarray,
               device: torch.device | None = None, max_candidates: int = 64, max_exchanges: int = 4,
               slots: int = 512) -> MatchResult:
    """Play every seed twice (A first, then B first) and summarize from A's side."""
    seeds = np.asarray(seeds, np.uint64)
    if isinstance(a, str) and isinstance(b, str):
        r = engine.play_pairs(a, leaves, b, leaves, seeds)
        return summarize(a, b, r["spread_first"].astype(np.float64), r["spread_second"].astype(np.float64),
                         r["score_a"].astype(np.float64), r["score_b"].astype(np.float64), r["turns"].astype(np.float64))
    if device is None:
        raise ValueError("a device is needed to evaluate a network")

    games = 2 * len(seeds)  # game 2i: A moves first; game 2i+1: B moves first
    slots = min(slots, games)
    env = _core.BatchEnv(engine, leaves, slots, max_candidates, max_exchanges)
    buffers = CandidateBuffers(slots, max_candidates)
    generator = torch.Generator(device=device)
    final = np.zeros((games, 2), np.int32)
    turns = np.zeros(games, np.int32)
    game_of_slot = np.full(slots, -1, np.int64)
    controller = lambda player: player if isinstance(player, str) else "external"  # noqa: E731
    next_game = 0

    def start(slot: int) -> None:
        nonlocal next_game
        if next_game >= games:
            env.clear(slot)
            game_of_slot[slot] = -1
            return
        first, second = (a, b) if next_game % 2 == 0 else (b, a)
        env.reset(slot, int(seeds[next_game // 2]), controller(first), controller(second))
        game_of_slot[slot] = next_game
        next_game += 1

    for slot in range(slots):
        start(slot)
    finished = 0
    while finished < games:
        buffers.prepare(env)
        for slot in np.flatnonzero(buffers.status == _core.STATUS_FINISHED):
            game = game_of_slot[slot]
            final[game] = buffers.scores[slot]
            turns[game] = env.turn(int(slot))
            finished += 1
            start(int(slot))
        deciding = np.flatnonzero(buffers.status == _core.STATUS_DECISION)
        if len(deciding) == 0:
            continue
        # Seat to move decides which player it is: in even games seat 0 is A.
        a_to_move = (buffers.mover[deciding] == 0) == (game_of_slot[deciding] % 2 == 0)
        actions = np.zeros(slots, np.int32)
        for player, mask in ((a, a_to_move), (b, ~a_to_move)):
            if isinstance(player, str) or not mask.any():
                continue
            choice = choose(buffers, player.scorer, 0.0, player.win_weight, player.win_scale, device, generator,
                            only=deciding[mask])
            actions[choice.slots] = choice.actions
        env.apply(actions)

    a_first, b_first = final[0::2], final[1::2]  # A is seat 0 in a_first, seat 1 in b_first
    return summarize(
        _name(a), _name(b),
        (a_first[:, 0] - a_first[:, 1]).astype(np.float64),
        (b_first[:, 1] - b_first[:, 0]).astype(np.float64),
        np.concatenate([a_first[:, 0], b_first[:, 1]]).astype(np.float64),
        np.concatenate([a_first[:, 1], b_first[:, 0]]).astype(np.float64),
        turns.astype(np.float64),
    )


def eval_seeds(pairs: int, offset: int = 0) -> np.ndarray:
    """Seeds reserved for evaluation, far from the range self-play uses."""
    return np.arange(pairs, dtype=np.uint64) + np.uint64(7_000_000_000_000 + offset)
