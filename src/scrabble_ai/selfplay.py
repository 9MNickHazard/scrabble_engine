"""Self-play: many games in lockstep, one network forward pass per step over every candidate.

Each step the Rust environment writes, for every game with a decision pending, the afterstates of
the candidate moves. `Scorer` values them all at once, a move is picked per game, and the chosen
afterstate is remembered. When a game ends its remembered afterstates get their targets (the
spread the mover actually went on to gain, and whether the mover won) and go to the replay buffer.

Perspective convention, used everywhere: a value belongs to an afterstate and is from the side of
the player who made the move. The next afterstate in a game belongs to the opponent, so anything
carried backwards from it changes sign (see `returns`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from scrabble_ai import _core
from scrabble_ai.features import FeatureBuilder, to_device
from scrabble_ai.replay import Replay

MAX_TURNS = 128  # afterstates remembered per game; games are about 24 turns long


class CandidateBuffers:
    """The arrays `BatchEnv.prepare` fills, for `slots` games and `k` candidates each."""

    def __init__(self, slots: int, k: int):
        self.slots, self.k = slots, k
        self.board = np.zeros((slots, k, _core.SQUARES), np.uint8)
        self.leave = np.zeros((slots, k, _core.TILE_TYPES), np.uint8)
        self.unseen = np.zeros((slots, k, _core.TILE_TYPES), np.uint8)
        self.meta = np.zeros((slots, k, _core.META), np.int16)
        self.leave_value = np.zeros((slots, k), np.float32)
        self.count = np.zeros(slots, np.int32)
        self.mover = np.zeros(slots, np.int8)
        self.status = np.zeros(slots, np.uint8)
        self.scores = np.zeros((slots, 2), np.int32)

    def prepare(self, env: _core.BatchEnv) -> None:
        env.prepare(self.board, self.leave, self.unseen, self.meta, self.leave_value,
                    self.count, self.mover, self.status, self.scores)


class Scorer:
    """A network as a function from afterstate codes to (value in points, win probability)."""

    # Batches go to the network in a few fixed sizes. A new batch shape makes cuDNN re-tune its
    # convolutions, and the number of candidates changes every step, so the last chunk is padded
    # up to a multiple of PAD rather than sent at its natural size.
    CHUNK = 4096
    PAD = 512

    def __init__(self, net: torch.nn.Module, engine: _core.Engine, value_scale: float, device: torch.device,
                 amp: bool = True):
        self.net = net
        self.engine = engine
        self.value_scale = value_scale
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.cross_checks = net.config.cross_checks
        self.leave_baseline = net.config.leave_baseline
        self.features = FeatureBuilder(device, self.cross_checks, torch.bfloat16 if self.amp else torch.float32)

    @torch.inference_mode()
    def __call__(self, board: np.ndarray, leave: np.ndarray, unseen: np.ndarray, meta: np.ndarray,
                 leave_value: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        self.net.eval()
        cross = None
        if self.cross_checks:
            cross = np.empty((len(board), _core.CROSS_WORDS), np.uint32)
            self.engine.cross_masks(board, cross)
        values, wins = [], []
        for start in range(0, len(board), self.CHUNK):
            rows = np.arange(start, min(start + self.CHUNK, len(board)))
            real = len(rows)
            padding = -real % self.PAD
            if padding:
                rows = np.concatenate([rows, np.full(padding, rows[0])])
            batch = to_device(board[rows], leave[rows], unseen[rows], meta[rows], leave_value[rows],
                              None if cross is None else cross[rows], self.device)
            planes, vector = self.features(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                value, win_logit = self.net(planes, vector)
            value = value.float() * self.value_scale
            if self.leave_baseline:
                value = value + batch.leave_value
            values.append(value[:real])
            wins.append(torch.sigmoid(win_logit.float())[:real])
        return torch.cat(values), torch.cat(wins)


@dataclass
class Choice:
    """What was decided for the slots that had a decision pending."""

    slots: np.ndarray  # slot index of each decision
    actions: np.ndarray  # chosen candidate index
    value: np.ndarray  # V of the chosen afterstate in points (0 if it ended the game by going out)
    win: np.ndarray  # P(mover wins) for the chosen afterstate


def choose(buffers: CandidateBuffers, scorer: Scorer | None, temperature: float, win_weight: float,
           win_scale: float, device: torch.device, generator: torch.Generator,
           only: np.ndarray | None = None) -> Choice:
    """Pick a move for every slot with a pending decision (or only the slots in `only`).

    With a scorer, utility = (1 - w) * (score + V) + w * win_scale * (2 P(win) - 1); without one
    it is the static value, score + leave value. Temperature 0 takes the best; above 0 it samples
    from softmax(utility / temperature), with temperature in points.
    """
    slots = np.flatnonzero(buffers.status == _core.STATUS_DECISION) if only is None else only
    if len(slots) == 0:
        empty = np.zeros(0, np.float32)
        return Choice(slots, np.zeros(0, np.int32), empty, empty)
    k = buffers.k
    counts = buffers.count[slots]
    valid = np.arange(k)[None, :] < counts[:, None]  # [D, k]
    meta = buffers.meta[slots][valid]  # [M, META]
    leave_value = buffers.leave_value[slots][valid]

    score = torch.from_numpy(meta[:, _core.M_SCORE].astype(np.float32)).to(device)
    went_out = torch.from_numpy(meta[:, _core.M_TERMINAL] == 1).to(device)
    if scorer is None:
        value = torch.from_numpy(leave_value).to(device)
        win = torch.full_like(value, 0.5)
    else:
        value, win = scorer(buffers.board[slots][valid], buffers.leave[slots][valid],
                            buffers.unseen[slots][valid], meta, leave_value)
    # Going out ends the game: nothing remains to be gained and the result is known.
    spread = torch.from_numpy(meta[:, _core.M_SPREAD].astype(np.float32)).to(device)
    value = torch.where(went_out, torch.zeros_like(value), value)
    win = torch.where(went_out, (spread > 0).float() + 0.5 * (spread == 0).float(), win)
    utility = (1.0 - win_weight) * (score + value) + win_weight * win_scale * (2.0 * win - 1.0)

    valid_t = torch.from_numpy(valid).to(device)
    padded = torch.full(valid.shape, -torch.inf, device=device)
    padded[valid_t] = utility
    if temperature > 0:
        uniform = torch.rand(padded.shape, device=device, generator=generator).clamp_(1e-12, 1.0 - 1e-7)
        noisy = padded / temperature - torch.log(-torch.log(uniform))  # Gumbel-max sampling
        actions = noisy.argmax(dim=1)
    else:
        actions = padded.argmax(dim=1)
    def of_chosen(per_candidate: torch.Tensor) -> np.ndarray:
        grid = torch.zeros(valid.shape, device=device)
        grid[valid_t] = per_candidate
        return grid.gather(1, actions[:, None]).squeeze(1).cpu().numpy()

    return Choice(slots, actions.cpu().numpy().astype(np.int32), of_chosen(value), of_chosen(win))


def returns(mover: np.ndarray, spread_after: np.ndarray, score: np.ndarray, value: np.ndarray,
            final_spread_seat0: int, td_lambda: float) -> np.ndarray:
    """Value targets for one game's afterstates, in points, each from its own mover's side.

    `mc[t]` is the spread the mover of turn t actually gained after that move:
    final spread (their side) minus their spread right after the move.

    With `td_lambda < 1` the target mixes in the value the network gave the *next* afterstate:

        g[t] = -( score[t+1] + (1 - lambda) * value[t+1] + lambda * g[t+1] )

    The minus sign is the two-player part: turn t+1 belongs to the opponent, so their move's score
    and everything they go on to gain counts against the mover of turn t. With lambda = 1 this
    reduces exactly to `mc`.
    """
    final = np.where(mover == 0, final_spread_seat0, -final_spread_seat0).astype(np.float32)
    mc = final - spread_after
    if td_lambda >= 1.0 or len(mc) < 2:
        return mc
    g = mc.copy()
    for t in range(len(mc) - 2, -1, -1):
        g[t] = -(score[t + 1] + (1.0 - td_lambda) * value[t + 1] + td_lambda * g[t + 1])
    return g


@dataclass
class ActorStats:
    games: int = 0
    positions: int = 0
    decisions: int = 0
    turns: int = 0
    score_sum: float = 0.0
    spread_abs_sum: float = 0.0
    first_player_wins: float = 0.0
    # How well the values used during play predicted what then happened. These games had not been
    # trained on when the predictions were made, so this is an honest, out-of-sample measure.
    value_sq_error: float = 0.0  # network value vs the spread actually gained
    leave_sq_error: float = 0.0  # the same for the static leave value alone
    win_sq_error: float = 0.0  # Brier score of the win probability
    recent: list[tuple[int, int, int]] = field(default_factory=list)  # (seed, score0, score1)

    def merge(self, other: "ActorStats") -> None:
        self.games += other.games
        self.positions += other.positions
        self.decisions += other.decisions
        self.turns += other.turns
        self.score_sum += other.score_sum
        self.spread_abs_sum += other.spread_abs_sum
        self.first_player_wins += other.first_player_wins
        self.value_sq_error += other.value_sq_error
        self.leave_sq_error += other.leave_sq_error
        self.win_sq_error += other.win_sq_error


class Actor:
    """Plays self-play games (both seats decided here) and feeds finished games to the replay."""

    def __init__(self, engine: _core.Engine, leaves: _core.LeaveModel, replay: Replay, slots: int,
                 max_candidates: int, max_exchanges: int, td_lambda: float, device: torch.device,
                 first_seed: int, sampling_seed: int):
        self.env = _core.BatchEnv(engine, leaves, slots, max_candidates, max_exchanges)
        self.buffers = CandidateBuffers(slots, max_candidates)
        self.replay = replay
        self.td_lambda = td_lambda
        self.device = device
        self.next_seed = first_seed
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(sampling_seed)
        self.slots = slots
        self.seeds = np.zeros(slots, np.uint64)
        # Afterstates chosen so far in each slot's current game.
        self.t_board = np.zeros((slots, MAX_TURNS, _core.SQUARES), np.uint8)
        self.t_leave = np.zeros((slots, MAX_TURNS, _core.TILE_TYPES), np.uint8)
        self.t_unseen = np.zeros((slots, MAX_TURNS, _core.TILE_TYPES), np.uint8)
        self.t_meta = np.zeros((slots, MAX_TURNS, _core.META), np.int16)
        self.t_leave_value = np.zeros((slots, MAX_TURNS), np.float32)
        self.t_value = np.zeros((slots, MAX_TURNS), np.float32)
        self.t_win = np.zeros((slots, MAX_TURNS), np.float32)
        self.t_mover = np.zeros((slots, MAX_TURNS), np.int8)
        self.length = np.zeros(slots, np.int32)
        self.truncated = np.zeros(slots, bool)
        for slot in range(slots):
            self._start(slot)

    def _start(self, slot: int) -> None:
        self.env.reset(slot, self.next_seed, "external", "external")
        self.seeds[slot] = self.next_seed
        self.next_seed += 1
        self.length[slot] = 0
        self.truncated[slot] = False

    def _finish(self, slot: int, stats: ActorStats) -> None:
        n = int(self.length[slot])
        score0, score1 = (int(s) for s in self.buffers.scores[slot])
        meta = self.t_meta[slot, :n]
        mover = self.t_mover[slot, :n]
        # A game longer than MAX_TURNS lost its tail, so bootstrapping across the gap is unsafe.
        lam = 1.0 if self.truncated[slot] else self.td_lambda
        target = returns(mover, meta[:, _core.M_SPREAD].astype(np.float32), meta[:, _core.M_SCORE].astype(np.float32),
                         self.t_value[slot, :n], score0 - score1, lam)
        final = np.where(mover == 0, score0 - score1, score1 - score0)
        won = ((final > 0) + 0.5 * (final == 0)).astype(np.float32)
        # Afterstates that ended the game by going out have a known value of 0: nothing to learn.
        keep = meta[:, _core.M_TERMINAL] != 1
        gained = (final - meta[:, _core.M_SPREAD])[keep].astype(np.float32)
        stats.value_sq_error += float(np.square(self.t_value[slot, :n][keep] - gained).sum())
        stats.leave_sq_error += float(np.square(self.t_leave_value[slot, :n][keep] - gained).sum())
        stats.win_sq_error += float(np.square(self.t_win[slot, :n][keep] - won[keep]).sum())
        self.replay.add({
            "board": self.t_board[slot, :n][keep],
            "leave": self.t_leave[slot, :n][keep],
            "unseen": self.t_unseen[slot, :n][keep],
            "meta": meta[keep],
            "leave_value": self.t_leave_value[slot, :n][keep],
            "value_target": target[keep].astype(np.float32),
            "win_target": won[keep],
        })
        stats.games += 1
        stats.positions += int(keep.sum())
        stats.turns += int(self.env.turn(slot))
        stats.score_sum += score0 + score1
        stats.spread_abs_sum += abs(score0 - score1)
        stats.first_player_wins += 1.0 if score0 > score1 else 0.5 if score0 == score1 else 0.0
        if len(stats.recent) < 8:
            stats.recent.append((int(self.seeds[slot]), score0, score1))

    def step(self, scorer: Scorer | None, temperature: float, win_weight: float = 0.0,
             win_scale: float = 100.0) -> ActorStats:
        """One lockstep move in every game. Finished games are flushed and restarted."""
        stats = ActorStats()
        b = self.buffers
        b.prepare(self.env)
        for slot in np.flatnonzero(b.status == _core.STATUS_FINISHED):
            self._finish(int(slot), stats)
            self._start(int(slot))  # joins in on the next step
        choice = choose(b, scorer, temperature, win_weight, win_scale, self.device, self.generator)
        if len(choice.slots) == 0:
            return stats
        slots, actions = choice.slots, choice.actions
        t = self.length[slots]
        room = t < MAX_TURNS
        self.truncated[slots[~room]] = True
        s, a, tt = slots[room], actions[room], t[room]
        self.t_board[s, tt] = b.board[s, a]
        self.t_leave[s, tt] = b.leave[s, a]
        self.t_unseen[s, tt] = b.unseen[s, a]
        self.t_meta[s, tt] = b.meta[s, a]
        self.t_leave_value[s, tt] = b.leave_value[s, a]
        self.t_value[s, tt] = choice.value[room]
        self.t_win[s, tt] = choice.win[room]
        self.t_mover[s, tt] = b.mover[s]
        self.length[s] += 1
        full = np.zeros(self.slots, np.int32)
        full[slots] = actions
        self.env.apply(full)
        stats.decisions = len(slots)
        return stats

    def state(self) -> dict:
        return {"next_seed": int(self.next_seed), "generator": self.generator.get_state()}

    def load_state(self, state: dict) -> None:
        """Resume the seed counter and sampling stream. Games that were in progress when the
        checkpoint was written are not restored: every slot starts a fresh game."""
        self.next_seed = int(state["next_seed"])
        try:
            self.generator.set_state(state["generator"].cpu())
        except RuntimeError:
            # The state came from a generator on another device (GPU vs CPU): start a new
            # stream, still fixed by the seed counter so a resume is repeatable.
            self.generator.manual_seed(self.next_seed)
        for slot in range(self.slots):
            self._start(slot)
