"""`selfplay.Scorer` and `selfplay.choose`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scrabble_ai import _core
from scrabble_ai.config import ModelConfig
from scrabble_ai.model import ValueNet
from scrabble_ai.selfplay import CandidateBuffers, Scorer, choose

from conftest import cuda_only

CPU = torch.device("cpu")
TINY = dict(blocks=1, channels=16, vector_hidden=16, head_hidden=16, value_channels=2)


def tiny_net(seed: int = 0, trained: bool = True, **overrides) -> ValueNet:
    """A 1x16 network. `trained=False` keeps the zero-initialized outputs of a fresh network."""
    torch.manual_seed(seed)
    net = ValueNet(ModelConfig(**{**TINY, **overrides}))
    if trained:
        torch.nn.init.normal_(net.value.weight, std=0.5)
        torch.nn.init.normal_(net.win.weight, std=0.5)
        torch.nn.init.normal_(net.value.bias, std=0.5)
        for block in net.blocks:
            if block.gate is not None:
                torch.nn.init.constant_(block.gate, 0.5)
    return net.eval()


def real_buffers(engine, leaves, slots=8, k=16, exchanges=3, moves=6, seed=0) -> CandidateBuffers:
    """Candidates from real positions: each slot plays a random number of random moves first."""
    env = _core.BatchEnv(engine, leaves, slots, k, exchanges)
    buffers = CandidateBuffers(slots, k)
    next_seed = 500 + seed * 1000
    for slot in range(slots):
        env.reset(slot, next_seed + slot, "external", "external")
    next_seed += slots
    rng = np.random.default_rng(seed)
    for step in range(moves + 1):
        buffers.prepare(env)
        finished = np.flatnonzero(buffers.status == _core.STATUS_FINISHED)
        for slot in finished:  # rare: six random scoreless turns in a row
            env.reset(int(slot), next_seed, "external", "external")
            next_seed += 1
        if len(finished):
            buffers.prepare(env)
        if step == moves:
            break
        actions = np.array([rng.integers(0, max(1, c)) for c in buffers.count], np.int32)
        env.apply(actions)
    assert (buffers.status == _core.STATUS_DECISION).all()
    return buffers


def flat(buffers: CandidateBuffers):
    slots = np.flatnonzero(buffers.status == _core.STATUS_DECISION)
    valid = np.arange(buffers.k)[None, :] < buffers.count[slots][:, None]
    take = lambda a: np.ascontiguousarray(a[slots][valid])  # noqa: E731
    return slots, valid, (take(buffers.board), take(buffers.leave), take(buffers.unseen), take(buffers.meta),
                          take(buffers.leave_value))


def synthetic(slots: int, k: int, count, score, leave_value, terminal=None, spread=None) -> CandidateBuffers:
    """Buffers on an empty board with the given per-candidate numbers ([slots, k] arrays)."""
    b = CandidateBuffers(slots, k)
    b.status[:] = _core.STATUS_DECISION
    b.count[:] = count
    b.meta[:, :, _core.M_SCORE] = score
    b.leave_value[:] = leave_value
    if terminal is not None:
        b.meta[:, :, _core.M_TERMINAL] = terminal
    if spread is not None:
        b.meta[:, :, _core.M_SPREAD] = spread
    b.meta[:, :, _core.M_BAG] = 50
    return b


def utilities(buffers, values, wins, win_weight=0.0, win_scale=100.0):
    """The documented utility per valid candidate, computed independently with NumPy in float32
    (the precision `choose` works in, so that ties come out the same)."""
    slots, valid, (_, _, _, meta, leave_value) = flat(buffers)
    f = np.float32
    score = meta[:, _core.M_SCORE].astype(f)
    went_out = meta[:, _core.M_TERMINAL] == 1
    spread = meta[:, _core.M_SPREAD]
    values = np.where(went_out, f(0), np.asarray(values, f))
    wins = np.where(went_out, ((spread > 0) + 0.5 * (spread == 0)).astype(f), np.asarray(wins, f))
    utility = f(1 - win_weight) * (score + values) + f(win_weight * win_scale) * (f(2) * wins - f(1))
    grid = np.full(valid.shape, -np.inf, f)
    grid[valid] = utility
    return slots, grid


# ---- Scorer ----------------------------------------------------------------------------------


def test_padding_and_chunking_do_not_change_results(engine, leaves):
    buffers = real_buffers(engine, leaves)
    _, _, arrays = flat(buffers)
    m = len(arrays[0])
    assert m > 70, m
    scorer = Scorer(tiny_net(), engine, 50.0, CPU)
    values, wins = scorer(*arrays)
    assert values.shape == (m,) and wins.shape == (m,)
    assert values.dtype == torch.float32 and wins.dtype == torch.float32
    assert values.std() > 1.0, "the network output should vary between candidates"

    # The same candidates split across calls of awkward sizes.
    pieces, start = [], 0
    for size in (1, 7, 13, m):
        stop = min(m, start + size)
        if stop > start:
            pieces.append(scorer(*(a[start:stop] for a in arrays)))
        start = stop
    torch.testing.assert_close(torch.cat([p[0] for p in pieces]), values, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(torch.cat([p[1] for p in pieces]), wins, rtol=1e-5, atol=1e-6)

    # Several chunks per call, each padded.
    small = Scorer(scorer.net, engine, 50.0, CPU)
    small.CHUNK, small.PAD = 32, 12
    v2, w2 = small(*arrays)
    torch.testing.assert_close(v2, values, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(w2, wins, rtol=1e-5, atol=1e-6)

    # Reversing the order of the candidates reverses the output.
    v3, _ = scorer(*(np.ascontiguousarray(a[::-1]) for a in arrays))
    torch.testing.assert_close(v3.flip(0), values, rtol=1e-5, atol=1e-4)


def test_value_scale_and_leave_baseline(engine, leaves):
    buffers = real_buffers(engine, leaves, slots=3)
    _, _, arrays = flat(buffers)
    net = tiny_net(seed=3)
    with_baseline, _ = Scorer(net, engine, 50.0, CPU)(*arrays)
    net.config.leave_baseline = False  # Scorer reads the flag at construction
    raw50, _ = Scorer(net, engine, 50.0, CPU)(*arrays)
    raw10, _ = Scorer(net, engine, 10.0, CPU)(*arrays)
    net.config.leave_baseline = True
    torch.testing.assert_close(raw50, raw10 * 5, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(with_baseline, raw50 + torch.from_numpy(arrays[4]), rtol=1e-5, atol=1e-4)


def test_untrained_network_with_leave_baseline_returns_the_leave_value_exactly(engine, leaves):
    buffers = real_buffers(engine, leaves, seed=2)
    _, _, arrays = flat(buffers)
    values, wins = Scorer(tiny_net(trained=False), engine, 50.0, CPU)(*arrays)
    assert torch.equal(values, torch.from_numpy(arrays[4]))
    assert (wins == 0.5).all()
    assert arrays[4].std() > 0, "the fitted leave values differ between candidates"


# ---- choose ----------------------------------------------------------------------------------


def test_temperature_zero_is_the_argmax_and_never_a_padding_index(engine, leaves):
    buffers = real_buffers(engine, leaves, seed=4)
    counts = buffers.count.copy()
    # Shrink some slots' candidate lists and poison everything past the count, so that reading
    # stale rows would be noticed.
    counts[::2] = np.maximum(1, counts[::2] // 3)
    buffers.count[:] = counts
    for slot, count in enumerate(counts):
        buffers.meta[slot, count:, _core.M_SCORE] = 30000
        buffers.leave_value[slot, count:] = 1e6
    scorer = Scorer(tiny_net(seed=5), engine, 50.0, CPU)
    generator = torch.Generator().manual_seed(0)

    for player in (None, scorer):
        choice = choose(buffers, player, 0.0, 0.0, 100.0, CPU, generator)
        _, _, arrays = flat(buffers)
        if player is None:
            values, wins = arrays[4], np.full(len(arrays[4]), 0.5, np.float32)
        else:
            v, w = player(*arrays)
            values, wins = v.numpy(), w.numpy()
        slots, grid = utilities(buffers, values, wins)
        assert choice.slots.tolist() == slots.tolist()
        assert (choice.actions < counts[choice.slots]).all() and (choice.actions >= 0).all()
        assert choice.actions.tolist() == grid.argmax(axis=1).tolist()
        # The reported value and win are those of the chosen candidate (0 / the result if it goes out).
        meta = arrays[3]
        went_out = meta[:, _core.M_TERMINAL] == 1
        spread = meta[:, _core.M_SPREAD]
        values = np.where(went_out, 0.0, values)
        wins = np.where(went_out, (spread > 0) + 0.5 * (spread == 0), wins)
        offsets = np.concatenate([[0], np.cumsum(counts[slots])[:-1]])
        np.testing.assert_allclose(choice.value, values[offsets + choice.actions], rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(choice.win, wins[offsets + choice.actions], rtol=1e-6)

    for temperature in (0.5, 50.0, 1e6):
        choice = choose(buffers, scorer, temperature, 0.0, 100.0, CPU, generator)
        assert (choice.actions < counts).all()


def test_without_a_network_temperature_zero_picks_candidate_zero(engine, leaves):
    buffers = real_buffers(engine, leaves, seed=6)
    choice = choose(buffers, None, 0.0, 0.0, 100.0, CPU, torch.Generator().manual_seed(0))
    assert (choice.actions == 0).all()


def test_untrained_leave_baseline_network_picks_the_static_best(engine, leaves):
    scorer = Scorer(tiny_net(trained=False), engine, 50.0, CPU)
    for seed in range(3):
        buffers = real_buffers(engine, leaves, seed=10 + seed, moves=4 + 5 * seed)
        choice = choose(buffers, scorer, 0.0, 0.0, 100.0, CPU, torch.Generator().manual_seed(0))
        assert (choice.actions == 0).all()


def test_only_restricts_the_slots(engine, leaves):
    buffers = real_buffers(engine, leaves, seed=7)
    only = np.array([1, 4, 6])
    choice = choose(buffers, None, 0.0, 0.0, 100.0, CPU, torch.Generator().manual_seed(0), only=only)
    assert choice.slots.tolist() == [1, 4, 6] and len(choice.actions) == 3


def test_no_pending_decision():
    buffers = CandidateBuffers(3, 4)
    buffers.status[:] = _core.STATUS_FINISHED
    choice = choose(buffers, None, 1.0, 0.0, 100.0, CPU, torch.Generator())
    assert len(choice.slots) == len(choice.actions) == len(choice.value) == 0


@pytest.mark.parametrize("use_network", [False, True])
def test_going_out_has_value_zero_and_a_known_result(engine, use_network):
    # Three slots, one candidate each, all going out: ahead by 5, level, behind by 5.
    buffers = synthetic(3, 4, count=1, score=20, leave_value=3.0, terminal=1,
                        spread=np.array([[5] * 4, [0] * 4, [-5] * 4]))
    scorer = Scorer(tiny_net(seed=8), engine, 50.0, CPU) if use_network else None
    choice = choose(buffers, scorer, 0.0, 0.0, 100.0, CPU, torch.Generator().manual_seed(0))
    assert choice.actions.tolist() == [0, 0, 0]
    assert choice.value.tolist() == [0.0, 0.0, 0.0]
    assert choice.win.tolist() == [1.0, 0.5, 0.0]


def test_going_out_utility_uses_score_only(engine):
    # Candidate 0: going out for 10 (leave value 99 must be ignored). Candidate 1: 12 points, keeps a
    # leave worth -1. Candidate 1 is better: 12 - 1 = 11 > 10 + 0.
    buffers = synthetic(1, 2, count=2, score=np.array([[10, 12]]), leave_value=np.array([[99.0, -1.0]]),
                        terminal=np.array([[1, 0]]), spread=np.array([[10, 12]]))
    choice = choose(buffers, None, 0.0, 0.0, 100.0, CPU, torch.Generator())
    assert choice.actions.tolist() == [1]
    assert choice.value.tolist() == [-1.0]


def test_win_weight_one_picks_by_win_probability(engine, leaves):
    buffers = real_buffers(engine, leaves, seed=9)
    scorer = Scorer(tiny_net(seed=11), engine, 50.0, CPU)
    _, _, arrays = flat(buffers)
    values, wins = scorer(*arrays)
    _, by_win = utilities(buffers, values.numpy(), wins.numpy(), win_weight=1.0)
    top = np.sort(by_win, axis=1)
    assert (top[:, -1] > top[:, -2]).all(), "no ties for the best win probability"
    _, by_spread = utilities(buffers, values.numpy(), wins.numpy(), win_weight=0.0)
    generator = torch.Generator().manual_seed(0)
    choice = choose(buffers, scorer, 0.0, 1.0, 100.0, CPU, generator)
    assert choice.actions.tolist() == by_win.argmax(axis=1).tolist()
    assert by_win.argmax(axis=1).tolist() != by_spread.argmax(axis=1).tolist(), "the test should tell them apart"
    # The result does not depend on win_scale when win_weight is 1.
    assert choose(buffers, scorer, 0.0, 1.0, 7.0, CPU, generator).actions.tolist() == choice.actions.tolist()


def test_sampling_follows_softmax_in_points():
    # 4000 identical slots, utilities 0, 1, 2 points; temperature 1 point.
    slots = 4000
    buffers = synthetic(slots, 4, count=3, score=np.array([0, 1, 2, 500]), leave_value=0.0)
    generator = torch.Generator().manual_seed(1)
    choice = choose(buffers, None, 1.0, 0.0, 100.0, CPU, generator)
    assert (choice.actions < 3).all()
    frequencies = np.bincount(choice.actions, minlength=3) / slots
    expected = np.exp([0.0, 1.0, 2.0]) / np.exp([0.0, 1.0, 2.0]).sum()  # 0.090, 0.245, 0.665
    np.testing.assert_allclose(frequencies, expected, atol=0.03)
    # Temperature 2 points flattens it: softmax([0, 0.5, 1]).
    choice = choose(buffers, None, 2.0, 0.0, 100.0, CPU, generator)
    frequencies = np.bincount(choice.actions, minlength=3) / slots
    expected = np.exp([0.0, 0.5, 1.0]) / np.exp([0.0, 0.5, 1.0]).sum()
    np.testing.assert_allclose(frequencies, expected, atol=0.03)


def test_high_temperature_explores_every_candidate(engine, leaves):
    buffers = real_buffers(engine, leaves, seed=12)
    scorer = Scorer(tiny_net(seed=13), engine, 50.0, CPU)
    generator = torch.Generator().manual_seed(2)
    seen = [set() for _ in range(buffers.slots)]
    for i in range(250):
        # The network is only used for a few draws: what matters here is the sampling.
        choice = choose(buffers, scorer if i < 5 else None, 1e4, 0.0, 100.0, CPU, generator)
        for slot, action in zip(choice.slots, choice.actions):
            seen[slot].add(int(action))
    for slot in range(buffers.slots):
        assert seen[slot] == set(range(buffers.count[slot])), slot
    # The same generator state gives the same draws.
    a = choose(buffers, scorer, 3.0, 0.0, 100.0, CPU, torch.Generator().manual_seed(5)).actions
    b = choose(buffers, scorer, 3.0, 0.0, 100.0, CPU, torch.Generator().manual_seed(5)).actions
    assert a.tolist() == b.tolist()


# ---- GPU --------------------------------------------------------------------------------------


@cuda_only
def test_gpu_bf16_scorer_matches_cpu_and_untrained_baseline_is_exact(engine, leaves):
    gpu = torch.device("cuda")
    buffers = real_buffers(engine, leaves, seed=14, slots=4)
    _, _, arrays = flat(buffers)
    untrained = tiny_net(trained=False).to(gpu).to(memory_format=torch.channels_last)
    scorer = Scorer(untrained, engine, 50.0, gpu, amp=True)
    assert scorer.amp
    values, wins = scorer(*arrays)
    assert torch.equal(values.cpu(), torch.from_numpy(arrays[4]))
    assert (wins == 0.5).all()
    choice = choose(buffers, scorer, 0.0, 0.0, 100.0, gpu, torch.Generator(device=gpu).manual_seed(0))
    assert (choice.actions == 0).all()

    trained = tiny_net(seed=15)
    cpu_values, _ = Scorer(trained, engine, 50.0, CPU)(*arrays)
    gpu_values, _ = Scorer(trained.to(gpu), engine, 50.0, gpu, amp=False)(*arrays)
    # Loose: cuDNN may use TF32 for the convolutions.
    torch.testing.assert_close(gpu_values.cpu(), cpu_values, rtol=2e-2, atol=0.5)
    choice = choose(buffers, Scorer(trained, engine, 50.0, gpu, amp=True), 2.0, 0.0, 100.0, gpu,
                    torch.Generator(device=gpu).manual_seed(0))
    assert (choice.actions < buffers.count).all()
