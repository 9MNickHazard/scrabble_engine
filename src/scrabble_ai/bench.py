"""`scrabble-ai bench`: measure the speed of each part on this machine.

Sections: move generation (one thread), bot games per second (all cores), what the network's
inputs cost the engine (all cores, no GPU), and, for each network size asked for, self-play and
training throughput on the GPU. Networks are untrained: only the speed is meaningful. Run it on
an otherwise idle machine.
"""

from __future__ import annotations

import time

import numpy as np

from scrabble_ai import _core
from scrabble_ai import engine as engine_module


def _engine_section(loaded: engine_module.Loaded, quick: bool) -> None:
    engine = loaded.engine
    print(f"lexicon {loaded.lexicon_id}: {engine.word_count:,} words, GADDAG {engine.gaddag_bytes / 1e6:.1f} MB")

    games = 10 if quick else 60
    print(f"\nmove generation, one thread, every turn of {games} greedy games:")
    print(f"  {'rack':<12}{'positions':>10}{'median':>12}{'99th pct':>12}{'slowest':>12}{'mean':>12}{'plays found':>14}")
    timings = engine.bench_movegen(np.arange(games, dtype=np.uint64) + np.uint64(42))
    for label, (positions, median, p99, slowest, mean, plays) in zip(("no blank", "one blank", "two blanks"), timings):
        print(f"  {label:<12}{positions:>10,}{median:>10.0f} us{p99:>10.0f} us{slowest:>10.0f} us{mean:>10.0f} us{plays:>14,.0f}")

    print("\nbot self-play, all cores:")
    zero = _core.LeaveModel.zero()
    for bot, leaves, pairs in (("greedy", zero, 1000 if quick else 10_000), ("greedy-leave", loaded.leaves, 500 if quick else 5_000)):
        seeds = np.arange(pairs, dtype=np.uint64) + np.uint64(8_000_000_000_000)
        start = time.perf_counter()
        result = engine.play_pairs(bot, leaves, bot, leaves, seeds)
        seconds = time.perf_counter() - start
        mean_score = (result["score_a"].mean() + result["score_b"].mean()) / 2
        print(f"  {bot:<13} {2 * pairs:>6,} games in {seconds:5.1f} s = {2 * pairs / seconds:>6,.0f} games/s "
              f"(mean score {mean_score:.0f}, {result['turns'].mean():.1f} turns/game)")


def _inputs_section(loaded: engine_module.Loaded, slots: int, candidates: int, quick: bool) -> None:
    """What the engine spends, on the CPU, preparing decisions for a network: candidates and
    their afterstates, the cross masks and openness worked out from the boards, and the inference
    about the opponent's rack at different numbers of sampled racks. The moves are the static
    evaluator's first choice, so no network or GPU is involved."""
    from scrabble_ai.selfplay import CandidateBuffers

    engine, leaves = loaded.engine, loaded.leaves
    steps = 12 if quick else 40
    print(f"\nnetwork inputs on the CPU, all cores, {slots} games x {candidates} candidates, {steps} steps each:")
    print(f"  {'racks sampled per move':<24}{'prepare':>10}{'apply':>10}{'cross masks':>13}{'openness':>10}{'decisions/s':>13}")
    for samples in (0, 2, 6, 12):
        env = _core.BatchEnv(engine, leaves, slots, candidates, 4, samples)
        buffers = CandidateBuffers(slots, candidates)
        for slot in range(slots):
            env.reset(slot, 2000 + slot, "external", "external")
        seed, decisions = 300_000, 0
        spent = {"prepare": 0.0, "apply": 0.0, "cross": 0.0, "open": 0.0}
        for step in range(steps + 3):
            timed = step >= 3
            t0 = time.perf_counter()
            buffers.prepare(env)
            t1 = time.perf_counter()
            for slot in np.flatnonzero(buffers.status == _core.STATUS_FINISHED):
                env.reset(int(slot), seed, "external", "external")
                seed += 1
            deciding = buffers.status == _core.STATUS_DECISION
            valid = np.arange(candidates)[None, :] < buffers.count[:, None]
            boards = buffers.board[valid & deciding[:, None]]
            cross = np.empty((len(boards), _core.CROSS_WORDS), np.uint32)
            opened = np.empty((len(boards), _core.SQUARES), np.uint8)
            t2 = time.perf_counter()
            engine.cross_masks(boards, cross)
            t3 = time.perf_counter()
            engine.open_squares(boards, cross, opened)
            t4 = time.perf_counter()
            env.apply(np.zeros(slots, np.int32))
            t5 = time.perf_counter()
            if timed:
                decisions += int(deciding.sum())
                for key, seconds in (("prepare", t1 - t0), ("cross", t3 - t2), ("open", t4 - t3), ("apply", t5 - t4)):
                    spent[key] += seconds
        per_step = {key: 1e3 * seconds / steps for key, seconds in spent.items()}
        print(f"  {samples:<24}{per_step['prepare']:>7.1f} ms{per_step['apply']:>7.1f} ms{per_step['cross']:>10.1f} ms"
              f"{per_step['open']:>7.1f} ms{decisions / sum(spent.values()):>13,.0f}")
    print("  (decisions/s is the most the engine could feed a network that took no time at all)")


def _gpu_section(loaded: engine_module.Loaded, spec: str, norm: str, slots: int, candidates: int, batch: int, quick: bool,
                 inputs: str) -> None:
    import torch

    from scrabble_ai.config import BEFORE_IT_EXISTED, ModelConfig
    from scrabble_ai.features import FeatureBuilder, FeatureSet, engine_inputs, to_device
    from scrabble_ai.model import ValueNet, parameter_count
    from scrabble_ai.replay import Replay
    from scrabble_ai.selfplay import Actor, CandidateBuffers, Scorer, choose

    blocks, channels = (int(part) for part in spec.split("x"))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    config = ModelConfig(blocks=blocks, channels=channels, norm=norm, **(BEFORE_IT_EXISTED["model"] if inputs == "original" else {}))
    net = ValueNet(config).to(device).to(memory_format=torch.channels_last)
    engine, leaves = loaded.engine, loaded.leaves
    print(f"\nnetwork {spec} (norm={norm}, {inputs} inputs: {net.features.planes} planes, {net.features.vector} vector), "
          f"{parameter_count(net):,} parameters:")

    # Self-play: lockstep games, every candidate scored by the network.
    scorer = Scorer(net.eval(), engine, 50.0, device)
    env = _core.BatchEnv(engine, leaves, slots, candidates, 4, config.belief_samples)
    buffers = CandidateBuffers(slots, candidates)
    for slot in range(slots):
        env.reset(slot, 1000 + slot, "external", "external")
    generator = torch.Generator(device=device)
    warm, total = (15, 45) if quick else (30, 150)
    seed, decisions, scored = 100_000, 0, 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(total):
        if step == warm:
            torch.cuda.synchronize()
            start = time.perf_counter()
        buffers.prepare(env)
        for slot in np.flatnonzero(buffers.status == _core.STATUS_FINISHED):
            env.reset(int(slot), seed, "external", "external")
            seed += 1
        choice = choose(buffers, scorer, 1.0, 0.0, 100.0, device, generator)
        actions = np.zeros(slots, np.int32)
        actions[choice.slots] = choice.actions
        env.apply(actions)
        if step >= warm:
            decisions += len(choice.slots)
            scored += int(buffers.count[choice.slots].sum())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    # Games per second is derived from decisions: the games here all start together, so counting
    # finishes over a short window would be lumpy.
    print(f"  self-play, {slots} games x {candidates} candidates: {decisions / seconds:,.0f} decisions/s "
          f"(about {decisions / seconds / 24:.0f} games/s at 24 turns a game), {scored / seconds:,.0f} positions scored/s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    # Training: replay sampling, cross-check masks, feature expansion, forward, backward, step.
    replay = Replay(200_000)
    actor = Actor(engine, leaves, replay, 512, 8, 2, 1.0, device, first_seed=1, sampling_seed=2,
                  belief_samples=config.belief_samples)
    while len(replay) < (20_000 if quick else 60_000):
        actor.step(None, 6.0)
    inputs_of_net = FeatureSet.of(config)
    features = FeatureBuilder(device, inputs_of_net, torch.bfloat16)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    net.train()

    def train_step() -> None:
        rows = replay.sample(batch, rng)
        cross, opened = engine_inputs(engine, inputs_of_net, rows["board"])
        data = to_device(rows["board"], rows["leave"], rows["unseen"], rows["meta"], rows["leave_value"], cross, device, opened)
        planes, vector = features(data)
        target = torch.from_numpy(rows["value_target"]).to(device)
        win_target = torch.from_numpy(rows["win_target"]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value, win_logit = net(planes, vector)
        loss = (torch.nn.functional.huber_loss(value.float(), target / 50.0)
                + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(win_logit.float(), win_target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(8):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    steps = 15 if quick else 60
    start = time.perf_counter()
    for _ in range(steps):
        train_step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(f"  training, batch {batch}: {seconds / steps * 1e3:5.0f} ms/step, {batch * steps / seconds:,.0f} positions/s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    del net, scorer, optimizer
    torch.cuda.empty_cache()


def run(lexicon: str, leaves: str, models: list[str], norm: str, slots: int, candidates: int, batch: int,
        quick: bool, gpu: bool, inputs: str = "all") -> int:
    """`inputs`: "all" times networks with every input group (the default for new runs),
    "original" those of before 2026-11-14, for comparing the two."""
    loaded = engine_module.load(lexicon, leaves, allow_missing_leaves=True)
    if loaded.leaves_path is None:
        print("note: no fitted leave values found; using all-zero leave values")
    _engine_section(loaded, quick)
    _inputs_section(loaded, slots, candidates, quick)
    if gpu:
        import torch

        if not torch.cuda.is_available():
            print("\nno GPU visible to PyTorch: skipping the network sections")
            return 0
        print(f"\nGPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, bf16 autocast")
        torch.cuda.set_per_process_memory_fraction(0.9)
        for spec in models:
            _gpu_section(loaded, spec, norm, slots, candidates, batch, quick, inputs)
    return 0
