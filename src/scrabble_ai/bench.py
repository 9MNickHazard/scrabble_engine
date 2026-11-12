"""`scrabble-ai bench`: measure the speed of each part on this machine.

Sections: move generation (one thread), bot games per second (all cores), and, for each network
size asked for, self-play and training throughput on the GPU. Networks are untrained: only the
speed is meaningful. Run it on an otherwise idle machine.
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


def _gpu_section(loaded: engine_module.Loaded, spec: str, norm: str, slots: int, candidates: int, batch: int, quick: bool) -> None:
    import torch

    from scrabble_ai.config import ModelConfig
    from scrabble_ai.features import FeatureBuilder, to_device
    from scrabble_ai.model import ValueNet, parameter_count
    from scrabble_ai.replay import Replay
    from scrabble_ai.selfplay import Actor, CandidateBuffers, Scorer, choose

    blocks, channels = (int(part) for part in spec.split("x"))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    config = ModelConfig(blocks=blocks, channels=channels, norm=norm)
    net = ValueNet(config).to(device).to(memory_format=torch.channels_last)
    engine, leaves = loaded.engine, loaded.leaves
    print(f"\nnetwork {spec} (norm={norm}), {parameter_count(net):,} parameters:")

    # Self-play: lockstep games, every candidate scored by the network.
    scorer = Scorer(net.eval(), engine, 50.0, device)
    env = _core.BatchEnv(engine, leaves, slots, candidates, 4)
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
    actor = Actor(engine, leaves, replay, 512, 8, 2, 1.0, device, first_seed=1, sampling_seed=2)
    while len(replay) < (20_000 if quick else 60_000):
        actor.step(None, 6.0)
    features = FeatureBuilder(device, True, torch.bfloat16)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    net.train()

    def train_step() -> None:
        rows = replay.sample(batch, rng)
        cross = np.empty((batch, _core.CROSS_WORDS), np.uint32)
        engine.cross_masks(rows["board"], cross)
        data = to_device(rows["board"], rows["leave"], rows["unseen"], rows["meta"], rows["leave_value"], cross, device)
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
        quick: bool, gpu: bool) -> int:
    loaded = engine_module.load(lexicon, leaves, allow_missing_leaves=True)
    if loaded.leaves_path is None:
        print("note: no fitted leave values found; using all-zero leave values")
    _engine_section(loaded, quick)
    if gpu:
        import torch

        if not torch.cuda.is_available():
            print("\nno GPU visible to PyTorch: skipping the network sections")
            return 0
        print(f"\nGPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, bf16 autocast")
        torch.cuda.set_per_process_memory_fraction(0.9)
        for spec in models:
            _gpu_section(loaded, spec, norm, slots, candidates, batch, quick)
    return 0
