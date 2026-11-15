"""Command line entry point: `uv run scrabble-ai <command>`.

    gpu-check    prove PyTorch is computing on the GPU
    lexicon      load a configured word list and print its ID and size
    fit-leaves   fit leave values from self-play (needed once per lexicon, before training)
    eval         paired match between two players (bots or saved networks)
    train        train the value network by self-play
    games        play games between static evaluators and save them, to train from
    train-games  train a network from a set of saved games
    bench        measure the speed of the engine, self-play and training on this machine
    app          open Scrabble Lab, the desktop app (train, watch, play, compare)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _gpu_check(_args: argparse.Namespace) -> int:
    from scrabble_ai.gpu_check import run

    return run()


def _lexicon(args: argparse.Namespace) -> int:
    from scrabble_ai.lexicon import load_lexicon

    lexicon = load_lexicon(args.name)
    lengths: dict[int, int] = {}
    for word in lexicon.words:
        lengths[len(word)] = lengths.get(len(word), 0) + 1
    print(f"lexicon:  {lexicon.name}")
    print(f"id:       {lexicon.id}")
    print(f"file:     {lexicon.path}")
    print(f"words:    {len(lexicon.words):,}")
    print("by length: " + "  ".join(f"{n}:{lengths[n]:,}" for n in sorted(lengths)))
    return 0


def _fit_leaves(args: argparse.Namespace) -> int:
    import numpy as np

    from scrabble_ai import _core
    from scrabble_ai import engine as engine_module
    from scrabble_ai.evaluate import eval_seeds, play_match

    engine = engine_module.load_engine(args.lexicon)
    out = Path(args.out) if args.out else engine_module.default_leaves_path(engine.lexicon_id)
    print(f"lexicon {engine.lexicon_id}: fitting leave values over {args.rounds} rounds of {args.games:,} games")
    leaves = _core.LeaveModel.zero()
    for round_index in range(args.rounds):
        # Round 0 learns from plain greedy play; later rounds from play that already uses the
        # previous round's values, which is closer to how the values will be used.
        bot = "greedy" if round_index == 0 else "greedy-leave"
        seeds = np.arange(args.games, dtype=np.uint64) + np.uint64(3_000_000_000_000 + round_index * 1_000_000_000)
        start = time.time()
        leaves, report = engine.fit_leaves(bot, leaves, seeds, args.ridge)
        seconds = time.time() - start
        print(f"round {round_index + 1}: {args.games:,} {bot} games in {seconds:.0f} s ({args.games / seconds:.0f} games/s)")
        for stage, (samples, mean, sd) in zip(("bag 7+", "bag 1-6", "bag empty"), report):
            print(f"  {stage:<10} {samples:>10,} samples, remaining spread mean {mean:+.1f}, sd {sd:.1f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    leaves.save(out)
    print(f"saved {out} (hash {engine_module.leaves_hash(leaves)})")

    nothing = leaves.value("", 60)
    print("value of keeping these tiles with a full bag, relative to keeping nothing (points):")
    for rack in ("?", "S", "X", "Z", "ERS", "AEINRT", "Q", "QU", "V", "UU", "VVW", "AEIOU"):
        print(f"  {rack:<7} {leaves.value(rack, 60) - nothing:+6.1f}")
    if args.pairs > 0:
        result = play_match(engine, leaves, "greedy-leave", "greedy", eval_seeds(args.pairs))
        print("check: " + result.line())
    return 0


def _eval(args: argparse.Namespace) -> int:
    import torch

    from scrabble_ai import engine as engine_module
    from scrabble_ai.evaluate import BOTS, NetPlayer, eval_seeds, play_match
    from scrabble_ai.selfplay import Scorer
    from scrabble_ai.train import load_network

    loaded = engine_module.load(args.lexicon, args.leaves)
    device = torch.device(args.device)

    def player(spec: str):
        if spec in BOTS:
            return spec
        path = Path(spec)
        if not path.exists():
            raise SystemExit(f"{spec!r} is neither a bot ({', '.join(BOTS)}) nor a checkpoint file")
        net, state = load_network(path, device)
        if state["lexicon_id"] != loaded.lexicon_id:
            raise SystemExit(f"{spec}: trained with lexicon {state['lexicon_id']}, evaluating with {loaded.lexicon_id}")
        if state["leaves_hash"] != loaded.leaves_hash:
            raise SystemExit(f"{spec}: trained with different leave values than the ones loaded")
        scale = state["config"]["train"]["value_scale"]
        own = state["config"].get("selfplay", {})
        win_weight = own.get("win_weight", 0.0) if args.win_weight is None else args.win_weight
        win_scale = own.get("win_scale", 100.0) if args.win_scale is None else args.win_scale
        return NetPlayer(f"{path.parent.parent.name}@{state['step']}", Scorer(net, loaded.engine, scale, device),
                         win_weight, win_scale)

    a, b = player(args.a), player(args.b)
    start = time.time()
    result = play_match(loaded.engine, loaded.leaves, a, b, eval_seeds(args.pairs, args.seed_offset), device,
                        args.candidates, args.exchanges)
    print(result.line())
    verdict = "significant" if result.significant else "not significant"
    print(f"95% interval on spread: [{result.spread - result.spread_ci:+.1f}, {result.spread + result.spread_ci:+.1f}] "
          f"({verdict}); {2 * args.pairs} games in {time.time() - start:.0f} s")
    return 0


def _train(args: argparse.Namespace) -> int:
    from scrabble_ai.config import ConfigError, apply_dict, apply_overrides, load_config
    from scrabble_ai.lexicon import REPO_ROOT
    from scrabble_ai.train import Trainer, load_checkpoint_config, resolve_resume

    overrides = list(args.set or [])
    if args.name:
        overrides.append(f"run.name={args.name}")
    if args.hours is not None:
        overrides.append(f"run.hours={args.hours}")
    try:
        resume = None
        if args.resume:
            # Resuming starts from the checkpoint's own settings; --config and --set adjust them.
            # Which run to look in comes from the same places a fresh run takes its name from.
            where = load_config(Path(args.config) if args.config else None, overrides).run
            out_dir = Path(where.out_dir)
            run_dir = out_dir / where.name if out_dir.is_absolute() else REPO_ROOT / out_dir / where.name
            resume = resolve_resume(args.resume, run_dir)
            config = load_checkpoint_config(resume)
            if args.config:
                import tomllib

                with open(args.config, "rb") as fh:
                    apply_dict(config, tomllib.load(fh))
            apply_overrides(config, overrides)
            config.validate()
        else:
            config = load_config(Path(args.config) if args.config else None, overrides)
    except (ConfigError, FileNotFoundError) as error:
        raise SystemExit(f"error: {error}") from None
    try:
        trainer = Trainer(config, resume)
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from None
    trainer.run()
    return 0


def _bench(args: argparse.Namespace) -> int:
    from scrabble_ai.bench import run

    return run(args.lexicon, args.leaves, args.models.split(","), args.norm, args.slots, args.candidates,
               args.batch, args.quick, not args.no_gpu, args.inputs)


def _games(args: argparse.Namespace) -> int:
    from scrabble_ai import engine as engine_module
    from scrabble_ai import games as games_module

    try:
        loaded = engine_module.load(args.lexicon, args.leaves)
        directory = games_module.set_dir(args.name, Path(args.dir) if args.dir else None)
        meta = games_module.generate(loaded, directory, args.games, opening_plies=args.opening_plies,
                                     opening_temperature=args.opening_temperature)
    except (games_module.GamesError, FileNotFoundError) as error:
        raise SystemExit(f"error: {error}") from None
    print(f"{meta['name']}: {meta['games']:,} games, {meta['bytes'] / 2**20:,.0f} MB, in {directory}")
    return 0


def _train_games(args: argparse.Namespace) -> int:
    from scrabble_ai import games as games_module
    from scrabble_ai.config import OFFLINE_DEFAULTS, Config, ConfigError, apply_dict, apply_overrides, load_config
    from scrabble_ai.lexicon import REPO_ROOT
    from scrabble_ai.offline import GameTrainer
    from scrabble_ai.train import load_checkpoint_config, resolve_resume

    overrides = list(args.set or [])
    if args.name:
        overrides.append(f"run.name={args.name}")
    if args.hours is not None:
        overrides.append(f"run.hours={args.hours}")
    if args.games:
        overrides.append(f"offline.games={args.games}")

    def games_defaults() -> list[str]:
        """What this way of training changes in the other sections (`OFFLINE_DEFAULTS`), as
        overrides that come before the user's own. A setting the config file gives a value other
        than the built-in one is the user's choice and is left alone; `--set` outranks both."""
        built_in = Config().to_dict()
        in_file = load_config(Path(args.config)).to_dict() if args.config else built_in
        chosen = {item.split("=", 1)[0].strip() for item in overrides}
        applied = {}
        for key, value in OFFLINE_DEFAULTS.items():
            section, name = key.split(".")
            if in_file[section][name] == built_in[section][name]:
                applied[key] = str(value).lower() if isinstance(value, bool) else value
        shown = ", ".join(f"{key}={value}" for key, value in applied.items() if key not in chosen)
        if shown:
            print(f"settings training from games starts from: {shown} (--set changes any of them)", flush=True)
        return [f"{key}={value}" for key, value in applied.items()]

    try:
        resume = None
        if args.resume:
            where = load_config(Path(args.config) if args.config else None, overrides).run
            out_dir = Path(where.out_dir)
            run_dir = out_dir / where.name if out_dir.is_absolute() else REPO_ROOT / out_dir / where.name
            resume = resolve_resume(args.resume, run_dir)
            config = load_checkpoint_config(resume)
            if args.config:
                import tomllib

                with open(args.config, "rb") as fh:
                    apply_dict(config, tomllib.load(fh))
            apply_overrides(config, overrides)
            config.validate()
        else:
            config = load_config(Path(args.config) if args.config else None, games_defaults() + overrides)
        trainer = GameTrainer(config, resume)
    except (ConfigError, games_module.GamesError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from None
    trainer.run()
    return 0


def _app(args: argparse.Namespace) -> int:
    from scrabble_ai.app.__main__ import main as app_main

    arguments = []
    if args.install_shortcut:
        arguments.append("--install-shortcut")
    if args.no_window:
        arguments.append("--no-window")
    if args.port:
        arguments += ["--port", str(args.port)]
    return app_main(arguments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scrabble-ai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    gpu = commands.add_parser("gpu-check", help="verify PyTorch runs real work on the GPU")
    gpu.set_defaults(func=_gpu_check)

    lex = commands.add_parser("lexicon", help="load a configured word list and print its ID and size")
    lex.add_argument("name", nargs="?", default=None, help="lexicon name from configs/lexicons.toml")
    lex.set_defaults(func=_lexicon)

    fit = commands.add_parser("fit-leaves", help="fit leave values from self-play")
    fit.add_argument("--lexicon", default="", help="lexicon name (default: the configured default)")
    fit.add_argument("--games", type=int, default=200_000, help="self-play games per round")
    fit.add_argument("--rounds", type=int, default=2, help="rounds of fitting")
    fit.add_argument("--ridge", type=float, default=50.0, help="ridge penalty, in samples")
    fit.add_argument("--pairs", type=int, default=2000, help="game pairs for the closing check against greedy (0 = skip)")
    fit.add_argument("--out", default="", help="output file (default: data/leaves/<lexicon id>.leaves)")
    fit.set_defaults(func=_fit_leaves)

    ev = commands.add_parser("eval", help="paired match between two players")
    ev.add_argument("a", help="random, greedy, greedy-leave, or a checkpoint file")
    ev.add_argument("b", help="the opponent, same choices")
    ev.add_argument("--pairs", type=int, default=2000, help="seeds; each is played twice with seats swapped")
    ev.add_argument("--seed-offset", type=int, default=0, help="shift the evaluation seeds to get a fresh sample")
    ev.add_argument("--lexicon", default="")
    ev.add_argument("--leaves", default="", help='leave-value file, or "zero"')
    ev.add_argument("--candidates", type=int, default=64, help="moves per decision a network evaluates")
    ev.add_argument("--exchanges", type=int, default=4)
    ev.add_argument("--win-weight", type=float, default=None,
                    help="0 = networks choose by spread, 1 = by win probability (default: as each network was trained)")
    ev.add_argument("--win-scale", type=float, default=None)
    ev.add_argument("--device", default="cuda")
    ev.set_defaults(func=_eval)

    tr = commands.add_parser("train", help="train the value network by self-play")
    tr.add_argument("--config", default="", help="TOML file of settings (see configs/train.toml)")
    tr.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override one setting; repeatable")
    tr.add_argument("--name", default="", help="run name: outputs go to runs/<name>/ (same as --set run.name=...)")
    tr.add_argument("--hours", type=float, default=None, help="stop after this many hours (same as --set run.hours=...)")
    tr.add_argument("--resume", default="", help='"latest" to continue runs/<name>/, or a checkpoint file')
    tr.set_defaults(func=_train)

    ga = commands.add_parser("games", help="play and save games between static evaluators, to train from")
    ga.add_argument("--name", required=True, help="name of the game set (a folder under data/games)")
    ga.add_argument("--games", type=int, required=True, help="how many games the set should hold; an existing set is topped up")
    ga.add_argument("--opening-plies", type=int, default=None, help="vary up to this many first plies of each game (default 4)")
    ga.add_argument("--opening-temperature", type=float, default=None, help="how freely, in points (default 6)")
    ga.add_argument("--dir", default="", help="folder that holds game sets (default: data/games)")
    ga.add_argument("--lexicon", default="")
    ga.add_argument("--leaves", default="")
    ga.set_defaults(func=_games)

    tg = commands.add_parser("train-games", help="train a network from a set of saved games")
    tg.add_argument("--games", default="", help="the game set to learn from (sets offline.games)")
    tg.add_argument("--config", default="", help="TOML file of settings. The few settings this way of training starts "
                    "from other values for (printed at the start) follow the file only where it changes them from the "
                    "built-in value; --set always counts")
    tg.add_argument("--name", default="", help="run name: outputs go to runs/<name>/")
    tg.add_argument("--hours", type=float, default=None, help="stop after this many hours (0 = run all the passes)")
    tg.add_argument("--resume", default="", help="'latest' or a checkpoint file of this run")
    tg.add_argument("--set", action="append", metavar="section.key=value", help="change one setting; may be repeated")
    tg.set_defaults(func=_train_games)

    be = commands.add_parser("bench", help="measure engine, self-play and training speed")
    be.add_argument("--models", default="6x64", help="network sizes to time, e.g. 6x64,10x128")
    be.add_argument("--norm", default="none", choices=["none", "group"])
    be.add_argument("--slots", type=int, default=256, help="self-play games in lockstep")
    be.add_argument("--candidates", type=int, default=64)
    be.add_argument("--batch", type=int, default=2048, help="training batch size")
    be.add_argument("--lexicon", default="")
    be.add_argument("--leaves", default="")
    be.add_argument("--quick", action="store_true", help="smaller samples: a fast sanity check, noisier numbers")
    be.add_argument("--no-gpu", action="store_true", help="engine sections only")
    be.add_argument("--inputs", default="all", choices=["all", "original"],
                    help="time networks with every input group, or with only those of before 2026-11-14")
    be.set_defaults(func=_bench)

    ap = commands.add_parser("app", help="open Scrabble Lab, the desktop app")
    ap.add_argument("--install-shortcut", action="store_true", help="put a shortcut on the Desktop and exit")
    ap.add_argument("--no-window", action="store_true", help="serve only and print the address")
    ap.add_argument("--port", type=int, default=0)
    ap.set_defaults(func=_app)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
