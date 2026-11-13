# scrabble_engine

A Scrabble AI trained on one desktop PC: a fast Rust engine with a verified move generator, and a
PyTorch value network that learns by self-play to judge the position each candidate move leaves
behind.

**Status (November 2026).** The engine, the baseline bots, the training pipeline and a desktop
app to run it all are written and tested. No network has been trained yet beyond smoke tests of
a minute or two, so there is no strength result for the network to report. What is measured so
far is in [Results](#results-so-far).

This is a rewrite. The first version of this project (Keras, 2025) is in [`legacy/`](legacy/) for
reference; it did not learn, for reasons set out in [docs/legacy-audit.md](docs/legacy-audit.md).

## How it works

A move in Scrabble is certain except for the tiles drawn afterwards. So the network learns one
thing: **V(x), the spread the mover goes on to gain from the position x a move leaves behind.** A
move's value is its score plus V of the position after it, and every legal move is evaluated in
one batched pass on the GPU.

- **Engine** (`crates/scrabble-core`, Rust): the lexicon compiled to a GADDAG, Gordon's move
  generation algorithm with cross-check sets, NASPA two-player rules, scoring, a seeded bag for
  reproducible games, baseline bots, and a batched environment that plays hundreds of games in
  lockstep.
- **Network** (`src/scrabble_ai`, PyTorch): a residual CNN over the 15 x 15 board plus the rack
  leave, the unseen tiles and a few scalars. Among its inputs are the cross-check sets (which
  letters fit on each empty square), which is how it sees the dictionary.
- **Training**: self-play in one process, a replay buffer in RAM, paired evaluation against the
  baselines, and resumable checkpoints.
- **Scrabble Lab** (`src/scrabble_ai/app`): a local desktop app to start and follow training
  runs, compare them, browse the saved networks, play against them, watch them play, and run
  fast matches with statistics. See [docs/APP.md](docs/APP.md).

[docs/DESIGN.md](docs/DESIGN.md) has the details and the reasoning.

## Results so far

All measured on a Ryzen 7 7800X3D and an RTX 5070 Ti with the NASPA School Word List 2023
(195,747 words). Commands are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Correctness of the engine**

| Check | Result |
|---|---|
| Move generator against an independent brute-force reference | 9,160 positions, 28,449,555 plays, 0 differences |
| Hand-worked scoring cases | 50 of 50 (35 legal, 15 illegal) |
| Random full games with the rules' invariants checked every turn | 100,000 games, 5.5 million turns, 0 violations |

**Baseline ladder** (2,000 seat-swapped pairs each; spread is points per game, with a 95% interval)

| Match | Spread | Wins |
|---|---|---|
| greedy vs random | +394.7 ± 2.4 | 100.0% |
| greedy + leave values vs greedy | +38.0 ± 2.7 | 65.7% |

**Speed**

| What | Measured |
|---|---|
| Move generation, one thread | median 78 µs per position (849 µs with one blank, 4.4 ms with two) |
| Greedy bot playing itself, 16 threads | 2,356 games per second |
| Self-play with the default network (6 x 64) | 103,000 positions scored per second, about 58 games per second with training running |
| Training, default network | 46,000 positions per second |

For comparison, the 2025 move generator took a median of 1.4 seconds per position and missed
8.6% of the legal placements.

## Setup

Needs Windows or Linux, [uv](https://docs.astral.sh/uv/), a Rust toolchain, and an NVIDIA GPU for
training. `pyproject.toml` pins a CUDA 13.2 build of PyTorch 2.15 (chosen for an RTX 50-series
card); change the index there for other hardware.

```powershell
git clone https://github.com/9MNickHazard/scrabble_engine
cd scrabble_engine
uv sync                        # Python 3.12 environment; builds the Rust extension
uv run scrabble-ai gpu-check   # confirms PyTorch is computing on the GPU
```

**Word list.** Word lists are copyrighted and are not in this repo. Put one in `data/lexicon/`
and register it in [configs/lexicons.toml](configs/lexicons.toml): one word per line, anything
after the first word on a line is ignored. Then:

```powershell
uv run scrabble-ai lexicon     # loads it and prints its ID and size
uv run scrabble-ai fit-leaves  # fits leave values for it (about 15 minutes); needed before training
```

## Use

```powershell
uv run scrabble-ai app                      # Scrabble Lab: everything below, in a window
uv run scrabble-ai app --install-shortcut   # put its shortcut on the Desktop
uv run scrabble-ai train --config configs/train.toml --name night1 --hours 8
uv run scrabble-ai train --name night1 --resume latest --hours 8
uv run scrabble-ai eval runs/night1/checkpoints/best.pt greedy-leave --pairs 2000
uv run scrabble-ai eval greedy-leave greedy --pairs 2000
uv run scrabble-ai bench
```

[docs/TRAINING.md](docs/TRAINING.md) covers starting, watching, stopping and resuming a run from
the command line, [docs/APP.md](docs/APP.md) does the same for the app, and every setting is
explained in [configs/train.toml](configs/train.toml).

The app is a web page served on 127.0.0.1 by the Python standard library and shown in its own
window; it needs nothing beyond what `uv sync` installs, and nothing leaves the machine.

## Tests

```powershell
uv run pytest                                  # Python
cargo test -p scrabble-core                    # engine, quick
cargo test -p scrabble-reference               # the independent reference
cargo test -p scrabble-core --test reference -- --ignored --nocapture     # generator vs reference, 9,000+ positions
cargo test -p scrabble-core --test properties -- --ignored --nocapture    # 100,000 random games
```

Tests that need the real word list skip themselves when it is not installed.

## Layout

```
crates/scrabble-core/        Rust engine
crates/scrabble-reference/   independent brute-force rules and move finder, used only as a test oracle
crates/scrabble-py/          PyO3 bindings (the scrabble_ai._core module)
src/scrabble_ai/             features, network, self-play, training, evaluation, CLI
src/scrabble_ai/app/         Scrabble Lab: local server, run manager, analysis, and the pages (static/)
configs/                     train.toml (every training setting), lexicons.toml
tests/                       Python tests
docs/                        DESIGN, TRAINING, APP, APP_API, RULES, BENCHMARKS, PLAN, legacy-audit
legacy/                      the 2025 version, for reference only
tools/                       migration scripts and the scripts behind the legacy audit
data/, runs/                 word lists, leave values, checkpoints, logs (not in the repo)
```

## Documents

- [docs/DESIGN.md](docs/DESIGN.md): what is trained, what the network is given, and why
- [docs/TRAINING.md](docs/TRAINING.md): running a training run and changing its settings
- [docs/APP.md](docs/APP.md): the desktop app, page by page
- [docs/APP_API.md](docs/APP_API.md): the contract between the app's pages and its server
- [docs/RULES.md](docs/RULES.md): the rules the engine implements
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): every measured number and the command behind it
- [docs/PLAN.md](docs/PLAN.md): what is done and what comes next
- [docs/legacy-audit.md](docs/legacy-audit.md): what was wrong with the first version
