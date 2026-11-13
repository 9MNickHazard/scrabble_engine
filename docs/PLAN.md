# Plan

Last updated 2026-11-12. This is the status document: what is done, how it was checked, and what
comes next. [DESIGN.md](DESIGN.md) explains how the system works and
[TRAINING.md](TRAINING.md) how to run it.

## Where things stand

| Phase | What | Status |
|---|---|---|
| 0 | Move the old project out of OneDrive, audit it, set up the repo, check the GPU | Done |
| 1 | Rust engine: lexicon, move generator, rules, tests, Python bindings | Done |
| 2 | Baseline bots, leave values, paired evaluation | Done |
| 3 | RL pipeline: features, network, self-play, training loop, checkpoints | Code done and smoke-tested. **No real training run yet.** |
| 4 | Desktop app (train, compare, play, watch, matches); `analyze` command, GCG import/export | App built and tested. `analyze` and GCG not started |
| 5 | Simulation, endgame solver, longer runs | Not started |

The next step is the first real training run, which is the owner's to start
(see [TRAINING.md](TRAINING.md)). Nothing about the network's strength is known until then.

## Phase 0: migration and audit (done)

- The old folder (6,973 files, 1.52 GB) was catalogued in a manifest with a SHA-256 for every
  file. The 444 files worth keeping (1.05 GB) were copied to an archive outside the repo and every
  copy was verified against the manifest. Virtual environments, caches and a private key were left
  out of the archive.
- The newest version of each still-relevant old file is in `legacy/` under its real name, with
  its source and hash in `legacy/PROVENANCE.md`.
- [legacy-audit.md](legacy-audit.md) records what was wrong with the old code, each finding marked
  as reproduced by running it or established by reading.
- PyTorch 2.15 (CUDA 13.2 build) computes on the RTX 5070 Ti: `uv run scrabble-ai gpu-check`.

## Phase 1: engine (done)

| Acceptance check | Target | Result |
|---|---|---|
| Hand-checked scoring cases | at least 30 | 50 pass (35 legal, 15 illegal), through the validator and, for the legal ones, the generator |
| Generator vs independent reference | identical on 5,000+ positions, 500+ with one blank, 100+ with two | Identical on 9,160 positions (1,854 one blank, 1,353 two blanks): 28,449,555 plays compared |
| Validator and scorer vs generator | every generated move re-checks to the same play and score | Yes, in the greedy-game and random-game tests |
| Random full games with invariants | 100,000 games | 100,000 games, 5,456,289 turns, 0 violations |
| Old engine's rule bugs | a test for each | In `validate.rs` and `game.rs` tests |
| Nothing the network sees depends on the hidden rack | test | 1,337 positions with the opponent's rack swapped: every input byte unchanged |
| Speed targets | median move generation under 200 microseconds; two-blank rack under 10 ms; 1,000+ greedy games/s | Median 78 µs; two-blank racks 4.4 ms at the median and 22 ms at worst; 2,356 greedy games/s. Details in [BENCHMARKS.md](BENCHMARKS.md) |

Not done from the original list: GCG import/export (moved to Phase 4) and criterion benchmarks
(`scrabble-ai bench` and the timing output of the tests cover the same ground).

## Phase 2: baselines and evaluation (done)

- Bots: `random`, `greedy` (highest score), `greedy-leave` (score + leave value, exchanges
  included).
- Leave values: a linear model fitted by ridge regression on self-play, three rounds of 200,000
  games (`uv run scrabble-ai fit-leaves`).
- `uv run scrabble-ai eval A B`: seat-swapped pairs, mean spread and win rate with 95% intervals.

| Acceptance check | Result |
|---|---|
| random < greedy, 2,000 pairs | greedy +394.7 ± 2.4 points a game |
| greedy < greedy-leave, 2,000 pairs | greedy-leave +38.0 ± 2.7 points a game, 65.7% of games |
| Same seeds give the same results | Yes (tested, independent of thread scheduling) |
| Greedy's mean score next to the old log | 411 points a game for greedy against itself. The old log's two games averaged 281 a side, with a 10-letter word list and unknown weights, so the two are not comparable. |

## Phase 3: RL pipeline (code done, not yet trained)

What exists: the afterstate encoder and batched environment in Rust; feature expansion on the
GPU; the value network with a win head; Monte Carlo and TD(lambda) targets with the two-player
sign; warm start from the static evaluator; lockstep self-play with temperature sampling; a
replay buffer; checkpoints that resume; TensorBoard and JSONL logging; periodic paired evaluation
against the baselines and gating against the best network; `scrabble-ai train`.

What has been checked: the full loop runs end to end (warm start, pretraining, self-play,
evaluation, gating, checkpoint, resume) in two dry runs of a few minutes each, and 231 Python
tests cover the pieces. The tests were written by a separate author from the code; they found
six bugs in the plumbing (replay reload after wrap-around, an interrupted warm start being
skipped on resume, the resume path, three unvalidated settings, cross-device resume), all fixed,
and none in the training targets, the input features or the sign convention, which they check
against hand-worked examples.

**What has not been checked, because it needs a real run:**

| Open question | How it gets answered |
|---|---|
| Does the network beat `greedy-leave`? | The eval lines of the first run. Until the interval excludes zero, it has not. |
| Is training stable at the default learning rate, without normalization? | The loss and the value-error lines of the first hour. If not: `train.learning_rate=3e-4`, then `model.norm="group"`. |
| What does pruning to 64 candidates cost? | Evaluate a trained network with `--candidates 64` against itself with `--candidates 256`. |
| Is TD(lambda) better than Monte Carlo targets here? | A second run with `train.td_lambda=0.9`, compared by evaluation. |
| How much do the cross-check planes buy? | A run with `model.cross_checks=false`, compared by evaluation. |

**Recommended first run:** the defaults, overnight.

```powershell
uv run scrabble-ai train --config configs/train.toml --name night1 --hours 8
```

A five-minute dry run of that command ran at 58 self-play games a second with training in the
loop, which extrapolates to about 1.7 million games in 8 hours. The estimate gets replaced by the
run's own log.

## Phase 4: the app, analysis and play (app done)

**Built: Scrabble Lab**, a local desktop app ([APP.md](APP.md)): start training runs one after
another or side by side with any settings, follow them live, read an automatic report of each
run, compare runs, browse saved networks, play against them with hints and the network's own
reasoning shown, watch any two players on a board, and run fast matches with statistics and
replayable games. It replaces the old tkinter GUIs in `legacy/`.

How it was checked:

| Check | Result |
|---|---|
| Server and analysis tests (`tests/test_app_*.py`, `tests/test_insights.py`) | 372 tests, written by a separate author from the code. They pinned 22 bugs in the server, all fixed; the worst was that the on-screen greedy and random bots differed slightly from the ones in matches. |
| A match game replays in the viewer move for move | Tested for all three bots and a network, both seatings |
| Start two runs from the form, one after another; the second starts by itself; stop, resume, delete | Driven through the real pages in a headless browser against the real server, with two one-minute CPU smoke runs |
| A whole game played through the Play page, and one watched to the end and replayed from its seed | Same way, against a smoke-test network |
| Every page loads without a script error, with the real server and on invented data, dark and light | Yes |

The charts and report pages were checked visually against invented long runs, since no real long
run exists yet. The first real run is their first real test.

Still to do in this phase:

- `scrabble-ai analyze`: given a board, rack and scores, print ranked moves with their score, leave
  value, network value and win probability (the app shows this during a game; this would do it
  for an arbitrary position).
- GCG import and export, so games can be viewed in standard tools.

## Phase 5: stretch

In rough order of expected value:

1. Simulation at inference time: play out the top candidates two moves deep with the value network
   judging the result.
2. Exact endgame solver once the bag is empty.
3. Pre-endgame solver for the last few tiles in the bag.
4. A larger network and a longer run, sized from the first run's results.
5. Proper inference of the opponent's rack from their play.
6. A reference engine (Quackle, Macondo or wolges) as an outside sparring partner, after checking
   licences. Not vendored.

An expectation worth stating before any training: from what is publicly known about Scrabble
engines, a greedy player with good leave values is already strong, and a network that judges
positions in one glance usually adds a modest amount to it. The large known gains come from items
1 and 2 above, which use a value network like this one. So the first run is judged on whether the
network beats `greedy-leave` at all, and the result will be reported whichever way it goes.

## Risks

| Risk | What is done about it |
|---|---|
| The network does not beat greedy-leave | Report it plainly. Check the pruning cost and the value error first; then bring simulation forward. |
| Monte Carlo targets are noisy (a game's remaining spread has a standard deviation of about 74 points) | Many games and large batches; TD(lambda) as the next experiment. |
| Exploration moves pollute the targets | The temperature falls over the run and is logged. |
| A subtle engine bug poisons the training data | The reference comparison and the random-game invariants above. |
| VRAM overflow silently slows training on Windows | PyTorch's share of VRAM is capped, so an oversized setting fails at once. |
| The word list is the school list, not the tournament list | Decide before a long run: a network is tied to the list it trained on. `configs/lexicons.toml` takes any list. |
