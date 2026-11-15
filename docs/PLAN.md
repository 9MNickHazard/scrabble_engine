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

## Phase 3: RL pipeline (code done; first runs did not learn)

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

### The first two runs (13 November 2026): both got worse as they trained

Two runs with the original defaults (replay ratio 4, Monte Carlo targets, learning rate 0.001,
batch 2,048, 100,000 warm-start games and 3,000 pretraining steps), started from the app. Spread
is points a game against `greedy-leave` over 250 seat-swapped pairs.

| | 6 x 64 (1,055,056 parameters) | 10 x 128 (3,655,316 parameters) |
|---|---|---|
| Self-play games, time | 450,005 in 2.19 h | 441,805 in 6.67 h |
| Speed (median report) | 61.7 games/s | 18.8 games/s |
| First evaluation | -3.2 ± 7.3 at 75,428 games | -7.7 ± 6.9 at 23,228 games |
| Later | -47.4 ± 7.7 at 151,015 games | -33.7 at 70,000, -60.4 at 93,000, about -80 from 138,000 on |
| Last evaluation | -44.7 ± 8.2 | -82.1 ± 8.3 |
| Promotions after the first network | 0 | 0 |
| Error on new games against leave values alone, end of run (RMSE) | 2.2 points worse | 5.0 points worse |
| Error on training batches, start to end (mean absolute) | 53.6 to 50.4 | 53.6 to 47.0 |

Neither run ever produced a network that beat its first evaluated one, so each run's `best.pt`
is a network from the first minutes of self-play. Played against each other in the app (the two
`best` networks, 500 pairs), the 10 x 128 one scored +4.4 ± 5.6 points a game, 51.7% of games,
and -4.1 ± 5.1 against `greedy-leave` (500 pairs): two nearly untrained networks, level with each other and with the
baseline. **That comparison says nothing about network size.**

**Why: the networks memorised their games.** Every position was trained on about four times,
each time against the final result of its own game. The positions of one game share that result
and nearly the same board, so a network with a million weights can learn "this board went on to
win by 60" in place of anything that carries over to a new game. A check of each run's final
network on the CPU (not in the repo: it read the runs' saved replay buffers), on 12,000 positions
from each end of the buffer and about 11,000 from fresh games:

| Positions | 6 x 64: correlation of its correction with the truth | 6 x 64: RMSE against leave values alone | 10 x 128: correlation | 10 x 128: RMSE against leave values alone |
|---|---|---|---|---|
| Oldest in its replay buffer (seen most often) | 0.340 | 4.18 points better | 0.522 | 10.82 points better |
| Newest in its replay buffer | 0.159 | 0.43 points better | 0.255 | 1.43 points better |
| Fresh games it never saw | 0.022 | 0.94 points worse | 0.031 | 2.71 points worse |

The corrections had a standard deviation of 13 to 21 points (6 x 64) and 22 to 36 points
(10 x 128) and, on new positions, almost no relation to what followed. Choosing a move means
taking the maximum over 64 candidates of score + leave value + correction, and a maximum over
noise picks out the noise:

| On 480 positions from greedy-leave games | 6 x 64 first network | 6 x 64 final | 10 x 128 first network | 10 x 128 final |
|---|---|---|---|---|
| Chooses the same move as the static evaluator | not recorded | 48% | 65% | 32% |
| Static value given up per decision | 1.50 points | 3.60 points | 0.84 points | 7.60 points |

Over the roughly 12 decisions a player makes in a game, 3.60 and 7.60 points a decision are about
the 45 and 80 points a game the two runs lost by. The larger network fitted its own games more
closely and played worse for it.

**What changed in response** (in `configs/train.toml`, the `[train]` section, and
`warmstart.steps`):

| Setting | Was | Now | Why |
|---|---|---|---|
| `train.replay_ratio` | 4 | 1 | Train on each position about once, so there is nothing to memorise |
| `train.td_lambda` | 1.0 | 0.7 | Targets lean on the estimate of the next position, not on one game's final result |
| `train.learning_rate` | 0.001 | 0.0003 | Chase less of the noise that remains |
| `train.batch_size` | 2,048 | 1,024 | Keep a useful number of updates per game at the lower replay ratio |
| `train.actor_sync_steps` | 250 | 100 | Fewer training steps happen per game now; keep the self-play network as fresh |
| `warmstart.steps` | 3,000 | 2,000 | Under one pass over the warm-start positions |
| `train.guard`, `train.guard_patience` | none | on, 2 | When a network is significantly behind the best one at two evaluations in a row, go back to the best network's weights and halve the learning rate |

The trainer now also logs its error on the batches it trains on as RMSE, so the app can draw it
beside the error on new games; the gap between the two is the memorising. The app's run page
states when a run has got worse, and [TUNING.md](TUNING.md) says what to try.

**These changes follow from the diagnosis and are covered by tests, but no full run has used
them yet.** Whether they are enough is the next thing to find out.

### What other people have measured (research, 14 November 2026)

One documented project has trained a value network that beats a static score + leave player:
Cesar Del Solar's networks for Macondo, the engine behind the Woogles bots
([June 2025](https://www.cesardelsolar.com/posts/2025-06-21-nn-scrabble/),
[October 2026](https://www.cesardelsolar.com/posts/2026-10-18-transformer-scrabble/),
[experiment log](https://raw.githubusercontent.com/domino14/macondo/master/pytorch/experiments.md)).
On the same class of GPU, with inputs close to the original set here (85 board planes), his
best network without search wins 57.65% +/- 0.18 of games against the static bot over 100,000
game pairs, level with a 2-ply simulation. His CNNs of 64 to 128 channels all stopped near
52.5%: "the network isn't the bottleneck". What his recipe does that the first runs here did not:

| | The first runs here | What worked there |
|---|---|---|
| Independent games | about 450,000 | 20 million and more (5.4 million was "starved": 50.98%) |
| Positions used per game | all, about four times each | one fresh random position per game per pass |
| Main target | final spread | win, draw or loss of the game; spread as a helper output |
| Training games | self-play, exploring all game | two static bots, only the opening randomised |
| Head | flattened per-square layer | global pooling |
| Extra outputs | none | the opponent's next score and bingo, and where the next moves land (about +4 points of win rate, measured on a transformer only) |

Most of the Scrabble-specific evidence is one author's; his headline result was reproduced
independently (John O'Laughlin). The changes of 13 November above stop the damage but do not
address the amount of data or the target, and a run of a few hours with them is still about a
tenth of what he found too little. Rebuilding training around static-bot games, a win/draw/loss
target and evaluation over tens of thousands of pairs was the recommended next step.

**That is now built (15 November 2026) and has not yet been run on the GPU:** `scrabble-ai games`
and `scrabble-ai train-games`, and the "Train from games" tab of the app
([DESIGN.md](DESIGN.md), "The second way to train"). It follows the recipe in the right-hand
column: one fresh position per game per pass, a win/draw/loss target with points as a helper,
a pooled head, and the helper outputs with their pull balanced. Three kinds of network can be
trained with it (`model.kind`): the convolutional one, the same with KataGo's board-wide pooling,
and a transformer. Also added on request, for the self-play trainer: deep Q-learning as a
learning rule (`train.target = "q"`).

On 14 November seven groups of inputs were added on request ([DESIGN.md](DESIGN.md), "What the
network is given"): the move's score and other scalars, the opponent's last move, board openness,
hook squares and an inference about the opponent's rack. Each has a switch. None has been used
in a run, and the research gives no reason to expect much from inputs: his networks have none of
the last three.

**What is still open:**

| Open question | How it gets answered |
|---|---|
| Does the network beat `greedy-leave` with the new defaults? | A 6 x 64 run of two to three hours. The edge (leave-only error minus the network's, on new games) should go above zero within the first 100,000 games, and the evaluation should follow. |
| If it only stays level? | In order: `train.td_lambda=0.5`; `train.learning_rate=1e-4`; a smaller head (`model.value_channels=2`, `model.head_hidden=128`: the first layer of the head has 460,800 weights that each read one channel of one square, out of the 6 x 64 network's 1,055,056, and that is where a particular board can be memorised); then targets from short look-ahead simulations in place of game results. |
| Is a larger network better? | Only worth asking once the 6 x 64 one has beaten the baseline. Compare with a match of 1,000 pairs or more. |
| What does pruning to 64 candidates cost? | Evaluate a trained network with `--candidates 64` against itself with `--candidates 256`. |
| How much do the cross-check planes buy? | A run with `model.cross_checks=false`, compared by evaluation. |

**Recommended next runs**, in this order (the first is CPU only and can run while the GPU is busy):

```powershell
uv run scrabble-ai games --name twenty-million --games 20000000     # about six hours, 5.7 GB
uv run scrabble-ai train-games --games twenty-million --name wdl-cnn-1
```

What to look for: the win rate against the static evaluator in the half-hourly matches, which
in the published results reached about 51 to 53% for convolutional networks trained this way and
57% with the helper outputs on a transformer. If the convolutional network stops near 52 to 53%,
the transformer (`--set model.kind=transformer --set model.blocks=8 --set model.channels=192`)
is the next thing to try on the same games.

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

The look was then redone as liquid glass (frosted panes, black outlines, light by default with a
dark theme in Settings) and checked page by page in a headless browser. The see-through window
that shows the desktop behind the glass has **not been seen on a screen**: it was built while only
headless testing was allowed. Its prerequisites load, its set-up is tested against fakes, and the
frosted look was previewed over a blurred picture of a desktop. If the real window disappoints,
Settings has a built-in backdrop, and `--window edge` opens the app in an Edge window.

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
| Game results are noisy targets (a game's remaining spread has a standard deviation of about 74 points), and the first runs memorised them | TD(lambda) targets at 0.7, each position trained on about once, and a guard that returns to the best network. Not yet confirmed by a run. |
| Exploration moves pollute the targets | The temperature falls over the run and is logged. |
| A subtle engine bug poisons the training data | The reference comparison and the random-game invariants above. |
| VRAM overflow silently slows training on Windows | PyTorch's share of VRAM is capped, so an oversized setting fails at once. |
| The word list is the school list, not the tournament list | Decide before a long run: a network is tied to the list it trained on. `configs/lexicons.toml` takes any list. |
