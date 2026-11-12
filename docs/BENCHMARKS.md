# Benchmarks and measurements

Every number here was measured on 2026-11-12 on one machine, and each has the command that
produced it.

**Machine:** AMD Ryzen 7 7800X3D (8 cores, 16 threads), 64 GB RAM, NVIDIA RTX 5070 Ti (16 GB),
Windows 11, Rust 1.92, Python 3.12, PyTorch 2.15.0+cu132.
**Lexicon:** NASPA School Word List 2023, 195,747 words (ID `NSWL2023-91a67230e1f6`).

Other programs were running on the machine during some of these, so treat the speeds as slightly
conservative. Correctness results do not depend on that.

## Engine correctness

| Check | Command | Result |
|---|---|---|
| Generator vs independent reference | `cargo test -p scrabble-core --test reference -- --ignored --nocapture` | 9,160 positions (5,953 no blank, 1,854 one blank, 1,353 two blanks), 28,449,555 plays compared, 0 differences, 73 s |
| Hand-worked scoring cases | `cargo test -p scrabble-core --test reference hand_checked` | 35 legal and 15 illegal cases agree; all 35 legal ones are also found by the generator with the same score |
| Random games with invariants | `cargo test -p scrabble-core --test properties -- --ignored --nocapture` | 100,000 games, 5,456,289 turns (3,484,666 placements, 1,286,153 exchanges, 685,470 passes); 43,955 ended by going out, 56,045 by six scoreless turns; 66,321 moves re-checked by the validator; 0 violations, 61 s |
| Hidden information | `cargo test -p scrabble-core --test properties hidden -- --nocapture` | 1,337 positions with the opponent's rack swapped for other unseen tiles: every network input identical; 16,878 endgame afterstates show the opponent's rack as the unseen pool, as they should |
| Greedy games, every move re-validated | `cargo test -p scrabble-core --test real_lexicon -- --nocapture` | 300 games, every generated play accepted by the independent validator with the same score, no duplicates |
| The reference's own tests | `cargo test -p scrabble-reference` | 26 tests pass, including agreement with an exhaustive enumerator on small racks |
| Python suite (targets, features, replay, self-play, trainer, CLI) | `uv run pytest` | 231 tests pass in about 16 s. The training targets and the input layout are checked against hand arithmetic. |

The reference (`crates/scrabble-reference`) was written by a different author from
`docs/RULES.md` alone, and finds moves by sliding every word in the lexicon along every row and
column, a different method from the engine's GADDAG traversal.

## Engine speed

`uv run scrabble-ai bench --no-gpu`

**Lexicon:** the GADDAG is 7.1 MB, builds in 0.9 s the first time and is decoded from its cache file
in about 7 ms (`cargo test -p scrabble-core --test real_lexicon gaddag_build -- --nocapture`).

**Move generation**, one thread, on every turn of 60 greedy games (1,475 positions per row):

| Rack | Median | 99th percentile | Slowest | Mean plays found |
|---|---|---|---|---|
| No blank | 78 µs | 316 µs | 400 µs | 332 |
| One blank | 849 µs | 3.5 ms | 6.1 ms | 3,385 |
| Two blanks | 4.4 ms | 17.4 ms | 22.1 ms | 14,322 |

Each time includes computing the cross-check sets for the position and scoring every play.
The plan's targets were a median under 200 µs (met) and a two-blank rack under 10 ms (met at the
median; the slowest two-blank positions take about 22 ms).

**Whole games**, all 16 threads:

| Bot playing itself | Games per second | Mean score | Turns per game |
|---|---|---|---|
| greedy | 2,356 | 409 | 25.0 |
| greedy-leave | 1,315 | 434 | 22.9 |

### Against the old move generator

The 2025 generator was measured on its own word list (144,375 words of at most 10 letters) with
all 16 threads; details and caveats are in [legacy-audit.md](legacy-audit.md).

| Rack | Old generator, median per position | New generator, median per position |
|---|---|---|
| No blank | 1.43 s (16 threads) | 78 µs (1 thread) |
| One blank | 35.3 s | 849 µs |
| Two blanks | about 1,059 s (3 positions) | 4.4 ms |

The positions differ and the new generator is working with a larger word list, so this is an
order-of-magnitude comparison, not a controlled one. The old generator also missed 8.6% of the
legal placements; the new one matches the reference exactly.

## Baseline ladder

`uv run scrabble-ai eval <A> <B> --pairs 2000` (each seed played twice with seats swapped; spread
is A's mean points per game; intervals are 95%).

| A | B | Spread | A's wins | Mean scores |
|---|---|---|---|---|
| greedy | random | +394.7 ± 2.4 | 100.0% | 539 - 144 |
| greedy-leave | greedy | +38.0 ± 2.7 | 65.7% ± 1.4 | 440 - 402 |
| greedy-leave | random | +444.4 ± 2.3 | 100.0% | 582 - 137 |
| greedy | greedy | 0.0 ± 0.0 | 50.0% | 411 - 411 |

The last row is a check on the pairing: a player against itself must come out exactly even.

Leave values came from `uv run scrabble-ai fit-leaves` (3 rounds of 200,000 games, 7.5 minutes).
Some of the fitted values, in points relative to keeping nothing, with a full bag: blank +33.0,
S +10.7, Z +5.7, X +4.8, ERS +18.1, Q -5.2, QU +2.1, V -4.8, UU -16.6, AEIOU -18.1, VVW -24.7.

## Network speed

`uv run scrabble-ai bench --models 6x64,10x128`

Untrained networks, bf16 autocast, no normalization. Self-play is 256 games in lockstep with up
to 64 candidates per decision, and the figure includes move generation, encoding, cross-check
masks, feature expansion and the forward pass. Training is batch 2,048 and includes sampling,
cross-check masks, feature expansion, forward, backward and the optimizer step.

| Network | Parameters | Self-play: positions scored/s | Self-play: decisions/s | Training: positions/s | Peak VRAM (training) |
|---|---|---|---|---|---|
| 6 x 64 | 1.06 M | 103,356 | 1,645 | 46,030 | 1.40 GB |
| 10 x 128 | 3.66 M | 31,902 | 508 | 12,427 | 4.05 GB |

At about 24 turns a game, 1,645 decisions a second is about 68 games a second.

With GroupNorm instead (`--norm group`), measured earlier the same day with the same code path:
the 6 x 64 network scored 54,371 positions/s in self-play and trained at 21,621 positions/s using
3.84 GB. That is why the default is no normalization.

## The whole training loop

A five-minute dry run of the real command with a shortened warm start (not a training result:
five minutes teaches the network nothing measurable):

```powershell
uv run scrabble-ai train --config configs/train.toml --name dryrun --hours 0.085 --set warmstart.games=20000 --set warmstart.steps=600 --set train.eval_pairs=150 --set train.gate_pairs=150
```

| Stage | Measured |
|---|---|
| Warm start (static evaluator, 512 games in lockstep) | 569 games/s; 20,008 games and 473,244 positions in 35 s |
| Pretraining | 600 steps of 2,048 in 27 s |
| Self-play with training in the loop (replay ratio 4) | 58 games/s over four minutes, about 2.7 training steps a second |
| Evaluation against two baselines, 150 pairs each | 5 s |
| Checkpoint with a 900,000-position replay buffer | under 1 s; 271 MB of replay, about 20 MB per checkpoint |
| Resume from that checkpoint | restored step, games and replay, and carried on at 54 games/s |

At 58 games a second, an 8-hour run is about 1.7 million games. That is an extrapolation from
four minutes.

For the record, the end-of-run evaluation of that five-minute network was +4.2 ± 7.8 points a
game against greedy-leave (150 pairs), which is indistinguishable from zero, as expected: with the
leave baseline an untrained network plays like greedy-leave.
