# Design

What is being trained, what it is given, and why each piece is the way it is. For how to run it
see [TRAINING.md](TRAINING.md); for the rules see [RULES.md](RULES.md).

## The idea in one paragraph

In Scrabble a move's result is certain except for the tiles drawn afterwards. So instead of
learning a value for every (position, move) pair, the network learns one function: **V(x), the
spread the mover goes on to gain from the position x that a move leaves behind**. The value of a
move is then `Q = score of the move + V(position after the move)`, and every legal move can be
scored in one batched forward pass. This is still Q-learning, with the move's own score, which is
known exactly, taken out of what has to be learned.

## Layout

```
crates/scrabble-core/        Rust engine, no Python dependency
  gaddag.rs                  lexicon compiled to a minimal GADDAG (7 MB for 195,747 words)
  movegen.rs                 Gordon's move generator with cross-check sets
  game.rs                    state and turn rules; validate.rs is an independent checker and scorer
  leave.rs, policy.rs        leave values, static evaluation, the baseline bots
  selfplay.rs                parallel bot-vs-bot games and the leave-value fit
  encode.rs, env.rs          afterstate encoding and the batched environment
crates/scrabble-reference/   independent brute-force rules and move finder, used only in tests
crates/scrabble-py/          PyO3 bindings: the scrabble_ai._core module
src/scrabble_ai/
  features.py                compact codes -> network inputs, on the GPU
  model.py                   the value network
  selfplay.py                lockstep self-play, targets, the Scorer
  replay.py                  replay buffer
  train.py                   the training loop, checkpoints, evaluation and gating
  evaluate.py                paired matches and their statistics
  config.py                  every training setting
```

## What the network is given

Everything below describes the position **after** a candidate move, from the side of the player
who made it, before any tiles are drawn. All of it is information that player has.

**Board planes (87 planes of 15 x 15):**

| Planes | Content |
|---|---|
| 26 | Which letter is on each square |
| 1 | The tile is a blank (worth 0 forever) |
| 1 | The tile was placed by this move |
| 1 | The square is empty |
| 4 | Empty double-letter, triple-letter, double-word and triple-word squares (a covered premium is spent) |
| 27 | For a play along the row: which of the 26 letters the tiles above and below allow on each empty square, plus a flag that such tiles exist |
| 27 | The same for a play down the column |

The last 54 are the **cross-check planes**, and they matter more than their place in the table
suggests. They are the only form in which the network sees the dictionary. A convolution can see
that a square next to a word is empty, but whether an S, or anything at all, can legally go there
depends on 195,747 words the network would otherwise have to memorize from game results. With
these planes a hook square for an S, a square nothing fits on, and a wide-open lane all look
different in the input.

**Vector (72 numbers):**

| Entries | Content |
|---|---|
| 27 | The leave: tiles kept on the rack, by type |
| 27 | The unseen pool: tiles the mover cannot see, by type (bag plus opponent's rack, plus anything the mover just threw back in an exchange) |
| 13 | Tiles left in the bag after the mover draws; tiles the mover will draw; leave size; spread; scoreless turns in a row; opponent's tile count; flags for an empty bag and a bag too small to exchange; whether the move was a placement, exchange or pass; size of the unseen pool; the static leave value |
| 5 | The opponent's previous turn: whether it was a placement, an exchange or a pass, how many tiles it used, and what it scored |

**What is deliberately withheld:** the opponent's rack, while tiles remain in the bag. Once the bag
is empty the unseen pool *is* the opponent's rack, which any player can work out, so the network
sees it then. A test swaps the opponent's hidden rack for other tiles and requires every input
byte to be unchanged (`tests/properties.rs`).

The opponent's previous turn is there because it is public and it hints at what they hold:
someone who just exchanged five tiles probably kept two good ones, and someone who just played a
bingo has a fresh random rack.

**What it does not have yet:**

- **Anything further back than one turn**, and nothing about *which* tiles the opponent played.
  A strong human reads more into the opponent's play than a three-number summary of their last
  turn. Proper rack inference is a project of its own.
- **Any look-ahead.** The network judges a position in one glance. It does not try the opponent's
  replies. Simulation and endgame search (Phase 5 of the plan) are where engines get most of their
  remaining strength, and they use a value network like this one at their leaves.

## The network

A residual CNN over the board: a 3 x 3 stem, then 6 blocks of 64 channels by default, no pooling
(positions matter) and no BatchNorm. The vector goes through a small MLP and enters twice: added
to the stem as a per-channel bias, so the trunk can relate the rack to the board from the first
layer ("I hold an S and there is an S hook"), and again in the head. The head sees a per-square
1 x 1 projection (so it knows where things are), global mean and max, and the vector features.

Two outputs: the **value** (spread still to be gained, in units of 50 points) and a **win
logit**. About 1.05 million parameters at the default size.

Two choices keep an untrained network sane:

- **Leave baseline.** The value output is added to the static leave value. A network that outputs
  zero therefore plays exactly like the greedy+leave bot, and training only has to learn what the
  board adds. The output layers start at zero, so this is how every run begins.
- **No normalization by default.** Each residual block is scaled by a learned gate that starts at
  zero. Measured on this GPU, the GroupNorm variant is half as fast and uses 2.7 times the
  training memory. `model.norm = "group"` switches it back.

## Targets and the two-player sign

A value always belongs to an afterstate and is always from the side of the player who made the
move. The next afterstate in the game belongs to the opponent.

- **Monte Carlo target** (default): for each afterstate, the mover's final spread minus their
  spread right after the move. Games are short, and this is unbiased.
- **TD(lambda)** (`train.td_lambda < 1`): mixes in the value given to the next afterstate:
  `g[t] = -(score[t+1] + (1 - lambda) * V[t+1] + lambda * g[t+1])`. The minus sign is the whole
  point: the opponent's score and everything they go on to gain count against the mover. The old
  project added these instead of subtracting them. The values used come from the self-play copy of
  the network, which lags the learner, so it plays the role of a target network.
- **Win target:** 1, 0.5 or 0 for the mover, trained with cross-entropy.
- **Going out** is exact: the game is over, nothing remains to be gained, so its value is 0 by
  definition and it is not a training example. The bonus is part of the move's score.
- The reward is the spread and nothing else. Gamma is 1. There are no shaping terms.

Loss: Huber on the value plus 0.5 x cross-entropy on the win logit.

## Self-play

256 games run in lockstep in one process. Each step the Rust environment generates the legal
moves for every game in parallel, ranks them by static value (score + leave value), and writes the
afterstates of the best 64 (the best placements, up to 4 exchanges, and pass when relevant) into
NumPy buffers as compact codes: 225 bytes of board plus about 80 bytes of counts per candidate.
The GPU expands the codes to planes, scores every candidate of every game in one pass, and a move
is sampled per game from `softmax(Q / temperature)`. The temperature is in points and falls over
the run. Evaluation never samples.

Candidates are pruned because a typical position has about 600 legal placements and most are
plainly bad. The cost of pruning to 64 has not been measured yet (see the plan).

## Warm start

Before the network plays, the static evaluator plays 100,000 games with a high temperature, so the
replay buffer starts with a spread of good and bad moves, and the network is pretrained on them.

## Evaluation

Every seed is played twice with the players swapping seats; the statistic is the mean over pairs
of the player's average spread in its two games, with a 95% interval from the per-pair standard
deviation. During training the network is evaluated against `greedy-leave` and `greedy`, and
against the best network so far; `best.pt` changes only when the new network wins with an interval
that excludes zero.

## Why the engine can be trusted

- The move generator is compared with an independent brute-force reference, written by a
  different author from the rules document alone: 9,160 positions, 28.4 million plays, identical.
- 50 hand-worked scoring cases, checked against the reference, the engine's validator and its
  generator.
- A second scorer and validator inside the engine re-checks generated moves in the tests.
- 100,000 random games with the invariants of the rules document checked after every turn.

The commands and results are in [BENCHMARKS.md](BENCHMARKS.md).

## Differences from the original brief

1. **Transpose is the only board symmetry used.** The premium layout has eight symmetries, but
   mirroring a position spells its words backwards, so only swapping rows with columns gives a
   real position.
2. **The engine sends compact codes, not one-hot planes**, and the GPU expands them: about 25
   times less memory to fill, copy and store.
3. **Cross-check planes are on by default**, not an optional extra, for the reason given above.
4. **Config is TOML read into dataclasses**, not YAML: standard library, no type guessing.
5. **Leave values are a fitted linear model** (tile counts, vowel/consonant balance, tile pairs,
   by stage of the game), not a full table of every leave. It is what the data supports: a table
   entry needs thousands of samples to be estimated to a point, and the network is the proper
   successor to a table anyway.
6. **Resuming does not restore games that were in progress**; it restores everything else and
   starts fresh games.
7. **Two engine crates plus a reference crate**, so `cargo test` needs no Python and the oracle
   shares no code with the engine.
