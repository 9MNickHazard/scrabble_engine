# Training

How to start, watch, stop and resume a training run, and how to change what it does.
[DESIGN.md](DESIGN.md) explains what is being trained and why it is built this way.

Everything here can also be done from the desktop app, Scrabble Lab ([APP.md](APP.md)): the same
runs, the same files, the same settings.

## Before the first run

These are one-off steps. On Nick's PC they are already done.

```powershell
cd C:\dev\scrabble-engine
uv sync                        # Python 3.12 environment, PyTorch for the GPU, builds the Rust engine
uv run scrabble-ai gpu-check   # must end with "RESULT: OK"
uv run scrabble-ai lexicon     # confirms the word list in data/lexicon/ loads (195,747 words)
uv run scrabble-ai fit-leaves  # fits leave values for that word list: about 15 minutes
```

`fit-leaves` writes `data/leaves/<lexicon id>.leaves`. Training needs that file: the leave values
rank candidate moves before the network sees them and are the baseline the network corrects.
Re-run it only if the word list changes.

## Start a run

```powershell
uv run scrabble-ai train --config configs/train.toml --name night1 --hours 8
```

- `--name` picks the output folder, `runs/night1/`. Use a new name for a new experiment.
- `--hours` is a wall-clock limit for this session. Without it the run goes until it is stopped.
- Before leaving it overnight, set Windows so the PC does not sleep (Settings, System, Power).

A fresh run does three things in order:

1. **Warm start** (about two and a half minutes): the static evaluator plays 100,000 games, with
   some randomness, to fill the replay buffer.
2. **Pretraining** (about a minute): 2,000 training steps on those games.
3. **Self-play**: the network plays itself and trains on the results until the limit.

It ends with an evaluation against the baseline bots and a final checkpoint.

## Watch it

The console prints a line about once a minute. This one is from a five-minute dry run:

```
games 14,037 (59.2/s), step 1,252, temp 3.84, score 415, value RMSE 73.7 vs leave-only 73.9, replay 807,353
```

- **value RMSE vs leave-only**: how far the network's value was from the spread that actually
  followed, against the same measure for the static leave value alone. These are computed on
  games the network had not been trained on yet, so they are honest. Game outcomes are noisy and
  both numbers stay large; what matters is the network's being lower, and the gap widening.
- **eval lines** (every 20 minutes): `net@1263 vs greedy-leave: spread +4.2 +/- 7.8 ...` is the
  network's mean spread per game over seat-swapped pairs, with a 95% confidence interval. This is
  the number that says whether training is working. Until the interval clears zero, as it had not
  in that dry run, the network has not been shown to beat the baseline.
- **gate lines**: the current network against the best one so far. `best.pt` is replaced only
  when the current network wins with an interval that excludes zero.
- **guard lines**: `guard: back to the best network (step 6,509), learning rate now 1.50e-04`
  means the network was significantly behind the best one at two evaluations in a row, so the
  trainer went back to the best network's weights and halved the learning rate. Once is no cause
  for alarm. Again and again means the settings make the network worse: see
  [TUNING.md](TUNING.md).

**If the value RMSE goes above leave-only and stays there, the run is memorising its own games,
and the evaluation will fall.** Both of the first real runs did this ([PLAN.md](PLAN.md)). The
app's run page says so in words and draws the two errors side by side.

For curves:

```powershell
uv run tensorboard --logdir runs    # then open http://localhost:6006
```

Everything logged also goes to `runs/night1/metrics.jsonl`, one JSON object per line.

## Stop it

Either of these stops cleanly: the run finishes its current step, evaluates, and saves a
checkpoint.

- Press `Ctrl+C` in the training window. (This skips the closing evaluation.)
- From another terminal: `New-Item runs\night1\STOP`

Killing the process or losing power loses at most the work since the last checkpoint (10 minutes
by default). Checkpoint files are written under a temporary name and renamed, so an interrupted
save cannot corrupt the previous one.

## Resume it

```powershell
uv run scrabble-ai train --name night1 --resume latest --hours 8
```

This restores the network, optimizer, replay buffer, counters and random-number streams from the
newest checkpoint in `runs/night1/`. Games that were in progress when the checkpoint was written
are not restored; self-play starts fresh games. A run resumes with the settings saved in its
checkpoint, and `--set` (or `--config`) changes only the settings named:

```powershell
uv run scrabble-ai train --name night1 --resume latest --hours 8 --set train.learning_rate=3e-4
```

A checkpoint refuses to load with a different word list or different leave values than it was
trained with, because its values would no longer mean the same thing.

## The other way: train from saved games

Two commands, or the "Train from games" tab of the app ([APP.md](APP.md)). [DESIGN.md](DESIGN.md)
says how it works and [PLAN.md](PLAN.md) why it was added.

```powershell
uv run scrabble-ai games --name twenty-million --games 20000000      # step 1: play and save the games (CPU only)
uv run scrabble-ai train-games --games twenty-million --name wdl1    # step 2: train a network on them (GPU)
uv run scrabble-ai train-games --name wdl1 --resume latest
```

- **Step 1** uses every CPU core and no GPU: 894 games a second measured here, so 20 million games
  take about six hours and 5.7 GB in `data/games/twenty-million/`. It can be stopped (`Ctrl+C`, or
  a file named `STOP` in the set's folder) and continued by running the same command again; a
  larger `--games` adds to a set. Twenty million is the size at which the published results
  stopped being short of data.
- **Step 2** goes through the set `offline.passes` times (6), taking one position from every game
  each time, then plays a final match of 10,000 pairs against the static evaluator. The console
  shows the loss on games held out of training and, every half hour, a match of 2,000 pairs.
  `best.pt` is the network with the highest win rate in those matches.
- Choose the network with `--set model.kind=cnn` (the default), `cnn-global` or `transformer`
  (with, say, `--set model.blocks=8 --set model.channels=192`).
- Training from games starts from other values than self-play for five settings, and prints them
  as it begins: `model.head=pooled`, `model.win_head=wdl`, `model.aux_heads=true`,
  `selfplay.win_weight=0.9`, `selfplay.max_candidates=32`. `--set` changes any of them. A
  `--config` file changes one only by giving it a value other than the built-in one, so passing
  `configs/train.toml`, which spells out every built-in value, changes nothing.
- Reading positions back out of the games is fast, 187,000 a second, unless the network takes
  the inference about the opponent's rack as an input (`model.belief_samples`, 4 by default):
  that costs four move generations per position and brings it to 11,000 a second, which then
  sets the pace of training. `--set model.belief_samples=0` removes it.
- A network trained this way is judged by how often it wins against the static evaluator, not
  by points; `uv run scrabble-ai eval runs\wdl1\checkpoints\best.pt greedy-leave --pairs 20000`
  prints both.

None of this has been run on the GPU yet, and no network has been trained with it.

To try **deep Q-learning** in an ordinary self-play run: `--set train.target=q`
([DESIGN.md](DESIGN.md), "Deep Q-learning as a learning rule").

## Evaluate a result

```powershell
uv run scrabble-ai eval runs\night1\checkpoints\best.pt greedy-leave --pairs 2000
uv run scrabble-ai eval runs\night1\checkpoints\best.pt greedy --pairs 2000
uv run scrabble-ai eval greedy-leave greedy --pairs 2000        # the baselines against each other
```

Each seed is played twice with the players swapping seats. Players are `random`, `greedy`,
`greedy-leave`, or a checkpoint file. `--seed-offset 5000` draws a fresh set of games.

## Change the settings

Every setting is in [configs/train.toml](../configs/train.toml) with an explanation, and the
values in that file are the defaults. Three ways to change one, later ones winning:

1. Edit `configs/train.toml`, or copy it and pass the copy with `--config`.
2. `--set section.key=value` on the command line, as many times as needed.
3. `--name` and `--hours` are shortcuts for `run.name` and `run.hours`.

A misspelled section or key stops the run with an error naming the valid ones. The settings in
force are saved to `runs/<name>/config.toml` at the start of every session.

[TUNING.md](TUNING.md) has every setting, what lowering and raising it does, and what to try for
each kind of trouble. The settings that matter most:

| Setting | Default | What it does | When to change it |
|---|---|---|---|
| `model.blocks`, `model.channels` | 6, 64 | Size of the network | A bigger network (10 x 128) is about 3 times slower per game. Try it once the small one has clearly learned something. Cannot be changed when resuming. |
| `selfplay.max_candidates` | 64 | Moves per decision the network evaluates | 32 is about 30% faster. Lower it only if evaluation shows no loss of strength. |
| `selfplay.temperature_start`, `temperature_end`, `temperature_decay_games` | 4.0, 0.5, 300,000 | How much self-play explores, in points, and how fast that falls | Raise the end value if training stalls from too little variety; lower it if late-run games look random. |
| `train.learning_rate` | 3e-4 | Step size | Lower it (1e-4) if a run keeps getting worse; raise it towards 1e-3 only if a run is stable and learning slowly. |
| `train.replay_ratio` | 1.0 | Training samples per new position: about how often each position is trained on | Leave it at 1 or below. At 4 the first runs memorised their games. |
| `train.td_lambda` | 0.7 | 1 trains on actual game results; lower leans on the estimate of the next position | 0.5 if a run still memorises; towards 0.9 if its estimates drift. |
| `train.guard`, `guard_patience` | true, 2 | Go back to the best network and halve the learning rate after this many evaluations in a row significantly behind it | Turn it off only to see what a setting does unprotected. |
| `train.replay_capacity` | 3,000,000 | Positions kept in RAM (1.1 GB) | Raise it to train on a longer history; 10,000,000 is about 3.7 GB. |
| `selfplay.win_weight` | 0.0 | 0 picks moves by expected spread, 1 by win probability | Leave at 0 while training. Use it at evaluation time to see how win-seeking play does. |
| `model.cross_checks` | true | Whether the network sees which letters fit on each empty square | Turn off only as an experiment to measure what it is worth. Cannot be changed when resuming. |
| `model.extra_scalars`, `last_move_plane`, `openness`, `hooks` | true | Input groups added on 14 November 2026 ([DESIGN.md](DESIGN.md)) | Turn one off to measure what it is worth. None has been shown to help yet. Cannot be changed when resuming. |
| `model.belief_samples` | 4 | Racks sampled per move to infer what the opponent kept; the result is an input | 0 removes the input and its cost, about 5.6 ms a step per sampled rack ([BENCHMARKS.md](BENCHMARKS.md)). Cannot be changed when resuming. |
| `warmstart.games` | 100,000 | Size of the warm start | 0 skips it and starts self-play with an untrained network (which, with `leave_baseline`, plays like the greedy+leave bot). |
| `train.eval_minutes`, `eval_pairs` | 20, 250 | How often and how carefully to evaluate | More pairs narrow the confidence interval and cost a few seconds each time. |

## What a run costs

Measured on this PC (RTX 5070 Ti, Ryzen 7 7800X3D) with the default 6 x 64 network; see
[BENCHMARKS.md](BENCHMARKS.md).

- The first full run (2.19 hours, with the old replay ratio of 4) played 450,005 games: 57 a
  second overall, about 205,000 games and 5 million new positions an hour. The 10 x 128 network
  managed 18 a second (441,805 games in 6.67 hours). The current replay ratio of 1 does a quarter
  of the training per game, so self-play should be somewhat faster; that has not been measured.
- GPU memory: under 2 GB. RAM: about 1 GB for the replay buffer.
- Disk: up to 1.1 GB for the replay buffer plus about 20 MB per checkpoint (5 are kept).

## Files a run writes

```
runs/night1/
  config.toml        settings in force for the latest session
  metrics.jsonl      every logged number
  tb/                TensorBoard files
  replay/            the replay buffer, rewritten at each checkpoint
  checkpoints/
    step_000014200.pt    full training state (the newest 5 are kept)
    latest.txt           name of the newest
    best.pt              the best network so far, on its own
```

`runs/` is gitignored. Nothing in it should be committed.
