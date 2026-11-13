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

1. **Warm start** (about 3 minutes): the static evaluator plays 100,000 games, with some
   randomness, to fill the replay buffer.
2. **Pretraining** (about 2 minutes): 3,000 training steps on those games.
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

The settings that matter most:

| Setting | Default | What it does | When to change it |
|---|---|---|---|
| `model.blocks`, `model.channels` | 6, 64 | Size of the network | A bigger network (10 x 128) is about 3 times slower per game. Try it once the small one has clearly learned something. Cannot be changed when resuming. |
| `selfplay.max_candidates` | 64 | Moves per decision the network evaluates | 32 is about 30% faster. Lower it only if evaluation shows no loss of strength. |
| `selfplay.temperature_start`, `temperature_end`, `temperature_decay_games` | 4.0, 0.5, 300,000 | How much self-play explores, in points, and how fast that falls | Raise the end value if training stalls from too little variety; lower it if late-run games look random. |
| `train.learning_rate` | 1e-3 | Step size | Lower it (3e-4) if the loss jumps around or goes up; this is the first thing to try when a run is unstable. |
| `train.replay_ratio` | 4.0 | Training samples per new position | Higher learns more from each game and slows self-play. |
| `train.td_lambda` | 1.0 | 1 trains on actual game outcomes; lower mixes in the network's own estimates | Try 0.8 to 0.9 after a first run works, to cut the noise in the targets. |
| `train.replay_capacity` | 3,000,000 | Positions kept in RAM (0.9 GB) | Raise it to train on a longer history; 10,000,000 is about 3 GB. |
| `selfplay.win_weight` | 0.0 | 0 picks moves by expected spread, 1 by win probability | Leave at 0 while training. Use it at evaluation time to see how win-seeking play does. |
| `model.cross_checks` | true | Whether the network sees which letters fit on each empty square | Turn off only as an experiment to measure what it is worth. Cannot be changed when resuming. |
| `warmstart.games` | 100,000 | Size of the warm start | 0 skips it and starts self-play with an untrained network (which, with `leave_baseline`, plays like the greedy+leave bot). |
| `train.eval_minutes`, `eval_pairs` | 20, 250 | How often and how carefully to evaluate | More pairs narrow the confidence interval and cost a few seconds each time. |

## What a run costs

Measured on this PC (RTX 5070 Ti, Ryzen 7 7800X3D) with the default 6 x 64 network; see
[BENCHMARKS.md](BENCHMARKS.md).

- Self-play and training together ran at 58 games per second in a five-minute dry run. If that
  holds, an hour is about 210,000 games and 5 million new positions, and 8 hours is about 1.7
  million games. That is an extrapolation; the run's own log is the real figure.
- GPU memory: under 2 GB. RAM: about 1 GB for the replay buffer.
- Disk: up to 0.9 GB for the replay buffer plus about 20 MB per checkpoint (5 are kept).

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
