# App API

The desktop app is a local web UI talking to a small HTTP server in the same process as the
engine (`python -m scrabble_ai.app`). It listens on 127.0.0.1 only. This file is the contract
between the two: every endpoint, every JSON shape. All bodies are JSON; times are Unix seconds.

Errors are `{"error": "message"}`: **400** when the request cannot be carried out (a bad value, a
name in use, a move that is not legal; the message is written to be shown to the user), **404**
for a run, model, match, game or endpoint that does not exist, **500** for a bug in the server
(logged with its traceback in `data/app/app.log`). Numbers that are not finite (a diverged loss
is logged as NaN) arrive as `null`.

## Vocabulary

- **Player spec** (string): `human`, `bot:random`, `bot:greedy`, `bot:greedy-leave`, or
  `model:<model id>`.
- **Model id**: `<run name>/<checkpoint file>`, for example `night1/best.pt`.
- **Square**: `[row, col]`, both 0-based, row 0 at the top. `H8` is `[7, 7]`.
- **Board**: 15 strings of 15 characters. `.` empty, upper case a tile, lower case a blank
  standing for that letter.
- **Rack**: upper-case letters, `?` for a blank.
- **Spread** is always from the first-named player's side (player A, or seat 0).

## Status

### `GET /api/status`

```json
{
  "version": "0.1.0",
  "lexicon": {"id": "NSWL2023-91a67230e1f6", "words": 195747},
  "leaves": {"ok": true, "file": "NSWL2023-91a67230e1f6.leaves"},
  "gpu": {"name": "NVIDIA GeForce RTX 5070 Ti", "utilization": 87, "memory_used_mb": 2140,
          "memory_total_mb": 16303, "temperature_c": 61},
  "warnings": [{"kind": "gpu_memory", "message": "Other programs are using 15.2 of 16 GB of GPU memory. Training started now would be slow until they are closed."}],
  "active_runs": ["night1"],
  "queued_runs": ["night2"],
  "active_matches": ["m_20261112_173005"]
}
```

`gpu` is `null` if `nvidia-smi` is unavailable. `warnings` is usually empty; show each message
where the user is about to start something. Poll this every 3 to 5 seconds for the header.

## Training configuration

### `GET /api/config/schema`

Everything the "new run" form needs.

```json
{
  "sections": [
    {
      "name": "model", "title": "Network",
      "fields": [
        {"key": "model.blocks", "label": "Residual blocks", "type": "int", "default": 6,
         "min": 1, "max": 40, "step": 1, "multiple_of": null, "unit": null, "options": null,
         "advanced": false,
         "locked_on_resume": true,
         "help": "Depth of the network. More blocks is slower and needs more games."}
      ]
    }
  ],
  "presets": [
    {"name": "Default 6x64", "description": "About 55 games/s. A good first run.",
     "overrides": {}},
    {"name": "Larger 10x128", "description": "About 18 games/s.",
     "overrides": {"model.blocks": 10, "model.channels": 128}}
  ],
  "speed": [
    {"network": "6x64", "parameters": 1055056, "games_per_second": 55, "measured": true,
     "vram_gb": 1.4},
    {"network": "10x128", "parameters": 3655316, "games_per_second": 18, "measured": false,
     "vram_gb": 4.1}
  ]
}
```

- `type` is `int`, `float`, `bool`, `str` or `choice` (then `options` is a list of strings).
- `step` is how far the arrow keys move a number, not a rule. `multiple_of` is a rule (8 for
  `model.channels`, `null` elsewhere).
- `unit` is what a number counts (`hours`, `games`, `points`, `steps`, `positions`, `minutes`,
  `pairs`, `evaluations`, `racks`) or `null`; show it beside the input.
- Every field also has `"effects": {"lower": "...", "higher": "...", "typical": "..."}`: what
  turning the setting down does, what turning it up does (for an on/off setting, `lower` is off
  and `higher` is on; for a choice, each describes one option), and the range worth trying. All
  three are sentences for the user; `typical` may be an empty string.
- `advanced` fields belong behind a "Show all settings" toggle.
- `speed` rows are benchmark figures for this machine. `measured: false` means worked out from
  separate measurements, so label those "estimated". For a size not in the list, interpolate
  nothing: say "not measured".
- Sections, in order: `run` (name, time limit, game limit), `model`, `selfplay`, `warmstart`,
  `train`, `offline` (training from games, below).

## Runs

A run is one training run in `runs/<name>/`. Runs are separate processes: they keep going when
the app window is closed, and the app finds them again when it opens.

### Run summary object

```json
{
  "name": "night1",
  "status": "running",
  "phase": "selfplay",
  "network": "6x64",
  "parameters": 1055056,
  "games": 123456,
  "step": 8200,
  "positions": 2900000,
  "hours": 1.23,
  "games_per_second": 55.1,
  "created": 1794520000.0,
  "updated": 1794524500.0,
  "limit": {"hours": 8.0, "max_games": 0},
  "progress": 0.154,
  "latest_eval": {
    "games": 120000,
    "greedy-leave": {"spread": 4.2, "ci": 7.8, "win_rate": 0.51},
    "greedy": {"spread": 41.9, "ci": 9.6, "win_rate": 0.677}
  },
  "best_step": 1263,
  "has_best": true,
  "checkpoints": 3,
  "changed": {"model.blocks": 10, "model.channels": 128},
  "error": null,
  "trend": {"games": [23228, 46612], "spread": [-7.7, -8.5], "ci": [6.9, 7.8]},
  "best_eval": {"games": 23228, "step": 4085, "spread": -7.7, "ci": 6.9},
  "health": {"state": "worse", "edge": -5.0,
             "note": "It got worse as it trained: its best network is the first one evaluated."}
}
```

- `status`: `queued` (waiting behind another run), `running`, `stopping` (STOP requested),
  `finished` (reached its limit), `stopped` (stopped by the user or interrupted), `failed`
  (the process died; `error` has the last lines of its log).
- `phase`: `warmstart`, `pretraining` or `selfplay`; for a run that trains from games,
  `training`. `null` when not started.
- `progress`: 0 to 1 against the time or game limit while the run is going, 1 when it has
  finished, `null` if it has no limit or is queued, stopped or failed.
- `latest_eval` is `null` before the first evaluation. `ci` is the half-width of the 95% interval.
- `changed`: settings that differ from the defaults.
- `trend`: every evaluation against greedy-leave, oldest first, as parallel arrays (for a small
  chart beside the run). Empty arrays before the first evaluation.
- `best_eval`: the evaluation against greedy-leave with the highest spread, or `null`.
- `health`: the one-line verdict on how training went, or `null` before the first evaluation.
  `state` is `ahead` (the latest evaluation is significantly above greedy-leave), `level`
  (its interval includes zero), `behind` (significantly below), or `worse` (significantly
  below the run's own best evaluation: training made it worse). `edge` is the average, over
  the last ten self-play reports, of leave-only error minus the network's error on unseen games,
  in points of RMSE (`null` if there are none); below zero the network predicts unseen games
  worse than leave values alone, which is what a network memorising its training games does.
  `note` is a sentence for the user.

### `GET /api/runs`

`{"runs": [<run summary>, ...]}`: runs that are going first, then queued ones, then the rest
with the most recently updated first.

### `POST /api/runs`

Start one or more runs.

```json
{
  "mode": "parallel",
  "runs": [
    {"name": "night1", "overrides": {"run.hours": 8, "model.blocks": 6}},
    {"name": "night1-wide", "overrides": {"run.hours": 8, "model.blocks": 10, "model.channels": 128}}
  ]
}
```

`mode` is `parallel` (all start now, sharing the GPU) or `queue` (one training run on the GPU
at a time, in order: a queue also waits for every run that is already training or queued,
however it was started). Returns `{"started": ["night1"], "queued": ["night1-wide"]}`; with
something already going, a queue returns `{"started": [], "queued": [...all of them]}`.

Errors (400): a name already in use (compared without regard to case, as Windows folders are),
a name with characters other than letters, digits, `-` and `_`, an invalid setting. A message
about a setting names it by its full key first (`train.min_replay must be at least
train.batch_size`), and when several runs were posted it starts with the run's name and a colon.
`run.name` and `run.out_dir` are the app's to set and are refused in `overrides`.

### `GET /api/runs/<name>`

```json
{
  "run": <run summary>,
  "config": {"run": {...}, "model": {...}, "selfplay": {...}, "warmstart": {...}, "train": {...}},
  "checkpoints": [{"file": "step_000001263.pt", "step": 1263, "size_bytes": 20480000,
                   "modified": 1794524500.0, "model_id": "night1/step_000001263.pt"}],
  "best": {"file": "best.pt", "step": 1263, "model_id": "night1/best.pt", "size_bytes": 4300000,
           "modified": 1794524500.0}
}
```

`best` is `null` until the first evaluation.

### `GET /api/runs/<name>/metrics?since=<n>`

`{"records": [...], "next": <n>}`. Each record is one line of the run's `metrics.jsonl`; pass
`next` back as `since` to get only new ones. Every record has `kind`, `time`, `step`, `games`,
`positions`, `hours`. By kind:

| kind | Extra fields |
|---|---|
| `warmstart` | `games_played`, `positions_added`, `seconds`, `mean_score` |
| `train` | `loss`, `value_loss`, `win_loss`, `value_error_points` (mean absolute error on the training batches, in points), `value_rmse_points` (the same as RMSE; only in runs made after 2026-11-13), `learning_rate` |
| `selfplay` | `games_per_second`, `positions_per_second`, `mean_score`, `mean_turns`, `mean_abs_spread`, `first_player_win_rate`, `temperature`, `value_rmse_points`, `leave_only_rmse_points`, `win_brier`, `replay_positions`, `replay_gb`, `bingos_per_game`, `exchanges_per_game`, `passes_per_game`, `max_game_score`, `max_play_score` |
| `eval` | `opponent` (`greedy-leave` or `greedy`), `spread`, `spread_ci`, `win_rate`, `win_ci`, `score_for`, `score_against`, `pairs` |
| `gate` | `spread`, `spread_ci`, `win_rate`, `best_step`, `promoted` (bool), `pairs` |
| `checkpoint` | `file`, `replay_positions` |
| `session` | `event`: `start` (with `parameters`, `limit_hours`, `limit_games`), `selfplay` (the network starts playing itself) or `stop` (with `reason` and `finished`) |
| `guard` | `action` (`rollback`), `to_step` (the best network's step, whose weights were restored), `learning_rate_scale` (what the learning rate is now multiplied by), `reason` |

`train` and `selfplay` arrive about once a minute, `eval` and `gate` every `train.eval_minutes`.
`value_rmse_points` against `leave_only_rmse_points` is the honest progress signal between
evaluations: both are measured on games the network had not trained on, and the network's
should fall below the leave-only figure.

### `GET /api/runs/<name>/log?tail=<n>`

`{"lines": ["[17:09:24] run 'night1' in ...", ...]}`: the last `n` console lines (default 200).

### `POST /api/runs/<name>/stop`

Ask the run to stop cleanly (it finishes its step, evaluates, saves). `{"ok": true}`. With body
`{"force": true}` the process is killed instead; anything since the last checkpoint is lost.
For a queued run this takes it out of the queue (its status becomes `stopped`; resume starts it).

### `POST /api/runs/<name>/rename`

Body `{"name": "sixty-four-run-1"}`. Renames the run, and with it every model of the run (models
are named after their run). Returns the run summary under its new name. The run's folder is
renamed, its saved settings, stars and notes follow, and matches already played show the new
name and can still be replayed; games in progress against one of its models carry on under the
new name. The checkpoint files are not rewritten (the app resumes a run by its folder's name).
400 while the run is running, stopping or queued, while a match using one of its models is
still running, for a name that is not allowed or already in use (compared without regard to
case), or the same name (a change of case alone is allowed); 404 for a run that does not exist.

### `POST /api/runs/<name>/resume`

Body `{"hours": 8, "max_games": 0, "overrides": {"train.learning_rate": 0.0003}}`, all optional.
Returns the run summary.

- `hours` is the time limit of the new session (0 for none). Without it the session gets the
  run's own time limit again.
- The run continues from its newest checkpoint with the settings saved in it, changed by
  `overrides`. Settings a checkpoint fixes cannot change: `run.lexicon`, `run.leaves`, everything
  under `model.`, and `train.value_scale`.
- A run that never reached a checkpoint (taken out of a queue, or dead in its first seconds) is
  started again as the run that was asked for, with `overrides` on top.
- 400 if the run is running, stopping or queued, or for an unknown, invalid or locked setting.

### `DELETE /api/runs/<name>`

Deletes the run's folder, checkpoints included. 400 while it is running. A queued run can be
deleted; its queue skips it. Stars and notes on its models go with it. `{"ok": true}`.

### `GET /api/runs/<name>/report`

Analysis of a run so far, for the post-training page (and usable mid-run).

```json
{
  "headline": "night1 beat greedy-leave by 6.1 points a game at its best, and the gain is significant.",
  "summary": {
    "games": 1612000, "hours": 8.0, "steps": 71000, "mean_games_per_second": 56.0,
    "best": {"step": 61000, "games": 1400000, "spread": 6.1, "ci": 3.9, "win_rate": 0.53},
    "final": {"step": 71000, "games": 1612000, "spread": 5.2, "ci": 4.1, "win_rate": 0.525},
    "first_significant": {"games": 520000, "hours": 2.6},
    "promotions": 7,
    "value_rmse_gain": 1.9
  },
  "events": [
    {"games": 100000, "hours": 0.08, "kind": "phase", "tone": "info",
     "title": "Warm start finished", "detail": "100,000 games in 176 s."},
    {"games": 520000, "hours": 2.6, "kind": "significant", "tone": "good",
     "title": "First significant win over greedy-leave", "detail": "+5.4 +/- 4.9 over 250 pairs."},
    {"games": 900000, "hours": 4.5, "kind": "regression", "tone": "warning",
     "title": "Evaluation dropped", "detail": "Fell 6.3 points from the previous evaluation."}
  ]
}
```

- `tone` is `good`, `info` or `warning`.
- Event kinds: `phase`, `significant`, `promotion`, `regression`, `plateau`, `loss_spike`,
  `slowdown`, `record` (a new high for a game score or a single play in self-play), `stop`.
- Any field of `summary` can be `null` when there is not enough data yet.
- `promotions` counts gate matches won. The first evaluated network becomes the best without a
  match and is reported as its own event ("First network to beat"), not counted.
- A run whose process died gets a last `stop` event with tone `warning`, "Run ended unexpectedly".
- More event kinds: `overfit` (the network began predicting unseen games worse than leave values
  alone) and `guard` (the trainer went back to its best network and lowered the learning rate).
- `summary.health` is the same object as the run summary's `health`.

## Training from saved games

The second way to train (`scrabble-ai train-games`, `src/scrabble_ai/offline.py`): first a **game
set** is made, a folder of games the static evaluator played against itself; then a run learns
from it, one position from every game per pass. Such a run is an ordinary run in `runs/`, started,
stopped, resumed, renamed and deleted through the endpoints above. What tells the two kinds apart
is its `mode`.

### `GET /api/gamesets`

```json
{
  "sets": [
    {"name": "twenty-million", "status": "building", "games": 6200000, "target": 20000000, "progress": 0.31,
     "bytes": 1773200000, "lexicon_id": "NSWL2023-91a67230e1f6", "leaves_hash": "ab12...",
     "opening_plies": 4, "opening_temperature": 6.0,
     "mean_turns": 23.0, "mean_score": 432.0, "first_player_win_rate": 0.554, "bingos_per_game": 3.94,
     "games_per_second": 1040.0, "seconds": 5961.5, "created": 1794700000.0, "updated": 1794705961.5,
     "error": null}
  ],
  "estimate": {"bytes_per_game": 286.0, "games_per_second": 1040.0}
}
```

- `status`: `building` (a process is adding games), `stopping` (it has been asked to stop and
  will after its current block of 5,000 games), `complete` (it holds its `target`), `partial`
  (stopped short of it; `POST` again with the same or a larger number to carry on), `empty`,
  or `failed` (the process died; `error` holds the end of its log).
- `progress` is `games / target`, or `null` with no target.
- The statistics (`mean_turns` to `games_per_second`) are `null` until the set has games.
  `mean_score` is per player; `bingos_per_game` counts both players' together.
  `lexicon_id` and `leaves_hash` are `null` until the process is up (it records them before
  its first block, so they can be set while `games` is still 0).
- `estimate` is for sizing a set before it is made: bytes on disk per game, from the sets that
  exist (286, measured, if there are none), and games per second from the most recent set, or
  `null` if no set has been timed. Building uses every CPU core and no GPU.
- Sets are listed building first, then the most recently updated.

### `POST /api/gamesets`

Body `{"name": "twenty-million", "games": 20000000, "opening_plies": 4, "opening_temperature": 6.0}`.
Creates the set and starts playing, or, for a set that exists, carries on up to `games` (which
must be more than it has). The two opening settings are optional (4 and 6.0 for a new set) and
cannot differ from an existing set's. Returns the set's summary. 400 for a bad name (letters,
digits, `-` and `_`, at most 64), a number of games outside 1 to 500,000,000, a set that is
already being built, or leave values not yet fitted.

`opening_plies`: each game varies between 0 and this many of its first plies (so half of this on
average); on those plies the move is drawn from the best 16 by score + leave value.
`opening_temperature`: how freely, in points: a move this many points worse is chosen about a
third as often. After the opening every move is the static evaluator's first choice.

### `GET /api/gamesets/<name>`, `GET /api/gamesets/<name>/log?tail=<n>`

The set's summary; the last lines of the output of the processes that built it, `{"lines": [...]}`.
404 for a set that does not exist, as for every route that names a set (also a name that could
not be one).

### `POST /api/gamesets/<name>/stop`

Body `{}` or `{"force": true}`. Asks the process to stop after its current block (nothing is
lost, and the set can be carried on later); with `force`, kills it (the set keeps what was
complete). 400 if the set is not being built.

### `DELETE /api/gamesets/<name>`

Deletes the folder. 400 while it is being built, or while a run that is running, stopping or
queued is training from it.

### Runs that train from games

Start one with `POST /api/runs` as usual; what makes it this kind is the override
`"offline.games": "<set name>"`. The server then applies `games_defaults` from the schema before
the run's own overrides, as `scrabble-ai train-games` does. 400 if the set does not exist, has
no games, or is still being built. Runs of both kinds can be queued together: both use the GPU.

The run summary gains, for every run:

```json
"mode": "games",
"game_set": "twenty-million",
"total_steps": 58500,
"passes": 6.0,
"pass": 2.14,
"positions_per_second": 41250.0
```

- `mode` is `"selfplay"` or `"games"`. The other five are `null` for a self-play run.
- `total_steps` is `null` until the run has started; `pass` is how many passes through the set
  are done, as a fraction. `step` is the training step and `games` (and `positions`) count the
  positions trained on, one from a game each. `progress` is `step / total_steps`.
- `latest_eval` entries, for runs of both kinds, also carry `"win_ci"` (the half-width of the 95%
  interval on `win_rate`, or `null` in runs from before it was recorded).
- A games run is **judged by its win rate**, not its spread: a network trained to win takes moves
  that score a little less and win a little more. So for these runs `health.state` is `ahead`
  when the latest win rate's whole interval is above 0.5, `behind` when it is below, `level`
  otherwise, and `worse` when it is significantly below the run's best and not ahead; `health.edge`
  is always `null`. `trend` has three more parallel arrays, `step`, `win_rate` and `win_ci`, and
  `best_eval` is the match with the highest win rate:
  `{"games", "step", "hours", "win_rate", "win_ci", "spread", "ci", "pairs"}`.
- Resume takes `hours` and `overrides`; `max_games` has no meaning for these runs. `offline.games`,
  `offline.batch_size` and `offline.validation_games` cannot change on resume (they decide which
  games each step trains on and which are held out; the schema marks them `locked_on_resume`),
  and a run cannot be resumed if its game set has changed size since.
- A games run that died has, like any other, a last event `Run ended unexpectedly` in its report.

Metrics records of a games run (`GET /api/runs/<name>/metrics`), all with the common fields:

| `kind` | Fields |
|---|---|
| `session` | `event`: `start` (with `mode: "games"`, `parameters`, `limit_hours`, `game_set`, `train_games`, `validation_games`, `total_steps`, `steps_per_pass`, `batch`) or `stop` (with `reason`, `finished`) |
| `train` | About once a minute: `loss`, `win_loss`, `spread_loss`, and with helper outputs `next_score_loss`, `next_bingo_loss`, `next_squares_loss`; the weights in force, `spread_weight` and with helper outputs `next_score_weight`, `next_bingo_weight`, `next_squares_weight`; `learning_rate`, `positions_per_second`, `pass` |
| `valid` | Every `offline.validate_steps` steps, on positions from games never trained on: the same `*_loss` fields, `win_accuracy` (share of positions whose most likely result was the actual one), `win_brier`, `value_rmse_points`, `leave_only_rmse_points`, `positions_checked` |
| `eval` | A match against the static evaluator: `opponent` (`greedy-leave`), `spread`, `spread_ci`, `win_rate`, `win_ci`, `score_for`, `score_against`, `pairs` |
| `best` | The match just logged was the best so far and its network was saved as `best.pt`: `win_rate`, `win_ci`, `pairs` |
| `checkpoint` | `file` |

`GET /api/runs/<name>/report` for a games run has the same three parts in other terms:

```json
{
  "headline": "night5 wins 53.4% +/- 1.3 of games against the static evaluator, more than half beyond the margin of error.",
  "summary": {
    "steps": 58500, "positions": 119808000, "hours": 5.4, "passes": 6.0, "game_set": "twenty-million",
    "train_games": 19980000, "mean_positions_per_second": 40100.0,
    "best": {"games": 99840000, "step": 48750, "hours": 4.5, "win_rate": 0.5361, "win_ci": 0.013, "spread": 2.1, "ci": 5.2, "pairs": 2000},
    "final": {"...": "the same fields"},
    "first_significant": null,
    "held_out": {"win_loss": 0.6521, "win_accuracy": 0.612, "value_rmse_points": 71.3, "leave_only_rmse_points": 73.9},
    "evaluations": 11,
    "health": {"state": "ahead", "edge": null, "note": "..."}
  },
  "events": [{"games": 0, "hours": 0.0, "kind": "phase", "tone": "info", "title": "Training began", "detail": "..."}]
}
```

`best`, `final` and `first_significant` are `null` when there is none; `held_out` is `null` before
the first check. Event kinds are `phase`, `significant`, `promotion` and `stop`.

The schema (`GET /api/config/schema`) describes both kinds of run:

- Every field has `"modes"`, a list holding `"selfplay"`, `"games"` or both: which form shows it.
- There is a sixth section, `offline` ("Training from games").
- `"games_presets"`: starting points for a games run, each `{name, description, overrides}`.
- `"games_defaults"`: the settings a games run changes before its own overrides (for example
  `"model.head": "pooled"`), so that the form can show the defaults that will really apply.
- New `[model]` settings choose the network: `model.kind` (`cnn`, `cnn-global`, `transformer`),
  `model.head`, `model.win_head`, `model.aux_heads`. `train.target` (`td` or `q`) chooses the
  learning rule of a self-play run; `q` is deep Q-learning.

A candidate object (below) may carry `"utility"`: for a network that picks moves by win chance,
the number it ranks them by. Candidates are then listed in that order, not by `total`.

## Model library

### `GET /api/models`

```json
{
  "bots": [
    {"spec": "bot:greedy-leave", "name": "Greedy + leave", "description": "Highest score plus leave value."},
    {"spec": "bot:greedy", "name": "Greedy", "description": "Highest score."},
    {"spec": "bot:random", "name": "Random", "description": "A random legal placement."}
  ],
  "models": [
    {
      "id": "night1/best.pt", "spec": "model:night1/best.pt", "name": "night1 (best)",
      "run": "night1", "file": "best.pt", "kind": "best",
      "network": "6x64", "parameters": 1055056, "step": 61000, "games": 1400000,
      "modified": 1794524500.0, "size_bytes": 4300000, "compatible": true,
      "incompatible_reason": null,
      "eval": {"greedy-leave": {"spread": 6.1, "ci": 3.9, "win_rate": 0.53},
               "greedy": {"spread": 44.0, "ci": 4.4, "win_rate": 0.69}},
      "starred": false, "note": ""
    }
  ]
}
```

- `kind`: `best` (the run's gated best network) or `checkpoint`.
- `compatible` is false when the model was trained with a different word list or leave values
  than are installed; it cannot be played then, and `incompatible_reason` says which in a sentence.
- `eval` is the evaluation of that network recorded during its run, or `null`.

### `POST /api/models/<id>/meta`

Body `{"starred": true, "note": "first to beat the baseline"}` (either field). Returns the model;
404 for a model that does not exist.

## Matches (fast, no board)

A match plays N seeds twice each with the two players swapping seats, as a background job.

### `POST /api/matches`

Body `{"a": "model:night1/best.pt", "b": "bot:greedy-leave", "pairs": 500, "candidates": 64}`.
Returns `{"id": "m_20261112_173005"}`. `candidates` (default 64) is how many moves a network
evaluates per decision.

### `GET /api/matches`

`{"matches": [<match summary>, ...]}`, newest first. A match summary:

```json
{"id": "m_20261112_173005", "a": "model:night1/best.pt", "b": "bot:greedy-leave",
 "a_name": "night1 (best)", "b_name": "Greedy + leave", "pairs": 500,
 "status": "finished", "progress": 1.0, "games_done": 1000, "created": 1794524500.0,
 "seconds": 21.4, "spread": 6.1, "spread_ci": 3.4, "win_rate": 0.531}
```

`status`: `running`, `finished`, `cancelled`, `failed` (then `error` says why; it is `null`
otherwise). The last three numbers are `null` until at least two pairs are done; they update
while the match runs.

### `GET /api/matches/<id>`

```json
{
  "match": <match summary>,
  "summary": {
    "spread": 6.1, "spread_ci": 3.4, "win_rate": 0.531, "win_ci": 0.022, "significant": true,
    "score_a": 436.2, "score_b": 430.1, "turns": 23.1,
    "a_first": {"spread": 18.0, "win_rate": 0.58}, "a_second": {"spread": -5.8, "win_rate": 0.48},
    "bingos_a": 1.91, "bingos_b": 1.84, "exchanges_a": 0.31, "exchanges_b": 0.28,
    "passes_a": 0.02, "passes_b": 0.02, "ended_by_going_out": 0.997
  },
  "games": {
    "seed": [7000000000000, 7000000000000], "a_first": [true, false],
    "score_a": [455, 398], "score_b": [401, 420], "turns": [22, 25],
    "bingos_a": [2, 1], "bingos_b": [1, 2], "exchanges_a": [0, 1], "exchanges_b": [0, 0],
    "passes_a": [0, 0], "passes_b": [0, 0],
    "best_play_a": [98, 74], "best_play_b": [83, 92],
    "best_word_a": ["RETAINS", "QUIXOTIC"], "best_word_b": ["ZANIEST", "OUTJINX"],
    "ended_by": ["out", "out"]
  },
  "highlights": [
    {"title": "Highest-scoring play", "detail": "QUIXOTIC for 212 by night1 (best)", "game": 143},
    {"title": "Biggest win", "detail": "night1 (best) won 612-298", "game": 77},
    {"title": "Closest game", "detail": "Tied at 401", "game": 12}
  ]
}
```

- `games` is columnar: index `i` across the arrays is one game. Games `2k` and `2k+1` share a
  seed with seats swapped. Arrays hold only the games finished so far.
- `highlights[].game` indexes into `games`; with its `seed` and `a_first` the game can be
  replayed in the viewer: create a watch game with the same seed and the player who moved first
  as `players[0]`. Play is deterministic, and the viewer's bots are the engine's own, with the
  same random stream as the match, so the replay is the same game move for move (a test checks
  this for every bot and for a network).
- `summary` fields are `null` until there is enough data.

### `POST /api/matches/<id>/cancel`

`{"ok": true}`.

## Interactive games (a board on screen)

Used for human against a model and for watching two players. The server keeps the game; the UI
asks for moves.

### `POST /api/games`

Body `{"players": ["human", "model:night1/best.pt"], "seed": 12345, "advisor": "model:night1/best.pt"}`.
`players[0]` moves first. `seed` is optional (random if absent). `advisor` (optional, default the
strongest non-human player in the game, else `bot:greedy-leave`) is who answers `hint`. Returns
the game state.

### Game state object

```json
{
  "id": "g_3f9a12", "seed": 12345,
  "players": [{"spec": "human", "name": "You"}, {"spec": "model:night1/best.pt", "name": "night1 (best)"}],
  "advisor": "model:night1/best.pt",
  "board": ["...............", "... 15 rows ..."],
  "last_placed": [[7, 7], [7, 8], [7, 9]],
  "racks": ["AEINRST", "QZ?EEIO"],
  "scores": [120, 98],
  "to_move": 0, "turn": 9, "bag": 42, "scoreless_turns": 0, "can_exchange": true,
  "unseen": {"A": 5, "B": 1, "?": 1},
  "unseen_by_seat": [{"A": 5, "B": 1, "?": 1}, {"A": 6, "B": 1}],
  "over": false,
  "result": null,
  "history": [
    {"turn": 1, "player": 0, "kind": "play", "text": "8D QUIZ", "word": "QUIZ", "position": "8D",
     "score": 48, "total": 48, "tiles": [[7, 3, "Q", false], [7, 4, "U", false]],
     "rack": "QUIZAEN", "leave": "AEN", "bingo": false, "exchanged": "", "analysed": false}
  ]
}
```

- `racks` holds both racks. In a human game the UI hides the opponent's rack unless the user
  turns on "show opponent's rack".
- `unseen` is from the side to move: tiles on neither the board nor their rack.
  `unseen_by_seat` gives the same from each seat's side, whoever is to move.
- History `kind`: `play`, `exchange` (then `text` is like `exchange 3`, and `tiles` is empty;
  `exchanged` has the letters, which a human opponent would not have seen), `pass`.
- History `analysed` is true when an AI made the move in this session: what it considered can be
  fetched with `analysis?turn=`.
- `result` when `over`: `{"scores": [431, 398], "winner": 0, "adjustment": [14, 0],
  "ended_by": "out"}`. `winner` is `null` for a tie; `ended_by` is `out` or `scoreless`.

### `GET /api/games/<id>`

The game state.

### `POST /api/games/<id>/move`

A human move. One of:

- `{"tiles": [{"row": 7, "col": 7, "letter": "C", "blank": false}, ...]}`: the newly placed tiles
  only. For a blank, `letter` is the letter it stands for and `blank` is true.
- `{"exchange": "AEI?"}`
- `{"pass": true}`

Returns the game state, or 400 with the reason the move is illegal (for example
`"CATOX is not a word"`, `"the rack does not hold the tiles for this play"`).

### `POST /api/games/<id>/ai`

Have the non-human player to move choose and play. Returns

```json
{"state": <game state>,
 "analysis": {"player": "night1 (best)", "turn": 9, "considered": 64, "candidates": [<candidate>, ...]}}
```

`candidates` are the best 12 by `total`, plus the move played if it is not among them.
400 if the side to move is human or the game is over.

### `GET /api/games/<id>/analysis?turn=<n>`

The `analysis` object of the move played on turn `n`, kept for every AI move of the session
(so a page opened part-way through a game can show earlier ones). 404 if that turn has none.

### `GET /api/games/<id>/hint?top=<n>`

The advisor's ranking of the side to move's options, best first (`top` from 1 to 200, default
15): `{"advisor": "night1 (best)", "candidates": [<candidate>, ...]}`. Asking changes nothing in
the game.

### Candidate object

```json
{"text": "8D QUIZ", "kind": "play", "word": "QUIZ", "position": "8D", "score": 48,
 "tiles": [[7, 3, "Q", false]], "leave": "AEN", "exchanged": "", "leave_value": 3.2,
 "value": 5.9, "total": 53.9, "win": 0.61, "chosen": true}
```

- `text`: for a placement, the square and the word, with letters already on the board in
  brackets and blanks in lower case (`8D QU(I)z`). For an exchange, `exchange AEI` (the letters
  are also in `exchanged`). `pass` for a pass.
- `score`: the move's own points, the same number the move history shows.
- `leave_value`: static value of the tiles kept, in points.
- `value`: the player's estimate of everything still to be gained after the move, in points. For
  a network this is its output; for `bot:greedy-leave` it equals `leave_value`; for `bot:greedy`
  and `bot:random`, which do not look past the score, it is 0. It is usually negative (the
  opponent moves next); only differences between candidates matter. For a move that ends the
  game by going out it is exactly the bonus: twice the value of the opponent's rack.
- `total`: `score + value`, what the ranking is by.
- `win`: estimated win probability, `null` for bots.
- `chosen`: the move actually played (in `ai` responses).

### `DELETE /api/games/<id>`

Forget the game. `{"ok": true}`.

## Other

### `GET /api/leaves?rack=<letters>&bag=<n>`

`{"rack": "ERS", "bag": 60, "value": -5.9, "relative": 18.1}`: the leave value of keeping those
tiles (up to seven, `?` for a blank) with `n` tiles left in the bag (0 to 100, default 60), and
the same relative to keeping nothing.

## Files the app keeps

Everything is under `data/app/` (not in the repository): `server.json` (the running app's port),
`app.log`, `library.json` (stars and notes), `matches/<id>/` (one folder per match),
`queues/` (one file per queue of runs, with its runner's log), `window-profile/` (the app
window's browser profile) and `scrabble-lab.ico`. Training runs live in `runs/<name>/` with the
marker files described at the top of `src/scrabble_ai/app/runs.py`.
