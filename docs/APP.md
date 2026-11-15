# Scrabble Lab, the desktop app

Scrabble Lab is a window onto everything in this project: start and watch training runs, browse
the networks they save, play against them, watch them play each other, and run fast matches
between any two players. It runs entirely on this machine. Nothing is sent anywhere, and there
is no sign-in.

## Start it

Double-click **Scrabble Lab** on the Desktop. From a terminal it is

```powershell
uv run scrabble-ai app                      # open the app
uv run scrabble-ai app --install-shortcut   # (re)create the Desktop shortcut
```

The app is a small local web server (on 127.0.0.1 only, on a free port) with its pages shown in
a window of their own. Closing the window ends the app. **Training runs and matches are separate
processes and keep going with the app closed**; the app finds them again the next time it opens.
Opening the shortcut while the app is already running shows the same app again.

### The look: liquid glass

Every pane of the app is a sheet of frosted glass: translucent, with a bright rim, rounded
corners and a black outline, the way lead holds the panes of a stained-glass window. Buttons,
inputs and tabs are pills. The board, its tiles and the racks are solid, since they are what you
are looking at. The app is **light** (white glass) by default, with a **dark** theme (smoked
glass) in Settings.

What is behind the glass is a choice, also in Settings:

- **Frosted desktop** (the default in the app's own window): your desktop and the windows behind
  the app show through it, blurred by Windows, so the glass takes its colour from whatever it is
  in front of.
- **Clear desktop**: the same, unblurred. Sharp text behind the app competes with the app's own,
  so the glass is tinted a step more in this mode.
- **Built-in**: the app's own soft backdrop, with nothing showing through. This is what a browser
  window gets, since a browser window cannot be see-through.

The **Tint** slider sets how milky the glass is: clearer shows more of what is behind it,
milkier is easier to read. All of it is remembered.

```powershell
uv run scrabble-ai app --window edge                      # an ordinary Edge app window (built-in backdrop)
uv run scrabble-ai app --install-shortcut --window edge   # make the Desktop shortcut open that one
```

How the see-through part works, for the curious: a browser window cannot be transparent to the
desktop, so the pages are shown in a WebView2 control (the engine of Microsoft Edge) inside a
native window made with [pywebview](https://pywebview.flowrl.com/). The window is switched to
per-pixel transparency through the Windows compositor, and for the frosted look Windows is asked
to blur whatever is behind it (the "acrylic" effect); see `src/scrabble_ai/app/window.py`. If
Windows refuses either, the app says so in Settings and uses its built-in backdrop, and if the
window cannot be made at all the app opens in an Edge app window.

> **The see-through window has not been seen on a screen.** It was built on 2026-11-12 while only
> headless testing was allowed. What has been checked: its prerequisites load on this PC, its
> set-up calls are tested against fakes, the look itself was checked page by page in a headless
> browser with the built-in backdrop, and the frosted look was previewed by compositing the pages
> over a blurred picture of a desktop. Whether Windows really shows and blurs the desktop behind
> the real window is still to be looked at. This note goes when someone has.

Before the first training run the project needs a word list and leave values
(see [TRAINING.md](TRAINING.md)); the app says so at the top if either is missing.

## The pages

### Overview

What is training now (progress against its limit, games per second, its latest result against
the baseline with a small chart), what is queued, the GPU's load and memory, the strongest
network of each run, and the most recent runs and matches. Runs trained from games are shown by
their win rate against the static evaluator, where self-play runs are shown by points per game.

When nothing is training, the top panel is the **latest run**: its verdict in a sentence, its
latest and best evaluations, how long it trained, its strength chart, and **Open all charts**.
The line under the page title says where things stand overall, and it does not call a network
better than the baseline until an evaluation's whole interval is above zero.

### Training

This page is for **self-play** runs: a network that learns from games it plays against itself.
Runs that learn from saved games have their own page, Train from games, below.

**Start training.** Pick a starting point (Default 6x64, Larger 10x128, Wide 12x192, Deep
Q-learning 6x64, or a 30-minute quick test), give the run a name and a time limit, and press
Start. Everything else keeps its default from [configs/train.toml](../configs/train.toml).

**Deep Q-learning 6x64** is the default network with one setting changed, the learning rule
(`train.target = "q"`, "Learning rule" in All settings): each position learns from the move the
network rates best for the opponent, where the default learns from the move that was played.
[DESIGN.md](DESIGN.md) says what that means here and why it may or may not help.

- **Network size** is residual blocks x channels. The sizes that have been benchmarked on this
  PC are one click each, and the line under them says how fast that size trains here and how
  many games the time limit buys. "Measured" and "estimated" mean what they say; a size nobody
  has timed says "not measured".
- **All settings** opens every training setting, grouped as in `train.toml`, each with what it
  does, what turning it down or up does, its default, and a search box. Changed settings are
  marked and can be reset one at a time. A lock marks the settings that cannot change once a run
  has started (the network's shape, the word list, the value scale).
- **Tuning guide** (the button at the top of the page, and the link beside All settings) is every
  setting in one table: what it does, what lower and higher values do, the range worth trying
  and the default, with a filter box. [TUNING.md](TUNING.md) is the same table with advice on
  what to watch and what to try for each kind of trouble.
- **Several runs.** "Add another run" copies the run being edited so one or two settings can be
  changed; the panel on the right lists what differs. Then choose how to run them:
  - **One after another** queues them. One run trains at a time, and a queued run also waits
    for anything already training. The queue carries on with the app closed.
  - **At the same time** starts them together, sharing the GPU.

**Does training several networks at once save time?** Hardly. The GPU is the limit and one run
already keeps it busy, so three runs side by side each go at about a third of the speed.
Measured here with three copies of the benchmark: 4% more self-play and 8% more training in
total than one process alone ([BENCHMARKS.md](BENCHMARKS.md)). One after another gives the first
result three times sooner and lets a later run be dropped if the first one answers the question;
at the same time is worth it only when the runs should see the same wall-clock hours, or when
they are small enough that the GPU is not the limit. GPU memory adds up when runs share it, and
the app warns when the total would not fit.

**History** lists every self-play run with its status, a verdict, a small chart of its strength over the
run, its latest results against both baselines, games, speed and time. The verdict is one of
**Beats the baseline** (the latest evaluation's whole interval is above zero), **Level with the
baseline**, **Behind the baseline**, or **Got worse** (the latest evaluation is significantly
below the run's own best: training made it worse); hover it for the sentence behind it. From a
row: **Charts** (the run's page, also reached by clicking its name), Stop (the run finishes its
step, evaluates, saves a checkpoint and stops), Cancel for a queued run, Resume (for another
stretch of hours, optionally with changed settings), **Rename**, and Delete (the folder and its
checkpoints, after a confirmation). Tick two or more runs and press **Compare** for their
strength curves on one chart, a table of results, and the settings that differ between them.

**Renaming.** Rename is on a run's History row, on its page, and on each of its models on the
Models page. Models are named after the run that trained them, so renaming a run renames all of
its models: `night1 (best)` becomes `sixty-four (best)`. The run's folder is renamed, stars and
notes follow, matches already played show the new name and can be replayed, and games in progress
carry on. A run cannot be renamed while it is running, stopping or queued, or while a match using
one of its models is still running. Names take letters, digits, `-` and `_`.

A result is written as **+4.2 ± 7.8**: the average points per game by which the network beat
that opponent over seat-swapped pairs of games, and the half-width of the 95% interval. A green
triangle means the whole interval is above zero, which is the only thing this app calls a win.

### Train from games

The second way to train ([DESIGN.md](DESIGN.md), "The second way to train"): a network learns
from millions of games the static evaluator played against itself, and is judged by how often it
then beats that evaluator. Two steps, top to bottom on the page.

**Step 1: make a game set.** Give it a name and a number of games (1, 5, 20 or 50 million are one
click each) and press Make game set. The line under the form says how much disk and time that
takes, from the speed of the last set built on this PC: about 286 bytes and a little over a
millisecond a game, so 20 million games are about 5.7 GB and six hours. Building uses every CPU
core and no GPU, and carries on with the app closed. **More options** holds how the first plies
of each game are varied, which is fixed once a set exists.

The table lists every set with its status (Building, Stopping, Complete, Partial, Empty or
Failed), games, size, speed, and what its games look like. From a row: **Stop** (it finishes its
block of 5,000 games; nothing is lost), **Add more games** (a stopped or complete set can be
made larger), **Log**, and Delete, which is refused while a run is training from the set.

**Step 2: train a network.** Choose the set, then the model:

| Model | What it is |
|---|---|
| CNN 6x64 | The convolutional network of the self-play runs, with the pooled head. The fastest |
| CNN with board-wide pooling 6x64 | The same with KataGo's global pooling: every square is told about the whole board |
| Transformer 8x192 | The shape of the strongest published Scrabble value networks. Several times slower; not timed here yet |
| Quick test | A tenth of a pass with short matches, to check that everything works |

Then a name, a time limit (0: until its passes are done) and the number of passes, and Start.
The panel on the right shows what the run will be: positions per pass (one from every game, less
the games held out), steps, and whether it starts now or waits in the queue behind what is
already training. **All settings** has the rest, with the defaults that apply to training from
games (pooled head, win/draw/loss output, helper outputs on, moves chosen by win chance).

**Runs trained from games** lists them with a verdict (Beats the static bot, Level with the
static bot, Behind the static bot, or Got worse), the win rate at each match as a small chart,
the latest win rate with its 95% interval, how far through its passes the run is, and Open,
Stop, Resume, Rename and Delete as on the Training page. A green triangle means the whole
interval is above 50%, which is the only thing the app calls a win.

A network trained here appears on Models, and can be played against, watched and put in the
Arena like any other; it picks its moves by win chance, as it was trained to.

### A run's page

Click a run's name or its Charts button. The six figures at the top are live while it trains.

For a run trained from games the page has the same five tabs in other terms: the main chart is
the **win rate against the static evaluator** with its interval (50% is level) and the spread as
a second chart; "Is it learning?" compares the loss on the batches it trains on with the loss on
held-out games (held-out rising while training falls is memorising), how often it names the
result of a held-out position, and its points error beside leave values alone; then the helper
outputs' losses and weights, the learning rate and positions per second. The rest of this
section describes a self-play run.

- **Progress** opens with **How it went**: the verdict in a sentence, and, when the run got worse
  or is predicting new games worse than leave values alone, a note saying so with a link to the
  tuning guide. Then the charts, in four groups:
  - **Strength against the baselines**: against greedy + leave (the baseline to beat) and against
    greedy, with the 95% interval shaded and a flag at each new best network.
  - **Is it learning?** The network's error on games it has not trained on beside the error of
    leave values alone, and beside its error on the positions it trains on (runs made after
    13 November 2026 record that; seen falling while unseen rises is memorising). The network's
    edge over leave values, which is the earliest sign of learning and should rise above zero.
    Each new network against the best so far, with promotions and any return to the best network
    by the guard marked. Win rates against the bots, scores in evaluation, and how accurate the
    win-probability estimate is.
  - **Training**: value loss, win loss, the learning rate (it steps down each time the guard goes
    back to the best network) and the replay buffer.
  - **Self-play games**: speed, exploration temperature, mean score, margin of victory, how often
    the first player wins, bingos, exchanges and passes, and game length.

  Charts can be drawn against games, hours or steps, with light or heavy smoothing, and every
  chart has a table view.
- **Report.** A plain-language verdict that never claims a win the interval does not support,
  the numbers that matter, and a timeline of what happened: when self-play began, the first
  significant result, each new best network, drops between evaluations larger than noise
  explains, the point where the network began predicting new games worse than leave values
  alone, each time the guard went back to the best network, plateaus, loss spikes, slowdowns,
  record scores, and how each session ended. Click an event to mark it on the chart.
- **Settings.** The run's full configuration, with what differs from the defaults marked.
- **Log.** The console output, followed live.
- **Checkpoints.** Every saved network, each with Play against it, Watch it play, and Run a match.

### Models

Every network the runs have saved, best of each run first, with the size, the games it had
played when saved, and its evaluation from training. Star the ones worth keeping track of and
add notes. **Rename run** on a model's row renames the run it came from, and so every model of
that run (see Renaming, above). A model trained with a different word list or with leave values that have since been
refitted is shown but cannot be played, and says why. Below the table is a small leave-value
explorer: type some tiles to see what keeping them is worth.

The three built-in bots are at the top: **Greedy + leave** (highest score plus the value of the
tiles kept; the baseline), **Greedy** (highest score) and **Random**.

### Play

Choose an opponent, who moves first, and whether hints are on. Then:

- Click a square and type. Click the square again, or press Space, to switch between across and
  down. Squares that already hold a tile are skipped. A letter you do not hold uses a blank if
  you have one (Shift forces a blank). Backspace takes back a tile, Esc clears the placement,
  Enter plays it. Tiles can also be dragged from the rack, and the rack reordered or shuffled.
- A move that is not legal is refused with the reason (not a word, not connected, and so on) and
  your tiles stay where you put them.
- **Hint** lists the best moves as the advisor ranks them, with each one's score, the tiles it
  keeps, what the advisor thinks the position is worth afterwards, and for a network its win
  estimate. Hover to preview a move on the board, click to place it.
- After each of the opponent's moves, **What it considered** shows the same table for the move
  it chose and the ones it passed over.
- Exchange and Pass are buttons. The side panel has the scores, the move list (click a move to
  see where it was played) and the tiles you have not seen.

### Watch

Two players, a board, both racks, at a speed you can follow: Slow, Normal, Fast or Instant, with
Pause and Step. Beside the board is what the player to move considered on each turn. Click any
move in the list (or use the arrow keys) to see the board at that point; Esc returns to the live
game. The second tab plots the score over the game. A game is fixed by its two players and its
seed, so Restart replays it exactly, and any game from a match can be opened here.

### Arena

Fast matches without a board: pick two players and a number of seed pairs. Each seed is played
twice with the seats swapped, which cancels most of the luck of the draw. Bots play about a
thousand games every two seconds; networks take longer, in line with their size.

A match's page leads with the spread per game, its interval and a one-line verdict, then win
rate, mean scores, bingos and exchanges per game; the distribution of spreads and of final
scores; how the estimate settled as games accumulated; the advantage of moving first; highlights
(highest-scoring play, biggest win, closest game, biggest swing between the two games of a pair,
longest game); and a table of every game. Any game has a Replay button that opens it in Watch.

### Settings

The theme (light or dark), what is behind the glass and how tinted it is (see "The look" above),
and a few facts about what the app is running on: its version, the word list, whether leave
values are fitted, and the GPU.

## Where things are kept

| | |
|---|---|
| `runs/<name>/` | One folder per training run of either kind: checkpoints, metrics, console log, settings |
| `data/games/<name>/` | One folder per game set: the games, their index, and what the set was made with |
| `data/app/matches/` | One folder per match |
| `data/app/library.json` | Stars and notes |
| `data/app/app.log` | The app's own errors, if any |

None of it is in the repository. Deleting a run or a match in the app deletes its folder.

## Stopping things

- **A training run:** Stop on its row or its page. It saves before stopping and can be resumed.
  From a terminal, `New-Item runs\<name>\STOP` does the same.
- **A queue:** Cancel the queued runs, or stop the running one; the queue then starts the next.
- **A game set being built:** Stop on its row; it keeps the games made so far and can be added to
  later. From a terminal, `New-Item data\games\<name>\STOP` does the same.
- **A match:** Cancel on its page.
- **The app:** close its window.

## For development

The pages are plain JavaScript, CSS and SVG in `src/scrabble_ai/app/static/` with no build step,
and the server is the Python standard library; [APP_API.md](APP_API.md) is the contract between
them. `uv run scrabble-ai app --no-window --port 8700` serves without opening a window, and
adding `?mock=1` to the address runs the pages on invented data with no server behind them
(`?mock=1&glass=1` makes the page behave as in its own window, with its background left clear).
`--runs-dir` and `--data-dir` point the app at other folders. Tests: `uv run pytest
tests/test_app_*.py tests/test_insights.py`.
