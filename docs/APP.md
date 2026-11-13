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

### The glass window

On Windows the window is see-through: the desktop and whatever is open behind the app show
through it. Not everything is glass:

- the **page between panels** is nearly clear;
- **panels** (the sidebar, the top bar, cards, page headings and toolbars) are translucent, each
  with a solid outline;
- everything to **read or use** is solid: text, buttons, inputs, menus, dialogs, the board, its
  tiles and the racks. The title bar and window frame are the ordinary Windows ones.

The **See-through** slider at the foot of the sidebar sets how clear the glass is, from solid
(the ordinary look) to almost all the way through; the app remembers it. Clear glass over a busy
window is harder to read than over a plain one, so slide it down when it gets in the way.

```powershell
uv run scrabble-ai app --window edge                      # an ordinary, opaque Edge app window instead
uv run scrabble-ai app --install-shortcut --window edge   # make the Desktop shortcut open that one
```

How it works, for the curious: a browser window cannot be transparent to the desktop, so the
pages are shown in a WebView2 control (the engine of Microsoft Edge) inside a native window made
with [pywebview](https://pywebview.flowrl.com/), and the window is switched to per-pixel
transparency through the Windows compositor (`src/scrabble_ai/app/window.py`). If that window
cannot be made, the app opens in an Edge app window as before, and if the transparency itself
cannot be set up the window is simply opaque.

> **Not yet seen on a screen.** The glass window was built on 2026-11-12 while only headless
> testing was allowed, so what has been checked is everything short of looking at it: its
> prerequisites load on this PC, its set-up calls are tested against fakes, and the glass styles
> were previewed by compositing the pages over a picture of a desktop. This note goes when
> someone has looked at the real thing.

Before the first training run the project needs a word list and leave values
(see [TRAINING.md](TRAINING.md)); the app says so at the top if either is missing.

## The pages

### Overview

What is training now (progress against its limit, games per second, its latest result against
the baseline with a small chart), what is queued, the GPU's load and memory, the strongest
network of each run, and the most recent runs and matches.

### Training

**Start training.** Pick a starting point (Default 6x64, Larger 10x128, Wide 12x192, or a
30-minute quick test), give the run a name and a time limit, and press Start. Everything else
keeps its default from [configs/train.toml](../configs/train.toml).

- **Network size** is residual blocks x channels. The sizes that have been benchmarked on this
  PC are one click each, and the line under them says how fast that size trains here and how
  many games the time limit buys. "Measured" and "estimated" mean what they say; a size nobody
  has timed says "not measured".
- **All settings** opens every training setting, grouped as in `train.toml`, each with what it
  does, its default, and a search box. Changed settings are marked and can be reset one at a
  time. A lock marks the settings that cannot change once a run has started (the network's
  shape, the word list, the value scale).
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

**History** lists every run with its status, size, games, speed and latest results against both
baselines. From a row: Stop (the run finishes its step, evaluates, saves a checkpoint and
stops), Cancel for a queued run, Resume (for another stretch of hours, optionally with changed
settings), and Delete (the folder and its checkpoints, after a confirmation). Tick two or more
runs and press **Compare** for their strength curves on one chart, a table of results, and the
settings that differ between them.

A result is written as **+4.2 ± 7.8**: the average points per game by which the network beat
that opponent over seat-swapped pairs of games, and the half-width of the 95% interval. A green
triangle means the whole interval is above zero, which is the only thing this app calls a win.

### A run's page

Click a run's name. The six figures at the top are live while it trains.

- **Progress.** Strength against greedy + leave (the baseline to beat) and against greedy, with
  the interval shaded and a flag at each new best network; how well the network predicts games
  it has not trained on, next to the leave values alone (the earliest sign of learning, long
  before the evaluations separate from zero); then loss, speed, exploration temperature, scores,
  bingos, game length and replay size. Charts can be drawn against games, hours or steps, and
  every chart has a table view.
- **Report.** A plain-language verdict that never claims a win the interval does not support,
  the numbers that matter, and a timeline of what happened: when self-play began, the first
  significant result, each new best network, drops between evaluations larger than noise
  explains, plateaus, loss spikes, slowdowns, record scores, and how each session ended. Click
  an event to mark it on the chart.
- **Settings.** The run's full configuration, with what differs from the defaults marked.
- **Log.** The console output, followed live.
- **Checkpoints.** Every saved network, each with Play against it, Watch it play, and Run a match.

### Models

Every network the runs have saved, best of each run first, with the size, the games it had
played when saved, and its evaluation from training. Star the ones worth keeping track of and
add notes. A model trained with a different word list or with leave values that have since been
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

## Where things are kept

| | |
|---|---|
| `runs/<name>/` | One folder per training run: checkpoints, metrics, console log, settings |
| `data/app/matches/` | One folder per match |
| `data/app/library.json` | Stars and notes |
| `data/app/app.log` | The app's own errors, if any |

None of it is in the repository. Deleting a run or a match in the app deletes its folder.

## Stopping things

- **A training run:** Stop on its row or its page. It saves before stopping and can be resumed.
  From a terminal, `New-Item runs\<name>\STOP` does the same.
- **A queue:** Cancel the queued runs, or stop the running one; the queue then starts the next.
- **A match:** Cancel on its page.
- **The app:** close its window.

## For development

The pages are plain JavaScript, CSS and SVG in `src/scrabble_ai/app/static/` with no build step,
and the server is the Python standard library; [APP_API.md](APP_API.md) is the contract between
them. `uv run scrabble-ai app --no-window --port 8700` serves without opening a window, and
adding `?mock=1` to the address runs the pages on invented data with no server behind them
(`?mock=1&glass=1` adds the glass styles, drawn over the browser's own white).
`--runs-dir` and `--data-dir` point the app at other folders. Tests: `uv run pytest
tests/test_app_*.py tests/test_insights.py`.
