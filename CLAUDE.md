# scrabble-engine: project instructions

A Scrabble AI trained locally: a Rust engine (`crates/`) and a PyTorch afterstate value network
(`src/scrabble_ai/`). Start with `docs/PLAN.md` for where the project stands, `docs/DESIGN.md` for
how it works, `docs/TRAINING.md` for running it and `docs/legacy-audit.md` for what was wrong with
the old code. Machine-specific notes, if any, are in `CLAUDE.local.md` (not committed).

## Hard rules

- **This repo is public** (github.com/9MNickHazard/scrabble_engine). Push source code only. Never
  push binaries, wheels, model weights or word lists; never create releases, tags with assets,
  gists or Pages sites; never force-push. Commit with the GitHub no-reply identity already
  configured in this clone, not a personal email address.
- **Never commit:** word lists (copyrighted NASPA/Collins lists), model weights (`*.h5`, `*.pt`,
  `*.safetensors`), pickles, `runs/`, `data/`, virtual environments, Rust `target/`, compiled
  extensions, keys or credentials. Run `git status` before every commit and check nothing like
  that is staged.
- **Measured numbers only:** every speed or strength claim in docs or reports needs a benchmark or
  evaluation behind it, with the command to reproduce. Label estimates as estimates. If a test
  fails or a step was skipped, say so.
- **No stray processes:** do not leave training or benchmark processes running at the end of a
  task unless asked for a long run. Say how to start, stop and resume them.
- **Training runs are the owner's to start.** Short smoke tests and dry runs to prove the code
  works are fine; a real training run is not, unless asked.

## Working here

- Python 3.12 through `uv` (`uv sync`, `uv run pytest`, `uv run scrabble-ai <command>`). Not 3.14.
- `uv sync` builds the Rust extension with maturin and rebuilds it when files under `crates/`
  change. On Windows a rebuild fails while another Python process has the extension loaded, so
  finish or stop running jobs before changing Rust code.
- Rust tests: `cargo test -p scrabble-core` (quick) and the long acceptance runs with
  `-- --ignored`. `cargo test` at the root leaves out the Python extension crate on purpose.
- The GPU is an RTX 5070 Ti (Blackwell, sm_120): PyTorch comes from the `cu132` index pinned in
  `pyproject.toml`. `uv run scrabble-ai gpu-check` must pass after any PyTorch change.
- Config is TOML read with the standard library into dataclasses (`src/scrabble_ai/config.py`);
  `configs/train.toml` must list every setting with its default (a test checks this). Every
  setting also needs its entry in `src/scrabble_ai/app/schema.py` (label, what it does, what
  lower and higher do, the range worth trying); `docs/TUNING.md` carries the tables that file
  generates (`python -m scrabble_ai.app.schema --write`), and a test fails when they differ.
- The first real runs (2026-11-13) memorised their own games and got worse; `docs/PLAN.md` has
  the numbers and the changed defaults (replay ratio 1, TD(lambda) 0.7, the guard). Those
  defaults are untested by a full run: do not describe them as working until one says so.
- Network inputs are groups in `src/scrabble_ai/features.py` (`FeatureSet`), each switched by a
  `[model]` setting. To add one: append it after the existing groups (never in between: a saved
  network must keep seeing its own layout), give it a setting, and list that setting in
  `BEFORE_IT_EXISTED` in `config.py` with the value meaning "not there", so that checkpoints
  written before it exist load as not having it. `tests/test_inputs.py` shows what to test.
  Anything worked out from the board alone (cross masks, openness) is recomputed by the engine
  for candidates and replayed positions alike and is not stored in the replay buffer.
- Run one `pytest` session at a time. Two at once delete each other's temporary folders (give
  each its own `--basetemp` if two must run).
- To keep a test run off the GPU, set `CUDA_VISIBLE_DEVICES=-1`. An empty value does not hide it
  on Windows (and PowerShell 5.1 deletes a variable set to `''`). Hidden, the suite skips its
  four `cuda_only` tests.
- There are two trainers: `train.py` (self-play; `train.target` is `td` or `q`) and `offline.py`
  (from a game set made by `games.py`; settings in `[offline]`, and `OFFLINE_DEFAULTS` in
  `config.py` for what it changes elsewhere). Runs of both kinds live in `runs/` and are told
  apart by `offline.games` being set. A run trained from games is judged by win rate against the
  static evaluator, never by spread. Game sets live in `data/games/` (not committed).
- `legacy/` is reference only. Do not import from it, fix it or extend it.
- The desktop app is `src/scrabble_ai/app/` (`uv run scrabble-ai app`; docs in `docs/APP.md`,
  the page/server contract in `docs/APP_API.md`). The server is standard-library Python bound to
  127.0.0.1; the pages in `static/` are plain JS/CSS/SVG with no build step and no CDN. To work
  on it without touching real runs: `--no-window --port <n> --runs-dir <folder> --data-dir
  <folder>`, and `?mock=1` in the address runs the pages on invented data. Check pages in a real
  browser before calling UI work done. It must stay local: never host it, and no sign-in pages.
  The look is "liquid glass" (`static/css/tokens.css` and `liquid.css`): frosted translucent
  panes with black outlines, light by default, dark in Settings. On Windows it opens in a native
  see-through window (`app/window.py`, pywebview + WebView2) that shows the desktop behind the
  glass, frosted or clear; `--window edge` is the browser-window fallback, which draws its own
  backdrop. The see-through part can only be judged on a real screen.
- The essential code (engine, move generation, rules, RL formulation, training loop) is written by
  the lead model; well-specified side tasks go to Opus subagents.
- The default lexicon is `data/lexicon/NSWL2023.txt`, the NASPA *school* list (195,747 words). A
  model is only valid for the lexicon and leave values it was trained with; checkpoints record both.
