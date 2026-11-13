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
  `configs/train.toml` must list every setting with its default (a test checks this).
- `legacy/` is reference only. Do not import from it, fix it or extend it.
- The desktop app is `src/scrabble_ai/app/` (`uv run scrabble-ai app`; docs in `docs/APP.md`,
  the page/server contract in `docs/APP_API.md`). The server is standard-library Python bound to
  127.0.0.1; the pages in `static/` are plain JS/CSS/SVG with no build step and no CDN. To work
  on it without touching real runs: `--no-window --port <n> --runs-dir <folder> --data-dir
  <folder>`, and `?mock=1` in the address runs the pages on invented data. Check pages in a real
  browser before calling UI work done. It must stay local: never host it, and no sign-in pages.
- The essential code (engine, move generation, rules, RL formulation, training loop) is written by
  the lead model; well-specified side tasks go to Opus subagents.
- The default lexicon is `data/lexicon/NSWL2023.txt`, the NASPA *school* list (195,747 words). A
  model is only valid for the lexicon and leave values it was trained with; checkpoints record both.
