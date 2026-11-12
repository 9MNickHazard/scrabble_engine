# Legacy code audit

Audited 2026-11-12. The code audited is in `legacy/`: the newest snapshot of each file from the old
OneDrive project, byte-identical to the `(2025_09_13 23_08_50 UTC)` copies (hashes in
`legacy/PROVENANCE.md`). Line numbers below refer to those files.

**Bottom line.** The old project could not train or play in the state it was left in, and fixing
the crashes would not have rescued it. The learning target has the wrong sign for a two-player
game, the win/loss reward never reaches the learner, the network can only reweight six hand-made
heuristics, the unseen-tile input is wrong, and the game engine accepts illegal moves and
mis-scores blanks. The move generator is exhaustive search that takes over a second per position
(35 seconds with a blank on the rack) and still misses 8.6% of the legal placements. Nothing here is
worth porting; the rewrite in [PLAN.md](PLAN.md) starts from the rules.

Most of the findings in the handoff brief are confirmed. These are the ones that changed when
checked:

| Brief said | What is actually true |
|---|---|
| `import game_api` only works on case-insensitive filesystems | It fails on Windows too: Python checks the file name's case. The local copy is named `game_API.py`; the copy on GitHub is `game_api.py`, where the import works. |
| The player who did not make the last move never gets the win/loss signal | Nobody ever gets it. The win/loss reward is unreachable code and no stored transition is ever terminal. |
| BatchNorm behaves badly with RL targets | The BatchNorm layers never run in training mode here, so they are just a scale and shift. Pointless rather than harmful. |
| The GUIs use a different move generator from training | They import the Python generator but never call it. All moves come from the Rust generator. |
| `ScrabbleDQN()` defaults to epsilon 1.0, so exploration crashes | Every caller sets epsilon to 0 first. The greedy path crashes instead, for the same missing argument. |
| `position_bonus` is always 0 | It is 0 unless the word covers a triple-letter square. Word premiums still never count. |
| The defensive heuristic is inverted for squares above and to the left | It is inverted for all four neighbours. |
| The old log shows AI vs AI games scoring 236 to 351 | The log holds two distinct games (one recorded twice), from unknown weights. |
| The Rust generator's prefix pruning drops legal moves | It does, but a second bug (a cap on tiles played) causes 98.6% of the misses. |
| `main.rs` fails on private imports | It fails earlier: a `cdylib`-only crate cannot be linked by a binary at all. |

How each finding was checked is marked **[run]** (reproduced by running code; commands are in the
last section) or **[read]** (established by reading the code).

## 1. Crashes and broken wiring

| Finding | Status | Evidence |
|---|---|---|
| Invalid format spec kills training at the first progress print | Confirmed **[run]** | `self_play.py:198` and `:200` put a conditional inside the format spec. Python raises `ValueError: Invalid format specifier '.4f if stats else 'N/A'' for object of type 'float'`. The print runs when `(episode + 1) % 5 == 0` (`:189`), so at the end of the fifth episode. |
| `import game_api` but the file is `game_API.py` | Modified **[run]** | `dqn_main.py:7`, both GUIs (`:9-10`) and `dqn_evaluator.py:7`. On Windows with Python 3.12 this raises `ModuleNotFoundError: No module named 'game_api'`: CPython enforces the case of module file names even on a case-insensitive disk. With the local files named as they are, `dqn_main.py train` fails on its first import. The same file in the GitHub repo is named `game_api.py` (identical contents), so the import works there; the mismatch is in the local copy only. |
| `global pd` import hack | Confirmed **[read]** | `dqn_main.py:117-118` imports pandas inside `main()` and makes it global so `train_model` (`:80`) can see it. It works, by accident of call order. |
| `normalize_strategic_value` is missing `self` | Confirmed **[read]** | `dqn_model.py:383`. Nothing calls it, so it never fails. |
| GUIs and evaluator call `act()` with the old signature | Confirmed **[run]** | Six call sites (`ai_vs_ai_gui.py:505`, `human_vs_ai_gui.py:428`, `dqn_evaluator.py:68`, `:231`, `:364`, `:632`) pass `(state, moves, reward_calculator, game, player)` to `act(state, moves, game, player, episode)` (`dqn_model.py:114`). The reward calculator lands in the game slot and the first thing to fail is `dqn_model.py:412`: `AttributeError: 'ScrabbleDQNReward' object has no attribute 'player_2_hand'`. The AI-vs-AI GUI catches the error and passes instead, so every turn is a pass and the game ends after two of them with both scores negative. If the arguments were fixed, the next failure is that `act()` now returns a `(move, features)` tuple and the callers index it as a move. |
| `ScrabbleDQN()` built with no reward calculator; exploration crashes | Modified **[run]** | The constructor default is `reward_calculator_ref=None` (`dqn_model.py:13`), and the GUIs build agents that way (`ai_vs_ai_gui.py:417`, `:422`; `human_vs_ai_gui.py:395`). But every caller then sets `epsilon = 0`, so the exploration branch is never taken. The greedy branch needs the reward calculator as well (`dqn_model.py:474`, `:485`) and raises on `None` there. |
| GUIs use a different move generator from training | **Refuted** **[read]** | Both GUIs import `ScrabbleAllValidMoves` from `valid_moves_script.py` (line 8) and never instantiate it. Every move list in the GUIs and the evaluator comes from `game.get_all_valid_moves()`, which calls the Rust generator (`game_API.py:690`). The only effect of the import is a second dictionary load. So training and play used the same generator, and `valid_moves_script.py` is dead code. |
| Other reasons the GUIs and evaluator cannot run | New **[run]** / **[read]** | `dqn_evaluator.py:90` and `:404` call `estimate_move_quality` with one argument too many (`TypeError`). `word_dictionary.csv` and the Rust extension must be in the current directory. The evaluator ignores `play_move`'s return value at eight call sites, so a move the validator rejects would be retried forever. |

In the local copy two independent blockers stand between `python dqn_main.py train` and a first
progress line: the import, and then the format spec at the end of the fifth game. The format-spec
crash is in the newest `self_play.py`, which was never pushed to GitHub; the older version there
does not have it.

## 2. RL formulation

| Finding | Status | Evidence |
|---|---|---|
| Q is a linear mix of 6 hand-made features | Confirmed **[read]** | The network's output is 7 numbers (`dqn_model.py:82`). `Q(s,a) = W(s)[:6] . f(a) + W(s)[6]` (`:186`, `:247`), where `f(a)` is score/25, tiles/7 and four heuristic values (`:504-523`). The network never sees which squares or letters a move uses. The bias `W(s)[6]` is the same for every move in a position, so it cannot change which move is chosen. |
| Two-player sign and perspective error | Confirmed **[read]** | After the mover plays, the next state is encoded from the *mover's* side (`self_play.py:108`), the legal moves listed are the *opponent's* (`:112-135`), and the target is `r + gamma * max` over those (`dqn_model.py:310-338`). The opponent's best reply is added to the mover's value when it should count against it. |
| The target uses hidden information | New **[read]** | Those opponent move features are computed from the opponent's actual rack (`self_play.py:115`, `:128-134`, via `dqn_model.py:412`). The mover could never know it. |
| Rewards do not match the objective | Confirmed **[run]** | `calculate_reward` (`dqn_reward_functions.py:18-109`) adds tile-usage, leave, setup, defensive and endgame terms to the move score (`:60-99`). The same heuristics are the action features, so the network is trained to predict its own inputs. Default gamma is 0.95 (`dqn_model.py:11`, `dqn_main.py:98`). |
| Bingo bonus counted twice | Confirmed **[run]** | The move score already includes the 50 (`lib.rs:807-810`; `game_API.py:368-370`), and `dqn_reward_functions.py:62-63` adds another 50. Measured: a 66-point bingo (16 + 50) gives a reward of 116 with the other terms zeroed. |
| Passes are never stored; the non-final player gets no win/loss signal | **Modified** **[run]** | Passes are not stored (`self_play.py:65-73`, `:82-87`). But the larger problem is that the win/loss reward never fires for anyone. It is gated on `next_game_state.game_over` (`dqn_reward_functions.py:102`). `game_over` is only set in `end_game()` (`game_API.py:655`), which is only called from `pass_turn()` (`:608-609`), and rewards are only computed after a successful `play_move()`. So the outcome reward is dead code, `done` is never true in replay memory (`self_play.py:105`, `:150-154`), and `--win-reward` and `--loss-penalty` have no effect. The same gate disables the play-out bonus in the endgame reward (`:845`). |
| Exploration is not exploration | Confirmed **[read]** | The exploration branch picks at random from the top 20% of moves ranked by a heuristic (`dqn_model.py:131-144`). Epsilon is multiplied by 0.995 on every `replay()` call (`:363-364`), which happens once per move (`self_play.py:156`). It reaches the 0.01 floor after 919 calls. With the default single shared agent (`dqn_main.py:55-56`) that is about 980 moves into training. |
| BatchNorm and MaxPool in a value network | **Modified** **[read]** | The 2x2 MaxPool is there (`dqn_model.py:58`) and does discard position. The seven BatchNorm layers (`:55`, `:57`, `:60`, `:65`, `:69`, `:73`, `:79`) are never run in training mode: every forward pass, including the one inside the gradient tape, passes `training=False` (`:165`, `:240`, `:311`). Their running statistics stay at the initial mean 0 and variance 1, so each layer is only a learnable scale and shift. |
| Unbounded per-candidate logging | Confirmed **[read]** | From episode 50 on (`dqn_model.py:40`), the greedy branch appends a 19-field record for every candidate move to an in-memory list (`:189-219`) that is only written out when training finishes (`dqn_main.py:78-82`). The `move_number` field actually holds the episode number (`:194`). The archived `move_evaluation_log.csv` is 10.6 MB. |
| Training cost per move | New **[read]** | `replay()` computes targets with a Python loop over the batch, one TensorFlow call and `.numpy()` per sample (`dqn_model.py:316-335`). Each stored transition carries the feature list for every legal reply (`:112`), typically hundreds of rows. |

## 3. State encoding

| Finding | Status | Evidence |
|---|---|---|
| The unseen-tile vector is wrong | Confirmed **[read]** | `encode_unseen_tiles` starts from `game_state.letter_counts` (`dqn_state.py:82`), which is the bag: `draw_tiles` decrements it as tiles are drawn (`game_API.py:119-132`). It then subtracts the board tiles (`:85-88`) and the mover's rack (`:90-93`) again. The result is "bag minus board minus my rack", clamped at zero, instead of "bag plus opponent's rack". A blank on the board is subtracted as its assigned letter, because the board stores only the letter. |
| Game features are raw scores | Confirmed **[read]** | `dqn_state.py:121-127`: scores, bag count and spread go in unnormalized. |
| Unknown letters map to A | New **[read]** | `LETTER_TO_INDEX.get(letter, 0)` at `dqn_state.py:37`, `:41`, `:66`. |

## 4. Heuristics in `dqn_reward_functions.py`

These fed both the reward and the action features. None is carried into the new design.

| Finding | Status | Evidence |
|---|---|---|
| `position_bonus` starts at 0 and is only multiplied | Modified **[run]** | `:401-416`. Triple-letter squares add 2; double- and triple-word squares multiply whatever has accumulated. So word premiums count only if a triple-letter square comes earlier in the word. Measured on 7-tile plays: a1-a7 (TW) 0, g1-g7 (DL) 0, b2-b8 (DW then TL) 2, b8-b14 (TL then DW) 3. |
| `one_point_tiles` is missing `u` | Confirmed **[read]** | `:377`. The two other copies of the set (`:214`, `:318`) include it. |
| Defensive value: first/second letter inverted above and left | Modified **[run]** | `:761-770`. All four branches are inverted, because the comments have the geometry backwards and the code follows the comments. Example: Q placed with an empty double-word square *below* it lets the opponent make QI, but the code scores it as fully blocked (6.0 instead of 2.0), and scores the square *above* a Q (where nothing fits) as open. |
| Two-letter words hard-coded | Confirmed **[run]** | `:696-704` lists 106 words. The lexicon has 107; `na` is missing. |
| Defensive value credits squares the move itself covers | New **[run]** | Emptiness is tested on the board before the move (`:753-754`). An opening CAT at h7-h9 earns 6.0 for "defending" h8, which it is sitting on. Unseen blanks are ignored when counting what the opponent could play (`:776`). |
| `score_efficiency` can be unbound | Confirmed, latent **[run]** | `:877-883` raises `UnboundLocalError` when the bag has 6 or fewer tiles and the move places no tiles. No current caller reaches it. |
| Leave calculations ignore the blank assignment | New **[run]** | `:186-189`, `:435-439`, `:581-584`. When a blank stands for a letter that is also on the rack, the real letter is removed and a blank that was actually played is kept. Rack `E?ARTSN`, ATE with the blank as E: leave valued at 9.12 instead of 8.12. |
| Leave term switched off too broadly | New **[read]** | `:78` and `dqn_model.py:473` skip the leave term whenever the *best available* move scores 56 or more, whatever move is being evaluated. The comment says 56 is the lowest possible bingo; two blanks and five one-point tiles make 55. |

## 5. Game rules and scoring (`game_API.py`)

All of these were reproduced by running the old code: `tools/legacy_audit/repro_game_rules.py`.

| Finding | Status | Evidence |
|---|---|---|
| The game does not end when a player goes out | Confirmed **[run]** | `play_move` (`:504-588`) never calls `end_game()`. After playing the last tiles with the bag empty: `game_over=False`, opponent to move. The opponent keeps playing until two passes in a row. The archived `debug_testing\game_API.py:585-586` has the missing call; it was never merged. |
| Blanks already on the board score at face value | Confirmed **[run]** | `point_calculation` only zeroes blanks declared in the current move (`:335-341`), and cross-word scoring never looks at the board's blanks (`:479-498`). A blank Q scores 0 when played, then 10 for every later word through it (measured: QI re-scored as 11 instead of 1). The board sent to Rust carries letters only (`:684-687`). |
| The main word is not extended through adjacent tiles | Confirmed **[run]** | Only the word as given is checked (`:237`). CAT at h8, then OX at h11: accepted for 17 points, leaving CATOX on the board. |
| A move need not place a tile | Confirmed **[run]** | Replaying CAT on its own squares is accepted, scores 5 and passes the turn (`:567-572`). |
| A blank can be used without being declared | Confirmed **[run]** | `validate_move` falls back to a blank silently (`:193-196`). `play_move` scores the move first (`:516`) and records the blank afterwards (`:551-557`). QI with a blank Q and no declaration scores 22 instead of 2. |
| Exchange rule | Confirmed **[run]** | Allowed whenever the bag holds at least as many tiles as are exchanged (`:622-624`); NASPA requires 7 in the bag. The tiles also go back into the bag *before* the replacements are drawn (`:635-639`), so they can be drawn straight back. The agent never exchanges: `self_play.py` only plays or passes. |
| Two passes end the game | Confirmed **[read]** | `:608`. NASPA: six consecutive scoreless turns. |
| Dictionary re-read for every game; whole game deep-copied every move | Confirmed **[run]** | `Scrabble_Game()` reads the CSV each time (`:56-82`): 92 ms. `self_play.py:61` deep-copies the game, word set included, on every move: 117 to 148 ms per move in two runs. The word set has 144,375 entries, not about 280,000. `validate_move` also deep-copies the board on every call (`:240`), and `draw_tiles` rebuilds the bag as a list for every tile drawn (`:123-127`). |
| Rust errors are swallowed | New **[read]** | Any exception from the generator is printed and replaced with an empty move list (`:689-695`), which the caller treats as "no moves, pass". |

## 6. Rust move generator (`scrabble_valid_moves/src/lib.rs`)

An Opus subagent built the crate as shipped and compared it with an independent brute-force
reference on 61 positions (8 hand-made, 53 from random games), using the same 144,375-word list the
old code loaded. Scripts and exact commands are in `tools/legacy_audit/`.

| Finding | Status | Evidence |
|---|---|---|
| `main.rs` does not build | Modified **[run]** | `cargo build --release --locked` fails, but not on privacy first: `error[E0432]: unresolved import 'scrabble_valid_moves_rust'` at `main.rs:1`, because a crate built only as a `cdylib` cannot be linked by a binary at all. With `rlib` added in a probe copy, the privacy errors the brief expected appear (`ScrabbleSolver`, `create_empty_board` and `DICTIONARY` are private). The library alone builds and imports under Python 3.12 unchanged. |
| PyO3 0.21 deprecated API | Confirmed **[run]** | 12 warnings on the library build, the GIL-ref deprecations among them (`PyList::empty`, `PyDict::new`, `&PyDict` arguments at `lib.rs:849`, `:884-897`, `:911`). |
| Brute force, not a GADDAG or DAWG | Confirmed **[run]** | Four directions (`:219`), every permutation of every rack subset (`:223`), 26 letters per blank by cartesian product (`:227-235`), every anchor offset (`:256`). The dictionary is a `HashSet<String>` of words plus one of 295,417 prefixes. One blank multiplies the time by a median of 24; a second by 22 to 38. |
| Prefix pruning loses legal moves | Modified **[run]** | It does (ADD + D,S for ADDS is never found: the check sees "ds", which starts no word, `:389-392`). But it is the smaller of two causes. See below. |
| A cap on tiles played loses far more | New **[run]** | `:222` limits a play to as many tiles as there are squares *after* the anchor (`get_max_space`, `:276-283`), wherever the anchor sits in the word. A play can therefore almost never reach the last column or the bottom row. Example: PLEA on the board, rack SANTDEU: PLEASANT to the edge is not generated. |
| Dedup keeps the first blank assignment | Confirmed **[run]** | `:488-500` and `:829-839`. Of 10,418 words with more than one legal blank assignment, 625 (6.0%) came back with a lower-scoring one; the worst cost 12 points. Which one is kept changes from call to call, because anchors come out of a `HashSet` and the sort is unstable (`:841`): eight calls on one position gave eight different move lists. |
| Blanks on the board score at face value | Confirmed **[run]** | Only blanks placed by the current move score 0 (`:760-761`, `:787-788`). All 220 score disagreements with the reference (0.28% of moves) are this bug; there is no other scoring difference. |
| Relative dictionary path, string-keyed board | Confirmed **[run]** | `Path::new("word_dictionary.csv")` (`:63`). A failed load is cached for the life of the process, so changing directory afterwards does not help. Boards are `HashMap<String, Option<char>>` and every square lookup allocates a `String` (`:131-138`). |
| Lone tiles forming two words are returned twice | New **[run]** | Once as "right" and once as "down": 484 duplicate entries in the 61 positions. |
| The GIL is held for the whole call | New **[read]** | No `allow_threads` anywhere, so a 35-second call blocks every other Python thread. |

**Completeness** (placements, ignoring which tiles are blanks):

| Positions | Legal placements (reference) | Returned by the old generator | Missed | Illegal moves returned |
|---|---|---|---|---|
| 61 | 84,197 | 76,963 | 7,234 (8.6%) | 0 |

Every miss was assigned a cause by re-running it on diagnostic builds with one bug removed at a
time: removing the tile cap alone recovers 98.6% of the misses, and removing the prefix check alone
recovers 17.8%. With both removed the old generator finds exactly the reference's set on the six
positions that were cross-checked, so these two are its only completeness bugs. Four in five of
the missed plays put a tile in the last column or the bottom row. Counting blank choices as
separate moves, 25,148 of 102,111 legal moves are missing.

**Speed** (wall time of one `get_valid_moves_rs` call, 7-tile racks on the same 30 mid-game boards,
release build, rayon on all 16 threads):

| Rack | Boards | Median | Fastest | Slowest |
|---|---|---|---|---|
| No blank | 30 (3 runs each) | 1.43 s | 0.95 s | 2.69 s |
| One blank | 30 | 35.3 s | 24.8 s | 62.7 s |
| Two blanks | 3 | 1,059 s | 760 s | 1,395 s |

The machine was not idle during these runs (other programs were using 13% to 24% of the CPU, more
during the two-blank run, whose slowest figure is inflated), and two-blank racks were only timed on
three boards because each takes about 18 minutes. Treat them as the right order of magnitude. For
comparison the new generator averages about 170 microseconds per turn on one thread across whole
games, blanks included (`cargo test -p scrabble-core --test real_lexicon -- --nocapture`).

## 7. Lexicon

| Finding | Status | Evidence |
|---|---|---|
| `word_dictionary.csv` is cut off at 10 letters | Confirmed **[run]** | 144,375 words, longest 10 letters. `word_dictionary_ALL.csv` has 195,744 words up to 15 letters; 51,369 of them (11 to 15 letters) are missing from the truncated file. Both the Python engine and the Rust generator loaded the truncated one. |
| The CSVs dropped three valid words | New **[run]** | `NSWL2023.txt` has 195,747 words; the "ALL" CSV has 195,744. The three missing are NA, NAN and NULL, which is what pandas turns into missing values when a CSV is read and written back. The new loader reads plain text with no type inference and has a test for these three. |
| `NSWL2023.txt` is the school list | New **[run]** | The file is the NASPA *School* Word List 2023: 195,747 words of 2 to 15 letters, 107 of them two letters long. Offensive words are removed (SHIT and FART are absent). The tournament list, NWL2023, is a different and slightly larger file. See change 6 in the plan. |

## 8. Results so far

| Finding | Status | Evidence |
|---|---|---|
| `logs/game_series_results.csv`: AI vs AI games scored 236 to 351 | Modified **[run]** | The file has three rows but only two distinct games: 260-278 (recorded twice) and 236-351, both won by the second player. The duplicate comes from the AI-vs-AI GUI calling its game-over handler twice when a game ends on a pass (`ai_vs_ai_gui.py:541-542` and `:559-560`), which also means a "series of N" plays about N/2 distinct games and double-counts them in its averages. The file is dated 2025-05-16, before the last edits to the model and GUI, and does not record which weights played. |

Two games from unknown weights, on a 10-letter lexicon, under rules that let the opponent keep
playing after a play-out, are not a usable baseline. Phase 2 measures greedy play properly and
reports its mean score next to these.

## What this means for the rewrite

- **Engine:** start from a written rules document and prove the generator against an independent
  reference. Every rules bug in section 5 becomes a regression test.
- **Learning target:** afterstate values from the mover's side, with the opponent's outcome
  subtracted, built only from what the mover can see. Terminal results must reach both players;
  there is a unit test for exactly the cases the old code got wrong.
- **Reward:** the spread and nothing else. No shaping terms, no heuristic features.
- **Inputs:** the unseen pool is the full 100-tile distribution minus the board minus the mover's
  rack, with board blanks counted as blanks.
- **Lexicon:** the full 15-letter list, loaded as text, identified by a hash that is stored in
  every checkpoint.

## Reproducing the checks

```powershell
cd C:\dev\scrabble-engine

# Rules and scoring bugs, and the per-move overhead (section 5)
py -3.12 tools\legacy_audit\repro_game_rules.py
py -3.12 tools\legacy_audit\repro_game_rules.py --full-dictionary "C:\dev\scrabble-engine-archive\onedrive-2025\MainDeepQ\scrabble_engine_github\word_dictionary (2025_09_13 23_08_50 UTC).csv"

# The format-spec crash (section 1)
py -3.12 -c "s=[0.5]; print(f\"{s[-1]:.4f if s else 'N/A'}\")"

# Lexicon facts (section 7)
uv run scrabble-ai lexicon
uv run pytest tests/test_lexicon.py
```

The Rust generator checks have their own instructions in `tools/legacy_audit/README.md`. The GUI,
evaluator and heuristic checks were run by a subagent against copies of the legacy files with
TensorFlow and the Rust extension stubbed out; its scripts were throwaway and are not in the repo,
so those rows are reproducible from the line references rather than from a saved command.
