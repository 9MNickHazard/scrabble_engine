# Legacy Rust move generator audit

Scripts that test the old `legacy\scrabble_valid_moves` crate (PyO3 module
`scrabble_valid_moves_rust`, entry point `get_valid_moves_rs(board, hand)`)
against an independent brute-force reference. Findings go in
`docs\legacy-audit.md`; this folder only holds the means to re-run them.

Nothing here writes into the repo. Build output, crate copies, dictionaries,
positions and results all go to a scratch folder you choose (`$S` below). The
lexicon files are copyrighted: keep them out of this folder.

## Files

| File | What it does |
|---|---|
| `build_legacy.ps1` | Copies the crate to `$S`, builds it as shipped (lib + bin, then lib only), builds an rlib probe of the bin, builds three diagnostic variants used only for attribution, and drops each DLL into `$S\pyd_<name>\scrabble_valid_moves_rust.pyd`. Logs in `$S\logs`. |
| `refgen.py` | Independent reference: brute-force legal-move enumerator, a separate per-move rule checker, standard scoring, random-game positions. Written from the rules, not from the Rust code. |
| `make_positions.py` | Writes test positions (hand-made ones aimed at specific claims, and random mid-game positions). |
| `legacy_runner.py` | Runs the legacy module on a positions file, the way `legacy\game_API.py` calls it (board = `{square: letter}`, no blank flags; blanks in the rack are `_`). |
| `compare.py` | Compares legacy output with the reference: missed moves, extra moves, duplicates, score disagreements, which blank assignment the legacy kept; optional attribution of each miss using the diagnostic builds. |
| `bench.py` | Wall-time baseline of `get_valid_moves_rs`, one worker process per position, with the process CPU share recorded. |
| `probe_misc.py` | Dictionary-path behaviour, memory of the dictionary, run-to-run determinism. |
| `summarize.py` | Prints the B/C tables, minimal missed-move examples per cause (with board diagrams) and the D summaries. |
| `repro_game_rules.py` | (added by the lead, not part of the Rust audit) Reproduces rule and scoring bugs in `legacy\game_API.py`; see its docstring. |

## Requirements

* Rust/cargo (tested with 1.92), Python 3.12 on PATH as `python` (PyO3 0.21 does
  not support 3.13+), or set `$env:PYO3_PYTHON`.
* The dictionary the legacy code loads: `word_dictionary.csv` (header row
  `valid_scrabble_words`, then 144,375 lowercase words of at most 10 letters).
  The audit used the old project's copy,
  `...\MainDeepQ\scrabble_engine_github\word_dictionary (2025_09_13 23_08_50 UTC).csv`
  in the OneDrive folder (read-only), copied to `$S\dict\word_dictionary_original.csv`.
  All comparisons use this same file for both generators, so differences are
  about the algorithm.
* The scripts set `sys.dont_write_bytecode`, so no `__pycache__` appears here.

## Commands (PowerShell, from the repo root)

```powershell
$S = "<scratch folder>"            # e.g. a folder under %TEMP%
$D = "$S\dict\word_dictionary_original.csv"
$T = "tools\legacy_audit"

# A. build (untouched lib+bin, lib only, rlib probe, diagnostic variants)
powershell -ExecutionPolicy Bypass -File $T\build_legacy.ps1 -Scratch $S

# B. positions
py -3.12 $T\make_positions.py --lexicon $D --out $S\B\pos_hand.json --hand
py -3.12 $T\make_positions.py --lexicon $D --out $S\B\pos_b0.json --random 30 --seed 1000 --blanks 0 --rack-size 7 --setup-blanks yes
py -3.12 $T\make_positions.py --lexicon $D --out $S\B\pos_b1.json --random 15 --seed 2000 --blanks 1 --rack-size 6 --setup-blanks no
py -3.12 $T\make_positions.py --lexicon $D --out $S\B\pos_b2.json --random 8  --seed 3000 --blanks 2 --rack-size 5 --setup-blanks no

# B. legacy outputs (one process per positions file)
foreach ($f in "hand","b0","b1","b2") {
  py -3.12 $T\legacy_runner.py --pyd-dir $S\pyd_untouched --dict $D --run-dir $S\run_orig `
     --positions $S\B\pos_$f.json --out $S\B\leg_$f.json
}

# B + C. comparison, scores, blank choice, attribution of every miss (about 11 min)
py -3.12 $T\compare.py --lexicon $D `
   --positions $S\B\pos_hand.json $S\B\pos_b0.json $S\B\pos_b1.json $S\B\pos_b2.json `
   --legacy    $S\B\leg_hand.json $S\B\leg_b0.json $S\B\leg_b1.json $S\B\leg_b2.json `
   --ref-cache $S\B\ref_cache.json --out $S\B\compare_all.json `
   --attribute --attribute-max-tiles 7 --dict $D --run-dir $S\run_attr `
   --pyd orig=$S\pyd_untouched --pyd noprune=$S\pyd_noprune --pyd nomaxspace=$S\pyd_nomaxspace --pyd both=$S\pyd_both
py -3.12 $T\summarize.py --compare $S\B\compare_all.json `
   --positions $S\B\pos_hand.json $S\B\pos_b0.json $S\B\pos_b1.json $S\B\pos_b2.json --examples 3

# cross-check of the reference: the "both" diagnostic build on 6 full-rack positions
py -3.12 -c "import json,sys; json.dump(json.load(open(sys.argv[1]))[:6], open(sys.argv[2],'w'))" $S\B\pos_b0.json $S\B\pos_b0_first6.json
py -3.12 $T\legacy_runner.py --pyd-dir $S\pyd_both --dict $D --run-dir $S\run_both `
   --positions $S\B\pos_b0_first6.json --out $S\B\leg_both_b0_first6.json
py -3.12 $T\compare.py --lexicon $D --positions $S\B\pos_b0_first6.json --legacy $S\B\leg_both_b0_first6.json `
   --ref-cache $S\B\ref_cache.json --out $S\B\compare_both_b0_first6.json

# D. speed baseline: same 30 boards, 7-tile racks with 0, 1, 2 blanks
foreach ($k in 0,1,2) {
  py -3.12 $T\make_positions.py --lexicon $D --out $S\D\pos_k$k.json --random 30 --seed 5000 --blanks $k --setup-blanks no --id-prefix s
}
py -3.12 $T\bench.py --pyd-dir $S\pyd_untouched --dict $D --run-dir $S\run_bench --positions $S\D\pos_k0.json --out $S\D\bench_k0.json --repeat 3
py -3.12 $T\bench.py --pyd-dir $S\pyd_untouched --dict $D --run-dir $S\run_bench --positions $S\D\pos_k1.json --out $S\D\bench_k1.json
# 2 blanks take tens of minutes per position: only the first few boards
py -3.12 $T\bench.py --pyd-dir $S\pyd_untouched --dict $D --run-dir $S\run_bench --positions $S\D\pos_k2.json --out $S\D\bench_k2.json --limit 3 --timeout 3600
py -3.12 $T\summarize.py --compare $S\B\compare_all.json --positions $S\B\pos_b0.json --examples 0 `
   --bench $S\D\bench_k0.json $S\D\bench_k1.json $S\D\bench_k2.json

# misc probes
py -3.12 $T\probe_misc.py --pyd-dir $S\pyd_untouched --dict $D --scratch $S
```

Run the benchmark on an otherwise idle machine; `bench.py` prints a
`cpu_share` per call so contention is visible.

## Notes on the reference

* A move is the set of placed tiles `{(square, letter, is_blank)}`; squares are
  named like the legacy code (`a1`..`o15`, letter = row, number = column, `h8`
  = centre).
* `refgen.legal_moves` enumerates, for every row and column, every maximal span
  that could hold the main word, matches lexicon words against it, then checks
  rack supply, connectivity and cross-words. Each legal choice of which tiles
  are blanks is a separate move.
* `refgen.check_placement` judges a single placement straight from the rules
  and is used to re-judge every disagreement, so a bug in the enumerator would
  show up as a disagreement rather than be silently trusted.
* Scores: premiums only under new tiles, blanks (new or already on the board)
  worth 0, every word of 2+ letters formed is scored, +50 for 7 tiles.
