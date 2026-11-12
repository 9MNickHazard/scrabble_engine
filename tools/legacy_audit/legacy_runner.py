"""Run the legacy Rust move generator (scrabble_valid_moves_rust) on a positions file.

The legacy module reads ./word_dictionary.csv from the *current directory* on its
first call, so this script copies the chosen dictionary into --run-dir and chdirs
there before importing. The board is passed the way legacy/game_API.py does it:
{square: letter} for occupied squares only, with no blank flag; blanks in the rack
are '_'.

Usage:
  python legacy_runner.py --pyd-dir DIR --dict CSV --run-dir DIR --positions P.json --out OUT.json
      [--repeat N]   (time each position N times; moves are taken from the first call)

DIR given to --pyd-dir must contain scrabble_valid_moves_rust.pyd. Run one
process per build: two builds share a module name and cannot load side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyd-dir", required=True)
    ap.add_argument("--dict", required=True, help="CSV in the legacy format (header + one word per row)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    positions = json.load(open(args.positions, encoding="utf-8"))
    os.makedirs(args.run_dir, exist_ok=True)
    shutil.copyfile(args.dict, os.path.join(args.run_dir, "word_dictionary.csv"))
    out_path = os.path.abspath(args.out)
    os.chdir(args.run_dir)
    sys.path.insert(0, os.path.abspath(args.pyd_dir))
    import scrabble_valid_moves_rust as legacy  # noqa: E402

    t = time.perf_counter()
    legacy.get_valid_moves_rs({}, ["a"])  # first call loads the dictionary
    load_s = time.perf_counter() - t

    results = []
    for n, pos in enumerate(positions):
        board = {s: v[0] for s, v in pos["board"].items()}
        rack = ["_" if ch == "?" else ch for ch in pos["rack"]]
        times = []
        moves = None
        for _ in range(args.repeat):
            t = time.perf_counter()
            res = legacy.get_valid_moves_rs(board, rack)
            times.append(time.perf_counter() - t)
            if moves is None:
                moves = res
        results.append({"id": pos["id"], "seconds": times, "moves": moves})
        print(f"[{n + 1}/{len(positions)}] {pos['id']}: {len(moves)} moves, "
              f"{min(times):.3f}s", file=sys.stderr, flush=True)

    meta = {"pyd": os.path.join(os.path.abspath(args.pyd_dir), "scrabble_valid_moves_rust.pyd"),
            "first_call_seconds": load_s, "python": sys.version, "cpu_count": os.cpu_count()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f)


if __name__ == "__main__":
    main()
