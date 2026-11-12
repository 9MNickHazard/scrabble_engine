"""Wall-time baseline for the legacy get_valid_moves_rs call.

Each position is timed in its own worker process (so a runaway call can be cut off
with --timeout). The worker chdirs to --run-dir (where word_dictionary.csv lives),
imports the .pyd, makes one warm-up call that loads the dictionary, then times
--repeat calls of get_valid_moves_rs(board, rack) with time.perf_counter(). The
board is passed the way legacy/game_API.py passes it: {square: letter}.

It also records the worker's CPU time during each call (os.times user+system)
and reports cpu_share = cpu_seconds / (wall_seconds * logical_cpus). A value well
below 1.0 while the call is CPU-bound means other processes were competing for
the cores (or rayon's per-anchor work items were unevenly sized).

--out is rewritten after every position; meta.complete is true only when the
whole run finished.

Usage:
  python bench.py --pyd-dir DIR --dict CSV --run-dir DIR --positions P.json [P2.json ...]
                  --out OUT.json [--repeat 1] [--timeout 1800] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict


def worker(a):
    pos = json.loads(sys.stdin.read())
    os.chdir(a.run_dir)
    sys.path.insert(0, os.path.abspath(a.pyd_dir))
    import scrabble_valid_moves_rust as legacy
    legacy.get_valid_moves_rs({}, ["a"])  # dictionary load, not timed
    board = {s: v[0] for s, v in pos["board"].items()}
    rack = ["_" if ch == "?" else ch for ch in pos["rack"]]
    runs = []
    for _ in range(a.repeat):
        t0, c0 = time.perf_counter(), os.times()
        res = legacy.get_valid_moves_rs(board, rack)
        t1, c1 = time.perf_counter(), os.times()
        cpu = (c1.user - c0.user) + (c1.system - c0.system)
        runs.append({"wall": t1 - t0, "cpu": cpu, "moves": len(res)})
    print(json.dumps(runs))


def summarize(rows):
    walls = [r["wall"] for r in rows if r.get("wall") is not None]
    timeouts = sum(1 for r in rows if r.get("timeout"))
    if not walls:
        return {"n": 0, "timeouts": timeouts}
    return {"n": len(walls), "timeouts": timeouts, "median_s": statistics.median(walls),
            "min_s": min(walls), "max_s": max(walls), "mean_s": statistics.fmean(walls),
            "median_cpu_share": statistics.median(r["cpu_share"] for r in rows if r.get("wall") is not None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyd-dir", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--positions", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="only the first N positions of each file")
    ap.add_argument("--worker", action="store_true")
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    os.makedirs(a.run_dir, exist_ok=True)
    shutil.copyfile(a.dict, os.path.join(a.run_dir, "word_dictionary.csv"))
    ncpu = os.cpu_count()
    rows = []

    def save(complete: bool):
        # rewritten after every position (atomically), so an interrupted run keeps what it measured
        by_k = {}
        for k in sorted({r["blanks"] for r in rows}):
            by_k[str(k)] = summarize([r for r in rows if r["blanks"] == k and r.get("repeat", 0) == 0])
            if a.repeat > 1:
                per_pos = defaultdict(list)
                for r in rows:
                    if r["blanks"] == k and r.get("wall") is not None:
                        per_pos[r["id"]].append(r["wall"])
                meds = [statistics.median(v) for v in per_pos.values()]
                by_k[str(k)]["median_of_per_position_median_s"] = statistics.median(meds)
                by_k[str(k)]["max_of_per_position_median_s"] = max(meds)
                by_k[str(k)]["max_spread_between_repeats_s"] = max(max(v) - min(v) for v in per_pos.values())
        out = {"meta": {"python": sys.version, "logical_cpus": ncpu, "repeat": a.repeat, "timeout_s": a.timeout,
                        "pyd": os.path.join(os.path.abspath(a.pyd_dir), "scrabble_valid_moves_rust.pyd"),
                        "complete": complete},
               "summary_by_blanks": by_k, "rows": rows}
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, a.out)
        return by_k

    for f in a.positions:
        plist = json.load(open(f, encoding="utf-8"))
        if a.limit:
            plist = plist[: a.limit]
        for pos in plist:
            k = pos["rack"].count("?")
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--pyd-dir", a.pyd_dir,
                   "--dict", a.dict, "--run-dir", a.run_dir, "--repeat", str(a.repeat)]
            try:
                p = subprocess.run(cmd, input=json.dumps(pos), capture_output=True, text=True,
                                   timeout=a.timeout, check=True)
                runs = json.loads(p.stdout.strip().splitlines()[-1])
                for n, r in enumerate(runs):
                    rows.append({"id": pos["id"], "blanks": k, "rack_len": len(pos["rack"]),
                                 "board_tiles": len(pos["board"]), "repeat": n, "wall": r["wall"],
                                 "cpu": r["cpu"], "cpu_share": r["cpu"] / (r["wall"] * ncpu),
                                 "moves": r["moves"]})
                    print(f"{pos['id']}: {k} blank(s) {r['wall']:.3f}s cpu_share {rows[-1]['cpu_share']:.2f} "
                          f"{r['moves']} moves", flush=True)
            except subprocess.TimeoutExpired:
                rows.append({"id": pos["id"], "blanks": k, "rack_len": len(pos["rack"]),
                             "board_tiles": len(pos["board"]), "timeout": a.timeout, "wall": None})
                print(f"{pos['id']}: {k} blank(s) TIMEOUT after {a.timeout}s", flush=True)
            save(complete=False)
    print(json.dumps(save(complete=True), indent=1))


if __name__ == "__main__":
    main()
