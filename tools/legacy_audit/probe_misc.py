"""Small probes of the legacy module's behaviour (each in a fresh process).

  python probe_misc.py --pyd-dir DIR --dict CSV --scratch DIR

1. dictionary path: first call from a directory without word_dictionary.csv,
   then chdir to a directory that has it and call again in the same process.
2. memory: process private bytes before and after the dictionary load.
3. determinism: the same position (a rack with a blank) called 5 times in one
   process and once in each of 3 more processes; compares the move lists.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
from ctypes import wintypes


def private_bytes() -> int:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PrivateUsage


# A small position with a blank in the rack: PLEA on h8-h11, rack E A S ?
BOARD = {"h8": "p", "h9": "l", "h10": "e", "h11": "a"}
RACK = ["e", "a", "s", "_"]


def child(mode, a):
    sys.path.insert(0, os.path.abspath(a.pyd_dir))
    if mode == "path":
        empty = os.path.join(a.scratch, "probe_empty_dir")
        os.makedirs(empty, exist_ok=True)
        os.chdir(empty)
        import scrabble_valid_moves_rust as legacy
        out = {}
        try:
            legacy.get_valid_moves_rs({}, ["a", "t"])
            out["first_call"] = "ok"
        except Exception as e:  # noqa: BLE001
            out["first_call"] = f"{type(e).__name__}: {e}"
        os.chdir(a.run_dir)
        out["cwd_now_has_dictionary"] = os.path.exists("word_dictionary.csv")
        try:
            r = legacy.get_valid_moves_rs({}, ["a", "t"])
            out["second_call"] = f"ok, {len(r)} moves"
        except Exception as e:  # noqa: BLE001
            out["second_call"] = f"{type(e).__name__}: {e}"
        print(json.dumps(out))
    elif mode == "memory":
        os.chdir(a.run_dir)
        before = private_bytes()
        import scrabble_valid_moves_rust as legacy
        after_import = private_bytes()
        legacy.get_valid_moves_rs({}, ["a"])
        after_load = private_bytes()
        print(json.dumps({"private_bytes_before_import": before, "after_import": after_import,
                          "after_first_call": after_load,
                          "dictionary_load_delta_MB": (after_load - after_import) / 2**20}))
    elif mode == "determinism":
        os.chdir(a.run_dir)
        import scrabble_valid_moves_rust as legacy
        runs = []
        for _ in range(a.calls):
            res = legacy.get_valid_moves_rs(BOARD, RACK)
            runs.append([[m["word"], m["start_space"], m["direction"], m["score"],
                          sorted(m["blank_assignments"].items())] for m in res])
        print(json.dumps(runs))


def run_child(mode, a, calls=1):
    cmd = [sys.executable, os.path.abspath(__file__), "--child", mode, "--pyd-dir", a.pyd_dir,
           "--dict", a.dict, "--scratch", a.scratch, "--calls", str(calls)]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    return json.loads(p.stdout.strip().splitlines()[-1]), p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyd-dir", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--child")
    ap.add_argument("--calls", type=int, default=1)
    a = ap.parse_args()
    a.run_dir = os.path.join(a.scratch, "probe_run_dir")
    if a.child:
        return child(a.child, a)
    os.makedirs(a.run_dir, exist_ok=True)
    shutil.copyfile(a.dict, os.path.join(a.run_dir, "word_dictionary.csv"))

    res, out, err = run_child("path", a)
    print("1. dictionary path:", json.dumps(res, indent=1))
    print("   module stdout/stderr:", (out.strip().splitlines()[:-1] + err.strip().splitlines()))

    res, _, _ = run_child("memory", a)
    print("2. memory:", json.dumps(res, indent=1))

    in_proc, _, _ = run_child("determinism", a, calls=5)
    others = [run_child("determinism", a, calls=1)[0][0] for _ in range(3)]
    runs = in_proc + others
    as_sets = [frozenset(json.dumps(m) for m in r) for r in runs]
    print("3. determinism over", len(runs), "calls (5 in one process, 3 in separate processes):")
    print("   moves per call:", [len(r) for r in runs])
    print("   identical move lists incl. order:", sum(r == runs[0] for r in runs), "of", len(runs))
    print("   identical move sets ignoring order:", sum(s == as_sets[0] for s in as_sets), "of", len(runs))
    by_word = {}
    for r in runs:
        for w, s, d, sc, bl in r:
            if bl:
                by_word.setdefault((w, s, d), set()).add((sc, json.dumps(bl)))
    varying = {k: sorted(v) for k, v in by_word.items() if len(v) > 1}
    print("   (word,start,dir) whose kept blank assignment changed between calls:", len(varying))
    for k, v in list(varying.items())[:5]:
        print("    ", k, "->", v)


if __name__ == "__main__":
    main()
