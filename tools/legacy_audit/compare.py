"""Compare the legacy Rust move generator with the independent reference.

Usage:
  python compare.py --lexicon CSV --positions P1.json [P2.json ...] --legacy L1.json [L2.json ...] --out OUT.json
         [--ref-cache REF.json]
         [--attribute --runner-python PY --dict CSV --run-dir DIR
          --pyd orig=DIR --pyd noprune=DIR --pyd nomaxspace=DIR --pyd both=DIR]

What it reports (all moves normalised to sets of placed tiles):
  * placement level  = {(square, letter)}: what the generator can play at all.
  * full level       = {(square, letter, is_blank)}: also which tiles are blanks.
  * missed           = reference-legal moves the legacy output lacks.
  * extra            = legacy moves the reference lacks; each is re-judged by
                       refgen.check_placement (a direct rule check that does not
                       use the enumerator), so a reference bug would show here.
  * duplicates       = legacy entries that repeat an identical placement.
  * scores           = legacy score vs reference score on moves both list;
                       disagreements are re-scored with board blanks counted at
                       face value to see whether that explains them.
  * blank choice     = for each word the legacy returns, whether the blank
                       assignment it kept is the best-scoring legal one.
--attribute re-runs every missed placement with the rack reduced to exactly its
letters (no blanks) on each given build, to see which change makes it appear.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

sys.dont_write_bytecode = True  # keep __pycache__ out of the repo
import refgen as R  # noqa: E402


def ref_moves_cached(positions, lex, cache_path):
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    changed = False
    out = {}
    for n, pos in enumerate(positions):
        key = pos["id"]
        if key not in cache:
            board = {s: tuple(v) for s, v in pos["board"].items()}
            mv = R.legal_moves(board, pos["rack"], lex)
            cache[key] = [[sorted([list(t) for t in p]), s] for p, s in mv.items()]
            changed = True
            print(f"  reference {n + 1}/{len(positions)} {key}: {len(mv)} moves", file=sys.stderr, flush=True)
        out[key] = {frozenset((t[0], t[1], bool(t[2])) for t in p): s for p, s in cache[key]}
    if cache_path and changed:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    return out


def normalise_legacy(mv, board):
    placed = []
    for i, sq in enumerate(mv["positions"]):
        if sq not in board:
            placed.append((sq, mv["word"][i], sq in mv["blank_assignments"]))
    problems = []
    for sq, letter in mv["blank_assignments"].items():
        if sq not in mv["positions"] or sq in board:
            problems.append(f"blank on {sq} is not a placed square")
        elif mv["word"][mv["positions"].index(sq)] != letter:
            problems.append(f"blank letter mismatch on {sq}")
    return frozenset(placed), problems


def strip(p):
    return frozenset((s, l) for s, l, _ in p)


def describe(board, placement):
    """Feature tags used to group missed moves."""
    grid = R.to_grid({s: tuple(v) for s, v in board.items()})
    sq = sorted(R.parse_sq(s) for s, _, _ in placement)
    rows = {r for r, _ in sq}
    tags = []
    if len(sq) == 1:
        tags.append("single tile")
        horizontal = None
    else:
        horizontal = len(rows) == 1
        tags.append("horizontal" if horizontal else "vertical")
    if any(c == R.SIZE - 1 for _, c in sq) and horizontal is not False:
        tags.append("new tile in column 15")
    if any(r == R.SIZE - 1 for r, _ in sq) and horizontal is not True:
        tags.append("new tile in row o")
    if horizontal is not None:
        dr, dc = (0, 1) if horizontal else (1, 0)
        r0, c0 = sq[0]
        if R._get(grid, r0 - dr, c0 - dc) is not None:
            tags.append("existing tile(s) before first new tile")
        r1, c1 = sq[-1]
        if R._get(grid, r1 + dr, c1 + dc) is not None:
            tags.append("existing tile(s) after last new tile")
        span = (sq[-1][1] - sq[0][1] + 1) if horizontal else (sq[-1][0] - sq[0][0] + 1)
        if span > len(sq):
            tags.append("through existing tile(s)")
    return tags


def run_legacy(python, runner, pyd, dict_csv, run_dir, positions):
    os.makedirs(run_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run_dir) as td:
        pin = os.path.join(td, "p.json")
        pout = os.path.join(td, "o.json")
        json.dump(positions, open(pin, "w", encoding="utf-8"))
        subprocess.run([python, runner, "--pyd-dir", pyd, "--dict", dict_csv, "--run-dir", run_dir,
                        "--positions", pin, "--out", pout], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.load(open(pout, encoding="utf-8"))["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--positions", required=True, nargs="+")
    ap.add_argument("--legacy", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-cache")
    ap.add_argument("--attribute", action="store_true")
    ap.add_argument("--runner-python", default=sys.executable)
    ap.add_argument("--dict")
    ap.add_argument("--run-dir")
    ap.add_argument("--pyd", action="append", default=[], help="name=DIR, first one is the baseline")
    ap.add_argument("--attribute-max-tiles", type=int, default=5,
                    help="only attribute missed moves with at most this many tiles (the no-pruning "
                         "build enumerates every permutation, so 6-7 tiles get slow)")
    ap.add_argument("--attribute-sample", type=int, default=0,
                    help="attribute at most this many eligible misses (all hand-made ones plus a "
                         "fixed-seed random sample of the rest); 0 = all")
    a = ap.parse_args()

    lex = R.load_lexicon(a.lexicon)
    positions = [p for f in a.positions for p in json.load(open(f, encoding="utf-8"))]
    legacy = {r["id"]: r for f in a.legacy for r in json.load(open(f, encoding="utf-8"))["results"]}
    positions = [p for p in positions if p["id"] in legacy]
    ref = ref_moves_cached(positions, lex, a.ref_cache)

    tot = Counter()
    per_pos = []
    missed_records, extra_records, score_records, blank_choice_records = [], [], [], []
    target_checks = []
    for pos in positions:
        pid = pos["id"]
        board = {s: tuple(v) for s, v in pos["board"].items()}
        board_fv = {s: (v[0], False) for s, v in pos["board"].items()}
        grid = R.to_grid(board)
        grid_fv = R.to_grid(board_fv)
        rmoves = ref[pid]
        lentries = []
        for mv in legacy[pid]["moves"]:
            p, probs = normalise_legacy(mv, pos["board"])
            lentries.append((p, mv, probs))
            tot["legacy_blank_problems"] += len(probs)
        ref_full = set(rmoves)
        ref_place = defaultdict(list)
        for p in ref_full:
            ref_place[strip(p)].append(p)
        leg_full = Counter(p for p, _, _ in lentries)
        leg_place = {strip(p) for p in leg_full}

        missed_place = set(ref_place) - leg_place
        missed_full = ref_full - set(leg_full)
        extra_full = set(leg_full) - ref_full
        dupes = sum(n - 1 for n in leg_full.values() if n > 1)

        # sanity: the direct rule check must accept every reference move
        bad_ref = [p for p in ref_full if not R.check_placement(board, pos["rack"], p, lex)[0]]
        tot["reference_moves_rejected_by_checker"] += len(bad_ref)

        for p in extra_full:
            ok, why = R.check_placement(board, pos["rack"], p, lex)
            try:
                words = R.words_formed(grid, p)
            except ValueError:
                words = []
            extra_records.append({"id": pid, "move": sorted(p), "checker_says_legal": ok, "reason": why,
                                  "words": words})
        for p in missed_place:
            variants = ref_place[p]
            best = max(variants, key=lambda v: rmoves[v])
            missed_records.append({
                "id": pid, "rack": pos["rack"], "move": sorted(best), "n_tiles": len(p),
                "words": R.words_formed(grid, best), "ref_score": rmoves[best],
                "tags": describe(pos["board"], best),
            })
        # scores on moves both list
        for p, mv, _ in lentries:
            if p not in rmoves:
                continue
            tot["score_pairs"] += 1
            ls, rs = mv["score"], rmoves[p]
            if ls != rs:
                fv = R.score_placement(grid_fv, p)
                score_records.append({"id": pid, "word": mv["word"], "start": mv["start_space"],
                                      "direction": mv["direction"], "move": sorted(p),
                                      "legacy": ls, "reference": rs, "reference_board_blanks_face_value": fv,
                                      "explained_by_board_blanks": fv == ls,
                                      "words": R.words_formed(grid, p)})
        # which blank assignment did the legacy keep?
        for p, mv, _ in lentries:
            variants = ref_place.get(strip(p), [])
            if len(variants) < 2 or p not in rmoves:
                continue
            best = max(rmoves[v] for v in variants)
            tot["legacy_words_with_alternative_blank_assignments"] += 1
            if rmoves[p] < best:
                tot["legacy_kept_lower_scoring_blank_assignment"] += 1
                blank_choice_records.append({
                    "id": pid, "word": mv["word"], "start": mv["start_space"], "direction": mv["direction"],
                    "rack": pos["rack"], "legacy_move": sorted(p), "legacy_choice_ref_score": rmoves[p],
                    "legacy_reported_score": mv["score"],
                    "best_variant": sorted(max(variants, key=lambda v: rmoves[v])), "best_score": best,
                    "n_variants": len(variants)})
        if pos.get("target"):
            t = frozenset((s, l) for s, l, _ in pos["target"])
            target_checks.append({"id": pid, "note": pos["note"], "target": sorted(t),
                                  "reference_has_it": t in ref_place, "legacy_has_it": t in leg_place})

        row = {"id": pid, "kind": pos["kind"], "rack": pos["rack"], "board_tiles": len(board),
               "board_blanks": sum(1 for v in board.values() if v[1]),
               "ref_full": len(ref_full), "ref_placements": len(ref_place),
               "legacy_entries": len(lentries), "legacy_full_unique": len(leg_full),
               "legacy_placements": len(leg_place), "missed_placements": len(missed_place),
               "missed_full": len(missed_full), "extra_full": len(extra_full), "duplicates": dupes,
               "legacy_seconds": min(legacy[pid]["seconds"])}
        per_pos.append(row)
        for k in ("ref_full", "ref_placements", "legacy_entries", "legacy_full_unique", "legacy_placements",
                  "missed_placements", "missed_full", "extra_full", "duplicates"):
            tot[k] += row[k]
        tot["positions"] += 1
        tot["positions_with_missed_placement"] += row["missed_placements"] > 0
        tot["positions_with_extra"] += row["extra_full"] > 0

    tot["score_disagreements"] = len(score_records)
    tot["score_disagreements_explained_by_board_blanks"] = sum(r["explained_by_board_blanks"] for r in score_records)
    tag_counts = Counter(t for r in missed_records for t in r["tags"])
    edge_or_back = Counter()
    for r in missed_records:
        edge = any("column 15" in t or "row o" in t for t in r["tags"])
        back = "existing tile(s) before first new tile" in r["tags"]
        edge_or_back[("edge" if edge else "no-edge") + "/" + ("before" if back else "no-before")] += 1

    attribution = None
    if a.attribute and missed_records:
        pyds = [x.split("=", 1) for x in a.pyd]
        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_runner.py")
        chosen = [r for r in missed_records if r["n_tiles"] <= a.attribute_max_tiles]
        eligible = len(chosen)
        if a.attribute_sample and len(chosen) > a.attribute_sample:
            import random
            hand = [r for r in chosen if r["id"].startswith("hand")]
            rest = [r for r in chosen if not r["id"].startswith("hand")]
            chosen = hand + random.Random(12345).sample(rest, max(0, a.attribute_sample - len(hand)))
        mini = []
        for n, r in enumerate(chosen):
            pos = next(p for p in positions if p["id"] == r["id"])
            mini.append({"id": f"m{n}", "kind": "mini", "note": "", "board": pos["board"],
                         "rack": "".join(l for _, l, _ in r["move"]), "target": None})
        found = {}
        for name, d in pyds:
            res = run_legacy(a.runner_python, runner, d, a.dict, a.run_dir, mini)
            found[name] = {}
            for m, rr in zip(mini, res):
                pl = set()
                for mv in rr["moves"]:
                    p, _ = normalise_legacy(mv, m["board"])
                    pl.add(strip(p))
                found[name][m["id"]] = pl
            print(f"  attribution run '{name}' done", file=sys.stderr, flush=True)
        classes = Counter()
        for n, r in enumerate(chosen):
            t = frozenset((s, l) for s, l, _ in r["move"])
            hit = {name: (t in found[name][f"m{n}"]) for name, _ in pyds}
            r["found_by_build"] = hit
            classes[" ".join(f"{k}={'Y' if v else 'n'}" for k, v in hit.items())] += 1
        attribution = {"attributed": len(chosen), "eligible": eligible,
                       "skipped_over_max_tiles": len(missed_records) - eligible,
                       "max_tiles": a.attribute_max_tiles, "classes": dict(classes)}

    secs = [r["legacy_seconds"] for r in per_pos]
    summary = {"totals": dict(tot), "missed_tag_counts": dict(tag_counts),
               "missed_edge_vs_before": dict(edge_or_back), "attribution": attribution,
               "legacy_seconds_median": statistics.median(secs) if secs else None,
               "target_checks": target_checks}
    json.dump({"summary": summary, "per_position": per_pos, "missed": missed_records,
               "extra": extra_records, "score_disagreements": score_records,
               "blank_choice": blank_choice_records}, open(a.out, "w", encoding="utf-8"), indent=1)
    printable = {k: v for k, v in summary.items() if k != "target_checks"}
    print(json.dumps(printable, indent=1))
    for t in target_checks:
        print(f"target {t['id']}: reference={t['reference_has_it']} legacy={t['legacy_has_it']}  ({t['note']})")


if __name__ == "__main__":
    main()
