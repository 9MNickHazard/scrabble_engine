"""Print the tables and minimal counterexamples from compare.py / bench.py output.

  python summarize.py --compare compare_all.json --positions P1.json [P2.json ...]
                      [--bench bench_k0.json bench_k1.json ...] [--examples 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True  # keep __pycache__ out of the repo
import refgen as R  # noqa: E402


def board_text(board, highlight=()):
    """Board diagram: existing tiles lowercase (a blank on the board is uppercase),
    the move's new tiles shown as +letter; rows far from any tile are omitted."""
    grid = R.to_grid({s: tuple(v) for s, v in board.items()})
    hl = {R.parse_sq(s): l for s, l, *_ in highlight}
    rows = set(r for r, _ in hl) | {r for r in range(R.SIZE) if any(grid[r])}
    lines = ["    " + "".join(f"{c + 1:>3}" for c in range(R.SIZE))]
    for r in range(R.SIZE):
        if r not in rows and not any(abs(r - x) <= 1 for x in rows):
            continue
        cells = []
        for c in range(R.SIZE):
            t = grid[r][c]
            if (r, c) in hl:
                cells.append(f" +{hl[(r, c)]}")
            elif t is None:
                cells.append("  .")
            else:
                cells.append("  " + (t[0].upper() if t[1] else t[0]))
        lines.append(f"  {R.ROWS[r]} " + "".join(cells))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True)
    ap.add_argument("--positions", nargs="+", required=True)
    ap.add_argument("--bench", nargs="*", default=[])
    ap.add_argument("--examples", type=int, default=4)
    a = ap.parse_args()
    pos = {p["id"]: p for f in a.positions for p in json.load(open(f, encoding="utf-8"))}
    d = json.load(open(a.compare, encoding="utf-8"))
    s = d["summary"]
    t = s["totals"]

    def group(row):
        if row["kind"] == "hand":
            return "hand-made"
        k = row["rack"].count("?")
        return f"random, {len(row['rack'])}-tile rack, {k} blank(s)"

    print("== B. completeness / correctness (placement = squares+letters; full = also blank flags)")
    g = defaultdict(Counter)
    for row in d["per_position"]:
        k = group(row)
        g[k]["positions"] += 1
        for f in ("ref_placements", "ref_full", "legacy_entries", "legacy_placements", "missed_placements",
                  "missed_full", "extra_full", "duplicates"):
            g[k][f] += row[f]
        g[k]["positions_with_miss"] += row["missed_placements"] > 0
    hdr = ["set", "positions", "pos. w/ miss", "ref placements", "legacy placements", "missed placements",
           "ref full", "missed full", "legacy extra", "legacy dup entries"]
    print(" | ".join(hdr))
    for k in sorted(g):
        c = g[k]
        print(" | ".join(str(x) for x in [k, c["positions"], c["positions_with_miss"], c["ref_placements"],
                                          c["legacy_placements"], c["missed_placements"], c["ref_full"],
                                          c["missed_full"], c["extra_full"], c["duplicates"]]))
    print("totals:", json.dumps(t))
    print("missed by feature:", s["missed_tag_counts"])
    print("missed edge/before split:", s["missed_edge_vs_before"])
    print("attribution:", json.dumps(s["attribution"]))
    for tc in s.get("target_checks", []):
        print(f"target {tc['id']}: reference={tc['reference_has_it']} legacy={tc['legacy_has_it']} ({tc['note']})")

    # minimal missed examples per cause
    by_cause = defaultdict(list)
    for r in d["missed"]:
        fb = r.get("found_by_build")
        if fb is None:
            cause = "unattributed"
        elif fb.get("orig"):
            cause = "found by orig with reduced rack (?)"
        elif fb.get("noprune") and not fb.get("nomaxspace"):
            cause = "prefix pruning only (no-pruning build finds it)"
        elif fb.get("nomaxspace") and not fb.get("noprune"):
            cause = "max_space only (no-max_space build finds it)"
        elif fb.get("noprune") and fb.get("nomaxspace"):
            cause = "either fix alone finds it"
        elif fb.get("both"):
            cause = "needs both fixes"
        else:
            cause = "none of the builds"
        by_cause[cause].append(r)
    for cause, rs in sorted(by_cause.items()):
        rs.sort(key=lambda r: (r["n_tiles"], len(r["words"][0]) if r["words"] else 99, r["id"]))
        print(f"\n== missed, cause: {cause} ({len(rs)})")
        seen_ids = set()
        shown = 0
        for r in rs:
            if shown >= a.examples:
                break
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            p = pos[r["id"]]
            print(f"-- {r['id']} rack {p['rack'].upper()}  move {[(m[0], m[1].upper(), 'blank' if m[2] else '') for m in r['move']]}"
                  f"  words {r['words']}  ref score {r['ref_score']}  tags {r['tags']}")
            print(board_text(p["board"], r["move"]))
            shown += 1

    print("\n== C. scores on moves both list")
    sd = d["score_disagreements"]
    print(f"pairs compared {t.get('score_pairs')}, disagreements {len(sd)}, "
          f"explained by board blanks at face value {sum(x['explained_by_board_blanks'] for x in sd)}")
    for x in sd[: a.examples]:
        print(" ", json.dumps(x))
    unexplained = [x for x in sd if not x["explained_by_board_blanks"]]
    for x in unexplained[: a.examples]:
        print("  UNEXPLAINED", json.dumps(x))
    bc = d["blank_choice"]
    print(f"\nlegacy words with >1 legal blank assignment: {t.get('legacy_words_with_alternative_blank_assignments', 0)}; "
          f"kept a lower-scoring one: {t.get('legacy_kept_lower_scoring_blank_assignment', 0)}")
    for x in sorted(bc, key=lambda x: x["legacy_choice_ref_score"] - x["best_score"])[: a.examples]:
        print(" ", json.dumps(x))

    print("\n== extra (legacy-only) moves")
    for x in d["extra"][: a.examples]:
        print(" ", json.dumps(x))

    for f in a.bench:
        b = json.load(open(f, encoding="utf-8"))
        print(f"\n== D. bench {f}")
        print(json.dumps(b["meta"]))
        print(json.dumps(b["summary_by_blanks"], indent=1))


if __name__ == "__main__":
    main()
