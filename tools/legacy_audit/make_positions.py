"""Write a JSON list of test positions: hand-made ones aimed at specific claims,
plus random mid-game positions produced by playing reference-legal moves.

Usage:
  python make_positions.py --lexicon CSV --out positions.json [--hand]
         [--random N] [--seed S] [--blanks natural|0|1|2] [--rack-size 7]
         [--plies-min 6] [--plies-max 14] [--id-prefix r]

--blanks natural : rack = next tiles from the shuffled bag (blanks may appear on
                   the board and in the rack at their natural rate)
--blanks 0|1|2   : the test rack is that many blanks plus non-blank letters from
                   the bag. For one seed the letters are the same for every k
                   (the k-blank rack replaces the last k letters with blanks).
--setup-blanks   : whether blanks are in the bag for the setup moves (so the board
                   can hold blanks). Use "no" to get identical boards for 0/1/2.
Position format: {"id", "kind", "note", "board": {square: [letter, is_blank]},
"rack": "letters, ? = blank", "target": [[square, letter, is_blank], ...] or null}
"""

from __future__ import annotations

import argparse
import json
import random
import sys

sys.dont_write_bytecode = True  # keep __pycache__ out of the repo
import refgen as R  # noqa: E402


def word_tiles(word: str, start: str, horizontal: bool, blanks=()):
    r, c = R.parse_sq(start)
    out = {}
    for i, ch in enumerate(word):
        rr, cc = (r, c + i) if horizontal else (r + i, c)
        out[R.sq_name(rr, cc)] = [ch, R.sq_name(rr, cc) in blanks]
    return out


def target(word_tiles_dict, board):
    return [[s, v[0], v[1]] for s, v in sorted(word_tiles_dict.items()) if s not in board]


def hand_positions():
    P = []

    def add(pid, note, board, rack, full_word=None):
        t = None
        if full_word:
            w, s, h = full_word
            t = target(word_tiles(w, s, h), board)
        P.append({"id": pid, "kind": "hand", "note": note, "board": board, "rack": rack, "target": t})

    # Claim 3: suffix extension (existing tiles before the first new tile).
    b = word_tiles("fig", "h6", True)
    add("hand_fight_h", "FIG on h6-h8; FIGHT needs H,T on h9-h10 (back extension, horizontal)",
        b, "htes", ("fight", "h6", True))
    b = word_tiles("bi", "g8", False)
    add("hand_bird_v", "BI on g8-h8 (down); BIRD needs R,D on i8-j8 (back extension, vertical)",
        b, "rdse", ("bird", "g8", False))
    b = word_tiles("ad", "h8", True)
    add("hand_adds_h", "AD on h8-h9; ADDS needs D,S on h10-h11 (back extension)",
        b, "dseo", ("adds", "h8", True))
    # Plays whose new tiles reach the last column / row.
    b = word_tiles("plea", "h8", True)
    add("hand_pleasant_h", "PLEA on h8-h11; PLEASANT needs S,A,N,T on h12-h15 (reaches column 15)",
        b, "santdeu", ("pleasant", "h8", True))
    b = word_tiles("plea", "h8", False)
    add("hand_pleasant_v", "PLEA on h8-k8 (down); PLEASANT needs S,A,N,T on l8-o8 (reaches row o)",
        b, "santdeu", ("pleasant", "h8", False))
    # Controls: front extensions (new tiles before an existing word).
    b = word_tiles("nation", "h3", True)
    add("hand_donation_h", "NATION on h3-h8; DONATION needs D,O on h1-h2 (front extension to column 1)",
        b, "doe", ("donation", "h1", True))
    b = word_tiles("ion", "h8", False)
    add("hand_union_v", "ION on h8-j8 (down); UNION needs U,N on f8-g8 (front extension, vertical)",
        b, "uns", ("union", "f8", False))
    # Claim 5: a blank already on the board.
    b = word_tiles("qi", "h8", True, blanks={"h8"})
    add("hand_blank_on_board", "QI on h8-h9 with the Q a blank; QIS on h10 scores the blank Q as 0",
        b, "s", ("qis", "h8", True))
    return P


def random_positions(lex, n, seed, blanks, rack_size, pmin, pmax, prefix, setup_blanks):
    P = []
    for i in range(n):
        s = seed + i
        rng = random.Random(s)
        plies = rng.randint(pmin, pmax)
        board, bag = R.random_game(lex, rng, plies, blanks_in_bag=setup_blanks)
        if blanks == "natural":
            rack = "".join(bag[-rack_size:])  # next tiles a player would draw
            note = f"seed {s}, {plies} plies, natural rack"
        else:
            k = int(blanks)
            letters = [t for t in bag if t != R.BLANK][-7:]  # same 7 letters for every k
            rack = "".join(letters[: rack_size - k]) + R.BLANK * k
            note = f"seed {s}, {plies} plies, {k} blank(s) in rack"
        note += ", blanks allowed during setup" if setup_blanks else ", no blanks during setup"
        P.append({"id": f"{prefix}{s}_b{blanks}", "kind": "random", "note": note,
                  "board": {sq: [l, b] for sq, (l, b) in board.items()},
                  "rack": rack, "target": None})
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--blanks", default="natural", choices=["natural", "0", "1", "2"])
    ap.add_argument("--rack-size", type=int, default=7)
    ap.add_argument("--plies-min", type=int, default=6)
    ap.add_argument("--plies-max", type=int, default=14)
    ap.add_argument("--id-prefix", default="r")
    ap.add_argument("--setup-blanks", default="auto", choices=["auto", "yes", "no"],
                    help="blanks in the bag during the setup moves; auto = yes for "
                         "--blanks natural/0, no for 1/2")
    a = ap.parse_args()
    lex = R.load_lexicon(a.lexicon)
    P = hand_positions() if a.hand else []
    if a.random:
        setup = a.setup_blanks == "yes" or (a.setup_blanks == "auto" and a.blanks in ("natural", "0"))
        P += random_positions(lex, a.random, a.seed, a.blanks, a.rack_size,
                              a.plies_min, a.plies_max, a.id_prefix, setup)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(P, f, indent=1)
    print(f"wrote {len(P)} positions to {a.out}")


if __name__ == "__main__":
    main()
