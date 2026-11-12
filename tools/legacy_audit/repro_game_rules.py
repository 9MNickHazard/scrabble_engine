"""Reproduce the rule and scoring bugs in legacy/game_API.py (see docs/legacy-audit.md).

Runs the old code as it was, against a tiny throwaway word list, and prints what it does.
The Rust move generator is not needed: these bugs are in the Python validate/score/play path.

Usage:
    py -3.12 tools/legacy_audit/repro_game_rules.py
    py -3.12 tools/legacy_audit/repro_game_rules.py --full-dictionary PATH\\word_dictionary.csv

--full-dictionary additionally times game construction and the per-move deepcopy with the
real 144,375-word list the old code loaded (a one-column CSV with a header line).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[2] / "legacy"
TINY_WORDS = ["cat", "cats", "ox", "at", "ta", "qi", "it"]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-dictionary", type=Path, default=None)
    args = parser.parse_args()

    # game_API imports the Rust extension at module level; a stub is enough for these checks.
    sys.modules["scrabble_valid_moves_rust"] = types.ModuleType("scrabble_valid_moves_rust")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(LEGACY))
    import game_API

    workdir = Path(tempfile.mkdtemp(prefix="legacy_rules_"))
    try:
        os.chdir(workdir)  # the old code opens "word_dictionary.csv" relative to the cwd
        Path("word_dictionary.csv").write_text("valid_scrabble_words\n" + "\n".join(TINY_WORDS) + "\n")

        def new_game(rack1: str, rack2: str, bag: dict | None = None):
            game = quiet(game_API.Scrabble_Game)
            game.player_1_hand, game.player_2_hand = list(rack1), list(rack2)
            if bag is not None:
                game.letter_counts = {letter: 0 for letter in game.letter_counts} | bag
            return game

        print("1. The main word is not extended through adjacent tiles")
        game = new_game("catoxxx", "oxeeeee")
        quiet(game.play_move, 1, "cat", "h8", "right")
        accepted = quiet(game.play_move, 2, "ox", "h11", "right")
        row = "".join(game.is_tile_present[f"h{col}"][1] or "." for col in range(7, 14))
        print(f"   CAT at h8, then OX at h11: accepted={accepted}, row h reads {row!r}, "
              f"scored {game.player_2_score} (CATOX is not a word)")

        print("2. A move that places no tiles is accepted and scores")
        game = new_game("catzzzz", "eeeeeee")
        quiet(game.play_move, 1, "cat", "h8", "right")
        accepted = quiet(game.play_move, 2, "cat", "h8", "right")
        print(f"   replaying CAT on its own squares: accepted={accepted}, "
              f"scored {game.player_2_score}, rack still {len(game.player_2_hand)} tiles")

        print("3. A blank used without being declared scores at face value")
        game = new_game("_ieeeee", "eeeeeee")
        accepted = quiet(game.play_move, 1, "qi", "h8", "right")
        print(f"   QI with the blank as Q, no blank_assignments: accepted={accepted}, "
              f"scored {game.player_1_score} (correct: 2)")

        print("4. A blank already on the board scores at face value on later turns")
        game = new_game("_ieeeee", "eeeeeee")
        quiet(game.play_move, 1, "qi", "h8", "right", {"h8": "q"})
        print(f"   declared blank Q scored {game.player_1_score} when played (correct); "
              f"re-scoring QI through it gives {game.point_calculation('qi', 'h8', 'right')} "
              f"(correct letter sum: 1)")

        print("5. Going out with an empty bag does not end the game")
        game = new_game("cat", "zeeeeee", bag={})
        quiet(game.play_move, 1, "cat", "h8", "right")
        print(f"   rack {game.player_1_hand}, bag empty: game_over={game.game_over}, "
              f"next player={game.current_player}")

        print("6. An exchange is allowed with fewer than 7 tiles in the bag")
        game = new_game("qqqqqqq", "eeeeeee", bag={"a": 1})
        print(f"   bag holds 1 tile, exchanging 1: accepted={game.exchange_tiles(1, ['q'])}")

        if args.full_dictionary:
            print("7. Per-game and per-move overhead with the full old word list")
            shutil.copyfile(args.full_dictionary, "word_dictionary.csv")
            start = time.perf_counter()
            games = [quiet(game_API.Scrabble_Game) for _ in range(5)]
            construct = (time.perf_counter() - start) / len(games)
            games[0].initialize_game()
            start = time.perf_counter()
            for _ in range(20):
                copy.deepcopy(games[0])
            deep = (time.perf_counter() - start) / 20
            print(f"   {len(games[0].words_set):,} words; Scrabble_Game() {construct * 1e3:.0f} ms; "
                  f"copy.deepcopy(game) {deep * 1e3:.0f} ms (self_play.py does one per move)")
    finally:
        os.chdir(LEGACY)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
