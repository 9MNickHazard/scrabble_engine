"""Independent brute-force reference move generator for standard 15x15 Scrabble.

Written from the rules, not ported from the legacy Rust crate. It is deliberately
simple rather than fast:

  for every row and every column,
    for every maximal span of squares that could hold the main word
    (the squares just before and after the span are empty or off the board),
      match every lexicon word of that length against the span
      (existing tiles must match, empty squares take rack letters),
      then check rack supply (blanks cover shortfalls), connectivity
      (first move covers h8, later moves touch an existing tile) and every
      cross-word of 2+ letters.

A move is the set of placed tiles {(square, letter, is_blank)}. Every legal way
of using blanks is a separate move (e.g. with rack "A?" the word "AA" can be
played with the blank on either square).

Squares are named like the legacy code: row letter a..o (a = top) followed by
column number 1..15 (1 = left). h8 is the centre.

Scoring follows the standard rules: letter and word premiums count only under
newly placed tiles; blanks (new or already on the board) are worth 0; every
word of 2+ letters formed by the play is scored; +50 when 7 tiles are placed.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from itertools import combinations

ROWS = "abcdefghijklmno"
SIZE = 15
CENTER = (7, 7)  # h8
BLANK = "?"

TILE_VALUES = {
    "a": 1, "b": 3, "c": 3, "d": 2, "e": 1, "f": 4, "g": 2, "h": 4, "i": 1,
    "j": 8, "k": 5, "l": 1, "m": 3, "n": 1, "o": 1, "p": 3, "q": 10, "r": 1,
    "s": 1, "t": 1, "u": 1, "v": 4, "w": 4, "x": 8, "y": 4, "z": 10,
}

BAG_COUNTS = {
    "a": 9, "b": 2, "c": 2, "d": 4, "e": 12, "f": 2, "g": 3, "h": 2, "i": 9,
    "j": 1, "k": 1, "l": 4, "m": 2, "n": 6, "o": 8, "p": 2, "q": 1, "r": 6,
    "s": 4, "t": 6, "u": 4, "v": 2, "w": 2, "x": 1, "y": 2, "z": 1, BLANK: 2,
}

# Standard premium layout. T = triple word, D = double word,
# t = triple letter, d = double letter. Row 0 is row "a".
PREMIUM_ROWS = [
    "T..d...T...d..T",
    ".D...t...t...D.",
    "..D...d.d...D..",
    "d..D...d...D..d",
    "....D.....D....",
    ".t...t...t...t.",
    "..d...d.d...d..",
    "T..d...D...d..T",
    "..d...d.d...d..",
    ".t...t...t...t.",
    "....D.....D....",
    "d..D...d...D..d",
    "..D...d.d...D..",
    ".D...t...t...D.",
    "T..d...T...d..T",
]


def sq_name(r: int, c: int) -> str:
    return f"{ROWS[r]}{c + 1}"


def parse_sq(s: str) -> tuple[int, int]:
    return ROWS.index(s[0]), int(s[1:]) - 1


# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------

class Lexicon:
    def __init__(self, words):
        self.words = frozenset(words)
        by_len: dict[int, list[str]] = {}
        for w in self.words:
            by_len.setdefault(len(w), []).append(w)
        # one newline-joined blob per length, searched with a multiline regex
        self._blob = {n: "\n".join(sorted(ws)) for n, ws in by_len.items()}

    def __contains__(self, w: str) -> bool:
        return w in self.words

    def __len__(self) -> int:
        return len(self.words)

    def match(self, parts: list[str]) -> list[str]:
        """parts: one regex atom per square (a literal letter or a char class)."""
        blob = self._blob.get(len(parts))
        if not blob:
            return []
        return re.findall("^" + "".join(parts) + "$", blob, re.M)


def load_lexicon(path: str) -> Lexicon:
    """.csv: legacy format (header row, word in the first field, trimmed and
    lowercased, exactly what the legacy Rust loader keeps).
    anything else: first whitespace-separated token of each line (NSWL2023.txt)."""
    words = set()
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rows = csv.reader(f)
            next(rows, None)  # header
            for row in rows:
                if row and row[0].strip():
                    words.add(row[0].strip().lower())
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                tok = line.split()
                if tok:
                    words.add(tok[0].lower())
    return Lexicon(words)


# --------------------------------------------------------------------------
# Board helpers. A board is a dict: square name -> (letter, is_blank).
# --------------------------------------------------------------------------

def to_grid(board: dict) -> list[list]:
    grid = [[None] * SIZE for _ in range(SIZE)]
    for s, (letter, is_blank) in board.items():
        r, c = parse_sq(s)
        grid[r][c] = (letter, bool(is_blank))
    return grid


def _get(grid, r, c):
    if 0 <= r < SIZE and 0 <= c < SIZE:
        return grid[r][c]
    return None


def _run(grid, r, c, dr, dc, override=None):
    """Maximal run of tiles through (r, c) along (dr, dc).
    override: dict (r, c) -> (letter, is_blank) for tiles being placed."""
    override = override or {}

    def tile(rr, cc):
        if (rr, cc) in override:
            return override[(rr, cc)]
        return _get(grid, rr, cc)

    while tile(r - dr, c - dc) is not None:
        r, c = r - dr, c - dc
    cells = []
    while tile(r, c) is not None:
        cells.append(((r, c), tile(r, c)))
        r, c = r + dr, c + dc
    return cells


def score_placement(grid, placement) -> int:
    new = {parse_sq(s): (letter, bool(b)) for s, letter, b in placement}
    rows = {r for r, _ in new}
    cols = {c for _, c in new}
    lines = []  # (dr, dc, cells-to-start-from)
    if len(new) == 1:
        (rc,) = new
        lines = [(0, 1, rc), (1, 0, rc)]
    elif len(rows) == 1:
        first = min(new)
        lines = [(0, 1, first)] + [(1, 0, rc) for rc in new]
    elif len(cols) == 1:
        first = min(new)
        lines = [(1, 0, first)] + [(0, 1, rc) for rc in new]
    else:
        raise ValueError("tiles not in one line")
    total = 0
    for dr, dc, (r, c) in lines:
        cells = _run(grid, r, c, dr, dc, new)
        if len(cells) < 2:
            continue
        word_sum, word_mult = 0, 1
        for (rr, cc), (letter, is_blank) in cells:
            v = 0 if is_blank else TILE_VALUES[letter]
            if (rr, cc) in new:
                p = PREMIUM_ROWS[rr][cc]
                if p == "d":
                    v *= 2
                elif p == "t":
                    v *= 3
                elif p == "D":
                    word_mult *= 2
                elif p == "T":
                    word_mult *= 3
            word_sum += v
        total += word_sum * word_mult
    if len(new) == 7:
        total += 50
    return total


def words_formed(grid, placement) -> list[str]:
    """All words of 2+ letters formed by a placement (main word first)."""
    new = {parse_sq(s): (letter, bool(b)) for s, letter, b in placement}
    rows = {r for r, _ in new}
    if len(new) == 1:
        (rc,) = new
        lines = [(0, 1, rc), (1, 0, rc)]
    elif len(rows) == 1:
        lines = [(0, 1, min(new))] + [(1, 0, rc) for rc in sorted(new)]
    else:
        lines = [(1, 0, min(new))] + [(0, 1, rc) for rc in sorted(new)]
    out = []
    for dr, dc, (r, c) in lines:
        cells = _run(grid, r, c, dr, dc, new)
        if len(cells) >= 2:
            out.append("".join(t[0] for _, t in cells))
    return out


# --------------------------------------------------------------------------
# Move generation
# --------------------------------------------------------------------------

def _blank_choices(needed: list[str], rack_counts: Counter, n_blanks: int):
    """Every set of indices into `needed` that can be blanks, given the rack."""
    for b in range(n_blanks + 1):
        for idx in combinations(range(len(needed)), b):
            chosen = set(idx)
            real = Counter(l for i, l in enumerate(needed) if i not in chosen)
            if all(real[l] <= rack_counts.get(l, 0) for l in real):
                yield chosen


def legal_moves(board: dict, rack, lex: Lexicon) -> dict:
    """Return {placement: score}; placement = frozenset of (square, letter, is_blank)."""
    grid = to_grid(board)
    rack = list(rack)
    rack_counts = Counter(rack)
    n_blanks = rack_counts.pop(BLANK, 0)
    n_rack = len(rack)
    board_empty = all(grid[r][c] is None for r in range(SIZE) for c in range(SIZE))
    if n_blanks:
        free = "[a-z]"
    else:
        free = "[" + "".join(sorted(rack_counts)) + "]"

    moves: dict = {}
    for horizontal in (True, False):
        dr, dc = (0, 1) if horizontal else (1, 0)   # main-word direction
        pr, pc = (1, 0) if horizontal else (0, 1)   # perpendicular direction
        for line in range(SIZE):
            cells = [(line, i) if horizontal else (i, line) for i in range(SIZE)]
            tiles = [grid[r][c] for r, c in cells]
            for i in range(SIZE):
                if i > 0 and tiles[i - 1] is not None:
                    continue  # span would not be maximal on the left
                for j in range(i + 1, SIZE):  # main word has 2+ squares
                    if j < SIZE - 1 and tiles[j + 1] is not None:
                        continue  # span would not be maximal on the right
                    empties = [k for k in range(i, j + 1) if tiles[k] is None]
                    if not empties or len(empties) > n_rack:
                        continue
                    span = cells[i:j + 1]
                    if board_empty:
                        if CENTER not in span:
                            continue
                    else:
                        touches = len(empties) < len(span) or any(
                            _get(grid, cells[k][0] - pr, cells[k][1] - pc) is not None
                            or _get(grid, cells[k][0] + pr, cells[k][1] + pc) is not None
                            for k in empties
                        )
                        if not touches:
                            continue
                    parts = [tiles[k][0] if tiles[k] is not None else free for k in range(i, j + 1)]
                    for word in lex.match(parts):
                        needed = [word[k - i] for k in empties]
                        need = Counter(needed)
                        short = sum(max(0, n - rack_counts.get(l, 0)) for l, n in need.items())
                        if short > n_blanks:
                            continue
                        ok = True
                        for k in empties:
                            r, c = cells[k]
                            cw = _run(grid, r, c, pr, pc, {(r, c): (word[k - i], False)})
                            if len(cw) >= 2 and "".join(t[0] for _, t in cw) not in lex:
                                ok = False
                                break
                        if not ok:
                            continue
                        for chosen in _blank_choices(needed, rack_counts, n_blanks):
                            placement = frozenset(
                                (sq_name(*cells[k]), word[k - i], n in chosen)
                                for n, k in enumerate(empties)
                            )
                            if placement not in moves:
                                moves[placement] = score_placement(grid, placement)
    return moves


def check_placement(board: dict, rack, placement, lex: Lexicon) -> tuple[bool, str]:
    """Judge one placement directly from the rules (independent of legal_moves).
    Returns (is_legal, reason)."""
    grid = to_grid(board)
    new = {}
    for s, letter, is_blank in placement:
        if letter not in TILE_VALUES:
            return False, f"bad letter {letter!r}"
        rc = parse_sq(s)
        if rc in new:
            return False, f"two tiles on {s}"
        new[rc] = (letter, bool(is_blank))
    if not new:
        return False, "no tiles"
    for (r, c) in new:
        if grid[r][c] is not None:
            return False, f"{sq_name(r, c)} already occupied"
    have = Counter(rack)
    use = Counter(BLANK if b else l for l, b in new.values())
    for t, n in use.items():
        if have.get(t, 0) < n:
            return False, f"rack lacks {n} x {t!r}"
    rows = {r for r, _ in new}
    cols = {c for _, c in new}
    if len(rows) > 1 and len(cols) > 1:
        return False, "tiles not in one row or column"
    if len(new) > 1:
        if len(rows) == 1:
            (r,) = rows
            line = [(r, c) for c in range(min(cols), max(cols) + 1)]
        else:
            (c,) = cols
            line = [(r, c) for r in range(min(rows), max(rows) + 1)]
        for rc in line:
            if rc not in new and grid[rc[0]][rc[1]] is None:
                return False, f"gap at {sq_name(*rc)}"
    board_empty = all(grid[r][c] is None for r in range(SIZE) for c in range(SIZE))
    if board_empty:
        if CENTER not in new:
            return False, "first move does not cover h8"
        if len(new) < 2:
            return False, "first move needs 2+ tiles"
    else:
        touches = any(
            _get(grid, r + dr, c + dc) is not None
            for (r, c) in new for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
        )
        if not touches:
            return False, "does not touch an existing tile"
    words = words_formed(grid, placement)
    if not words:
        return False, "forms no word of 2+ letters"
    for w in words:
        if w not in lex:
            return False, f"{w!r} not in lexicon"
    return True, "ok"


# --------------------------------------------------------------------------
# Random positions
# --------------------------------------------------------------------------

def full_bag(include_blanks: bool = True) -> list[str]:
    bag = []
    for letter, n in BAG_COUNTS.items():
        if letter == BLANK and not include_blanks:
            continue
        bag.extend([letter] * n)
    return bag


def placement_key(p) -> tuple:
    return tuple(sorted(p))


def random_game(lex: Lexicon, rng: random.Random, plies: int, *, blanks_in_bag: bool = True,
                top_n: int = 10):
    """Play `plies` moves from an empty board, each chosen uniformly from the
    reference's top_n highest-scoring legal moves. Returns (board, remaining_bag)."""
    bag = full_bag(blanks_in_bag)
    rng.shuffle(bag)
    board: dict = {}
    rack = [bag.pop() for _ in range(7)]
    for _ in range(plies):
        moves = legal_moves(board, rack, lex)
        if not moves:
            bag.extend(rack)
            rng.shuffle(bag)
            rack = [bag.pop() for _ in range(min(7, len(bag)))]
            continue
        ranked = sorted(moves.items(), key=lambda kv: (-kv[1], placement_key(kv[0])))
        placement, _ = rng.choice(ranked[:top_n])
        for s, letter, is_blank in placement:
            board[s] = (letter, is_blank)
            rack.remove(BLANK if is_blank else letter)
        while len(rack) < 7 and bag:
            rack.append(bag.pop())
    return board, bag


def board_to_text(board: dict) -> str:
    """15 lines; '.' empty, lowercase letter = tile, uppercase = blank tile."""
    grid = to_grid(board)
    lines = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            t = grid[r][c]
            row.append("." if t is None else (t[0].upper() if t[1] else t[0]))
        lines.append(f"{ROWS[r]} " + "".join(row))
    return "\n".join(lines)
