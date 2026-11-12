# Rules the engine implements

Two-player Scrabble under the NASPA rules, minus the parts that only exist for humans at a table
(clocks, challenges, drawing for first turn). Section numbers in brackets refer to the NASPA
Official Tournament Rules, Player Edition, 2017-01-20. This document is the specification: the
engine (`crates/scrabble-core`) and the independent reference used to test it
(`crates/scrabble-reference`) are both written from it.

## Board and coordinates

The board is 15 x 15. Internally a square is `(row, col)` with both counted from 0, row 0 at the
top and column 0 at the left. Square names use the standard form: column letter `A`-`O`, then row
number `1`-`15`. `A1` is `(0, 0)`, `O1` is `(0, 14)`, `A15` is `(14, 0)`, and the centre square
`H8` is `(7, 7)`.

Premium squares. `=` triple word, `-` double word, `"` triple letter, `'` double letter. The centre
square is a double word square.

```
     A B C D E F G H I J K L M N O
 1   = . . ' . . . = . . . ' . . =
 2   . - . . . " . . . " . . . - .
 3   . . - . . . ' . ' . . . - . .
 4   ' . . - . . . ' . . . - . . '
 5   . . . . - . . . . . - . . . .
 6   . " . . . " . . . " . . . " .
 7   . . ' . . . ' . ' . . . ' . .
 8   = . . ' . . . - . . . ' . . =
 9   . . ' . . . ' . ' . . . ' . .
10   . " . . . " . . . " . . . " .
11   . . . . - . . . . . - . . . .
12   ' . . - . . . ' . . . - . . '
13   . . - . . . ' . ' . . . - . .
14   . - . . . " . . . " . . . - .
15   = . . ' . . . = . . . ' . . =
```

## Tiles

100 tiles. `?` is the blank.

| Tile | Count | Value | Tile | Count | Value | Tile | Count | Value |
|---|---|---|---|---|---|---|---|---|
| A | 9 | 1 | J | 1 | 8 | S | 4 | 1 |
| B | 2 | 3 | K | 1 | 5 | T | 6 | 1 |
| C | 2 | 3 | L | 4 | 1 | U | 4 | 1 |
| D | 4 | 2 | M | 2 | 3 | V | 2 | 4 |
| E | 12 | 1 | N | 6 | 1 | W | 2 | 4 |
| F | 2 | 4 | O | 8 | 1 | X | 1 | 8 |
| G | 3 | 2 | P | 2 | 3 | Y | 2 | 4 |
| H | 2 | 4 | Q | 1 | 10 | Z | 1 | 10 |
| I | 9 | 1 | R | 6 | 1 | ? | 2 | 0 |

A blank is given a letter when it is played. It keeps that letter and scores 0 for the rest of the
game, in every word it is part of [IV.A].

## Start

Each player draws 7 tiles. Player 0 moves first.

## A turn

On a turn the player does exactly one of: place tiles, exchange, or pass [IV.A].

### Placing tiles

A placement puts 1 to 7 tiles from the rack onto empty squares. It is legal when all of these hold:

1. **One line.** All placed tiles are in one row or all in one column.
2. **No gaps.** After the placement, every square from the first placed tile to the last placed
   tile along that line is occupied (by a placed tile or a tile that was already there).
3. **Main word.** The main word is the longest unbroken run of occupied squares along the line of
   play that contains the placed tiles. It includes any tiles already on the board directly before
   the first placed tile or after the last one. It must have at least two letters and be in the
   lexicon.
4. **Cross-words.** For each placed tile, take the longest unbroken run of occupied squares through
   it in the other direction. If that run has two or more letters it is a cross-word and must be in
   the lexicon.
5. **Connection.** If the board is empty, one of the placed tiles is on `H8`. Otherwise at least one
   placed tile is next to (shares an edge with) a tile that was already on the board.

Words read left to right or top to bottom [IV.A].

**A single placed tile** has no line of its own. It is legal when the words it forms across and
down (each of two or more letters) are all in the lexicon, at least one such word exists, and rule
5 holds. One tile on a square is one play, whichever direction is used to describe it.

**Two placements are the same play** exactly when they put the same tiles on the same squares,
counting a blank standing for a letter as different from the real letter. Playing a real `E` and
playing a blank as `E` on the same square are different plays with different scores.

Every play must be valid: the engine has no phony plays and no challenges.

### Scoring a placement

[IV.A]

- A **word's score** is the sum of its letters' values, where a letter on a double or triple letter
  square *covered this turn* counts double or triple, and the total is then doubled or tripled for
  each double or triple word square *covered this turn* in that word. Blanks are worth 0, including
  under a premium square. A premium square under a tile played on an earlier turn does nothing.
- The **play's score** is the main word's score, plus the score of every cross-word, plus 50 if
  the play placed 7 tiles.

A single placed tile scores every word of two or more letters it forms (one or two words).

After a placement the player draws tiles from the bag until the rack holds 7 or the bag is empty.

### Exchanging

Allowed only when the bag holds at least 7 tiles [IV.E]. The player picks 1 to 7 tiles from the
rack, draws the same number from the bag, and only then returns the picked tiles to the bag, which
is mixed. An exchange scores 0. Two exchanges are the same move when they return the same multiset
of tiles.

### Passing

Always allowed. Scores 0.

## End of the game

The game ends in one of two ways.

1. **Going out** [V.A]. A player places tiles, the rack is now empty and the bag is empty. That
   player's score increases by twice the total value of the tiles on the opponent's rack [V.G].
   The opponent's score does not change.
2. **Six scoreless turns** [V.B]. Six turns in a row, counting both players, were passes or
   exchanges. Each player's score is reduced by the total value of the tiles on their own rack
   [V.G]. Any placement of tiles resets the count, even one that scores 0. (The NASPA text counts
   "passes, exchanges, challenges, or illegal plays"; the engine has only the first two.)

The player with more points wins. Equal scores are a tie, worth half a win to each [V.H]. The
**spread** is a player's final score minus the opponent's.

## Invariants

These hold after every turn and are checked by the fuzz tests.

- Tiles on the board + tiles on both racks + tiles in the bag = 100, tile type by tile type (a
  blank on the board counts as a blank).
- Each player's score equals the sum of the scores of their turns plus the end-of-game adjustment.
- While the game is running and the bag is not empty, both racks hold 7 tiles.
- Every game ends: passes and exchanges are bounded by the six-turn rule, and placements by the
  number of tiles.

## The bag and reproducible games

A game is created from a seed. The seed fixes the order of the tiles in the bag, and draws take
tiles from the front in that order, so two games with the same seed see the same tiles in the same
order as long as the same number of tiles has been drawn. An exchange draws first, then puts the
returned tiles back at positions chosen by the game's own random generator. The same seed and the
same moves always give the same game. Paired evaluation uses this: each seed is played twice with
the two players swapping seats.

## Canonical forms used in tests

- **A play** is the set of placed tiles, each `(row, col, letter, is_blank)`, sorted by row then
  column, plus its score.
- **A board in text** is 15 lines of 15 characters: `.` for an empty square, an upper-case letter
  for a normal tile, a lower-case letter for a blank standing for that letter.
- **A rack in text** is its tiles as upper-case letters, with `?` for a blank.
- **A placed tile in text** is `<square>=<letter>`, for example `H8=C` or `J8=t` (lower case: a
  blank played as T).

## Not modelled

Clocks and overtime penalties, challenges, courtesy rules, and drawing tiles to decide who goes
first.
