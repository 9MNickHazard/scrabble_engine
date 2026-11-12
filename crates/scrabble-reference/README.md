# scrabble-reference

An independent, deliberately simple implementation of the placement rules in
[`docs/RULES.md`](../../docs/RULES.md). It exists only to test `scrabble-core`: on many random
positions the engine's move generator must produce exactly the set of plays, with exactly the
scores, that this crate does.

It was written from `docs/RULES.md` alone, without reading `scrabble-core`, `legacy/` or
`tools/legacy_audit/`, and it shares no code with them. It uses no dependencies and favours
obviously correct code over speed: no trie, DAWG or GADDAG.

## API

```rust
pub const N: usize = 15;
pub struct PlacedTile { pub row: u8, pub col: u8, pub letter: u8, pub blank: bool }
pub struct RefPlay { pub tiles: Vec<PlacedTile> /* sorted by (row, col) */, pub score: i32 }
pub struct RefBoard { pub cells: [[Option<(u8, bool)>; N]; N] }   // [row][col]
// RefBoard::empty, from_text, to_text, place, is_empty, transposed
pub struct RefLexicon;  // from_words, from_file, contains, len, is_empty

pub fn check_placement(board: &RefBoard, tiles: &[PlacedTile], lexicon: &RefLexicon) -> Result<i32, String>;
pub fn legal_plays(board: &RefBoard, rack: &str, lexicon: &RefLexicon) -> Vec<RefPlay>;
pub fn legal_plays_exhaustive(board: &RefBoard, rack: &str, lexicon: &RefLexicon) -> Vec<RefPlay>;

pub fn tile_value(letter: u8) -> i32;
pub fn premium_at(row: u8, col: u8) -> Premium;
pub fn square_name(row: u8, col: u8) -> String;              // (7, 7) -> "H8"
pub fn parse_square(name: &str) -> Option<(u8, u8)>;
pub fn parse_placed_tile(text: &str) -> Result<PlacedTile, String>;   // "H8=C", "J8=t" (blank)
pub fn placed_tile_text(tile: &PlacedTile) -> String;

pub struct Case { pub name: String, pub words: Vec<String>, pub board: RefBoard,
                  pub tiles: Vec<PlacedTile>, pub expect: Option<i32> }
pub fn parse_cases(text: &str) -> Result<Vec<Case>, String>;
pub const SCORING_CASES: &str;   // testdata/scoring_cases.txt, built in
```

- `check_placement` judges one placement rule by rule (count, squares, one line, no gaps,
  connection, main word, cross-words) and scores it. It looks at no rack.
- `legal_plays` slides every lexicon word along every row and column, keeps the spans that match
  the board, are maximal runs, cover 1 to 7 empty squares including one next to an old tile (H8 on
  an empty board) and can be supplied from the rack, enumerates every choice of which new letters
  are blanks, and passes each candidate to `check_placement`, which decides. Racks are upper-case
  letters with `?` for a blank; any other character panics.
- `legal_plays_exhaustive` puts rack tiles on every set of empty squares in a row or column with
  every letter for each blank and keeps what `check_placement` accepts. Racks of about 4 tiles at
  most.

## Tests

```
cargo test -p scrabble-reference
```

- `tests/scoring_cases.rs`: every hand-checked case, its transpose, the reason each illegal case
  is rejected, and `legal_plays` (plus the brute force for small cases) agreeing with every case.
- `tests/cross_check.rs`: `legal_plays` equals `legal_plays_exhaustive` on positions built by
  playing random legal plays with a small made-up lexicon (racks of 1 to 4 tiles, up to 2 blanks),
  every play passes `check_placement`, and transposing the board transposes the plays.
- `tests/basics.rs`: board text and square name round trips, premium squares, tile values,
  lexicon loading.
- `tests/real_lexicon.rs`: uses the real word list from `SCRABBLE_LEXICON`, or
  `data/lexicon/NSWL2023.txt` under the workspace root; skipped with a printed note if missing.
  Opening answers (best opening for `AEINRST` is 66), brute force agreement on small racks, and a
  timing printout (`-- --nocapture` to see it).

Release-mode timing of `legal_plays` with the full lexicon:

```
cargo run --release -p scrabble-reference --example time_legal_plays -- [games] [seed]
```

## Case-file format

`testdata/scoring_cases.txt`, read by `parse_cases`:

```
# comment lines start with '#'
case: opening-cat
words: CAT CATS AT
board:
...............
(15 lines of 15 characters: '.' empty, A-Z a tile, a-z a blank standing for that letter)
place: H8=C I8=A J8=T
expect: 10
# arithmetic: (3+1+1) x2 for the centre double word = 10
```

A case starts with `case:` and ends at a blank line. `words:` is the entire lexicon for the case
(it may be repeated; the lists are joined). `place:` lists placed tiles as `<square>=<letter>`,
lower case for a blank. `expect:` is an integer score or `illegal`. Every other key appears once
per case and case names are unique. Each expected score was worked out by hand from the rules
first, with the arithmetic in a comment under the case.
