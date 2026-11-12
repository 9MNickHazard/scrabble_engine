//! Tiles, letter values, premium squares and square names, transcribed from `docs/RULES.md`.

use std::fmt;

/// The board is `N` x `N` squares.
pub const N: usize = 15;

/// One tile put on the board by a play.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PlacedTile {
    /// 0 is the top row (row number 1 in square names).
    pub row: u8,
    /// 0 is the left column (column letter `A` in square names).
    pub col: u8,
    /// The letter the tile shows, `b'A'..=b'Z'`. For a blank, the letter it stands for.
    pub letter: u8,
    /// True when the tile is a blank.
    pub blank: bool,
}

/// A legal placement: its tiles sorted by `(row, col)`, and its score.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct RefPlay {
    pub tiles: Vec<PlacedTile>,
    pub score: i32,
}

/// The value of a letter tile, from the table in RULES.md "Tiles". A blank is worth 0, but that is
/// decided by the caller: this function only takes letters `b'A'..=b'Z'` and panics on anything else.
pub fn tile_value(letter: u8) -> i32 {
    match letter {
        b'A' => 1,
        b'B' => 3,
        b'C' => 3,
        b'D' => 2,
        b'E' => 1,
        b'F' => 4,
        b'G' => 2,
        b'H' => 4,
        b'I' => 1,
        b'J' => 8,
        b'K' => 5,
        b'L' => 1,
        b'M' => 3,
        b'N' => 1,
        b'O' => 1,
        b'P' => 3,
        b'Q' => 10,
        b'R' => 1,
        b'S' => 1,
        b'T' => 1,
        b'U' => 1,
        b'V' => 4,
        b'W' => 4,
        b'X' => 8,
        b'Y' => 4,
        b'Z' => 10,
        _ => panic!("tile_value: {letter} is not a letter b'A'..=b'Z'"),
    }
}

/// What a square does to the tile placed on it this turn.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Premium {
    Plain,
    DoubleLetter,
    TripleLetter,
    DoubleWord,
    TripleWord,
}

/// The premium-square diagram from RULES.md with the spaces removed, one string per row, row 1 first.
/// `=` triple word, `-` double word, `"` triple letter, `'` double letter, `.` plain.
/// (Each literal must be exactly 15 bytes or this does not compile.)
const LAYOUT: [&[u8; N]; N] = [
    b"=..'...=...'..=",     // 1
    b".-...\"...\"...-.",   // 2
    b"..-...'.'...-..",     // 3
    b"'..-...'...-..'",     // 4
    b"....-.....-....",     // 5
    b".\"...\"...\"...\".", // 6
    b"..'...'.'...'..",     // 7
    b"=..'...-...'..=",     // 8
    b"..'...'.'...'..",     // 9
    b".\"...\"...\"...\".", // 10
    b"....-.....-....",     // 11
    b"'..-...'...-..'",     // 12
    b"..-...'.'...-..",     // 13
    b".-...\"...\"...-.",   // 14
    b"=..'...=...'..=",     // 15
];

/// The premium square at `(row, col)`. Panics if the square is off the board.
pub fn premium_at(row: u8, col: u8) -> Premium {
    match LAYOUT[row as usize][col as usize] {
        b'=' => Premium::TripleWord,
        b'-' => Premium::DoubleWord,
        b'"' => Premium::TripleLetter,
        b'\'' => Premium::DoubleLetter,
        b'.' => Premium::Plain,
        other => unreachable!("bad layout byte {other}"),
    }
}

/// `"H8"` for `(7, 7)`: column letter `A`-`O`, then row number `1`-`15`. Panics off the board.
pub fn square_name(row: u8, col: u8) -> String {
    assert!(
        (row as usize) < N && (col as usize) < N,
        "square_name: ({row}, {col}) is off the board"
    );
    format!("{}{}", (b'A' + col) as char, row as u32 + 1)
}

/// The inverse of [`square_name`]. Only the canonical form is accepted: an upper-case column letter
/// `A`-`O` followed by a row number `1`-`15` without a leading zero or sign.
pub fn parse_square(name: &str) -> Option<(u8, u8)> {
    let bytes = name.as_bytes();
    if bytes.len() < 2 || bytes.len() > 3 {
        return None;
    }
    let col_letter = bytes[0];
    if !(b'A'..=b'O').contains(&col_letter) {
        return None;
    }
    let digits = &bytes[1..];
    if !digits.iter().all(|b| b.is_ascii_digit()) || digits[0] == b'0' {
        return None;
    }
    let mut number: u32 = 0;
    for &d in digits {
        number = number * 10 + (d - b'0') as u32;
    }
    if !(1..=15).contains(&number) {
        return None;
    }
    Some(((number - 1) as u8, col_letter - b'A'))
}

/// Parse a placed tile in the text form of RULES.md "Canonical forms": `H8=C`, or `J8=t` for a
/// blank played as T.
pub fn parse_placed_tile(text: &str) -> Result<PlacedTile, String> {
    let (square, letter) = text
        .split_once('=')
        .ok_or_else(|| format!("placed tile {text:?} has no '='"))?;
    let (row, col) = parse_square(square)
        .ok_or_else(|| format!("placed tile {text:?}: bad square {square:?}"))?;
    let letter_bytes = letter.as_bytes();
    if letter_bytes.len() != 1 {
        return Err(format!(
            "placed tile {text:?}: expected one letter after '='"
        ));
    }
    let (letter, blank) = match letter_bytes[0] {
        b @ b'A'..=b'Z' => (b, false),
        b @ b'a'..=b'z' => (b.to_ascii_uppercase(), true),
        _ => {
            return Err(format!(
                "placed tile {text:?}: the letter must be A-Z, or a-z for a blank"
            ))
        }
    };
    Ok(PlacedTile {
        row,
        col,
        letter,
        blank,
    })
}

/// The text form of a placed tile: `H8=C`, or `J8=t` for a blank played as T.
pub fn placed_tile_text(tile: &PlacedTile) -> String {
    let letter = if tile.blank {
        tile.letter.to_ascii_lowercase()
    } else {
        tile.letter
    };
    format!("{}={}", square_name(tile.row, tile.col), letter as char)
}

impl fmt::Display for PlacedTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&placed_tile_text(self))
    }
}

impl fmt::Display for RefPlay {
    /// `H8=C I8=A J8=T (10)`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for tile in &self.tiles {
            write!(f, "{tile} ")?;
        }
        write!(f, "({})", self.score)
    }
}

/// The tiles of a rack in text form: upper-case letters, `?` for a blank. Panics on anything else.
pub(crate) fn parse_rack(rack: &str) -> Vec<u8> {
    rack.bytes()
        .inspect(|&b| {
            assert!(
                b.is_ascii_uppercase() || b == b'?',
                "rack {rack:?}: tiles must be A-Z or '?' for a blank"
            );
        })
        .collect()
}
