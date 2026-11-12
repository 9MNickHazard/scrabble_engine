//! The 15x15 board.
//!
//! A square is one byte: 0 if empty, otherwise `letter + 1` in the low five bits, with
//! [`BLANK_BIT`] set when the tile is a blank standing for that letter. Squares are indexed
//! `row * 15 + col`, row 0 at the top. See docs/RULES.md for coordinates and the premium layout.

use crate::tiles::{BLANK, TILE_VALUES};

pub const N: usize = 15;
pub const SQUARES: usize = N * N;
pub const CENTER: usize = 7 * N + 7;

pub const EMPTY: u8 = 0;
pub const BLANK_BIT: u8 = 0x80;
const LETTER_BITS: u8 = 0x1F;

/// `=` triple word, `-` double word, `"` triple letter, `'` double letter.
const LAYOUT: [&[u8; N]; N] = [
    b"=..'...=...'..=",
    b".-...\"...\"...-.",
    b"..-...'.'...-..",
    b"'..-...'...-..'",
    b"....-.....-....",
    b".\"...\"...\"...\".",
    b"..'...'.'...'..",
    b"=..'...-...'..=",
    b"..'...'.'...'..",
    b".\"...\"...\"...\".",
    b"....-.....-....",
    b"'..-...'...-..'",
    b"..-...'.'...-..",
    b".-...\"...\"...-.",
    b"=..'...=...'..=",
];

const fn multipliers(letter: bool) -> [u8; SQUARES] {
    let mut out = [1u8; SQUARES];
    let mut row = 0;
    while row < N {
        let mut col = 0;
        while col < N {
            out[row * N + col] = match (LAYOUT[row][col], letter) {
                (b'\'', true) | (b'-', false) => 2,
                (b'"', true) | (b'=', false) => 3,
                _ => 1,
            };
            col += 1;
        }
        row += 1;
    }
    out
}

/// Letter multiplier of each square (1, 2 or 3).
pub const LETTER_MULT: [u8; SQUARES] = multipliers(true);
/// Word multiplier of each square (1, 2 or 3).
pub const WORD_MULT: [u8; SQUARES] = multipliers(false);

/// Square byte for a tile: `letter` is 0..=25, `blank` says the tile is a blank standing for it.
#[inline]
pub const fn make_square(letter: u8, blank: bool) -> u8 {
    (letter + 1) | if blank { BLANK_BIT } else { 0 }
}

/// Letter 0..=25 on an occupied square.
#[inline]
pub const fn square_letter(square: u8) -> u8 {
    (square & LETTER_BITS) - 1
}

#[inline]
pub const fn square_is_blank(square: u8) -> bool {
    square & BLANK_BIT != 0
}

/// Points the tile on an occupied square is worth (0 for a blank).
#[inline]
pub const fn square_value(square: u8) -> i32 {
    if square & BLANK_BIT != 0 {
        0
    } else {
        TILE_VALUES[((square & LETTER_BITS) - 1) as usize] as i32
    }
}

/// Tile type (0..=25, or [`BLANK`]) that an occupied square took from a rack.
#[inline]
pub const fn square_tile(square: u8) -> u8 {
    if square & BLANK_BIT != 0 {
        BLANK
    } else {
        (square & LETTER_BITS) - 1
    }
}

/// `"H8"` for (7, 7): column letter then row number.
pub fn square_name(row: usize, col: usize) -> String {
    format!("{}{}", (b'A' + col as u8) as char, row + 1)
}

/// Parse a square name such as `H8` or `o15` into (row, col).
pub fn parse_square(name: &str) -> Option<(usize, usize)> {
    let mut chars = name.chars();
    let col = match chars.next()? {
        c @ 'A'..='O' => c as usize - 'A' as usize,
        c @ 'a'..='o' => c as usize - 'a' as usize,
        _ => return None,
    };
    let row: usize = chars.as_str().parse().ok()?;
    (1..=N).contains(&row).then(|| (row - 1, col))
}

#[derive(Clone, PartialEq, Eq)]
pub struct Board {
    pub cells: [u8; SQUARES],
    tiles: u8,
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

impl Board {
    pub const fn new() -> Self {
        Board { cells: [EMPTY; SQUARES], tiles: 0 }
    }

    /// Number of tiles on the board.
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.tiles as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tiles == 0
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.cells[row * N + col]
    }

    /// Put a tile on an empty square.
    #[inline]
    pub fn set(&mut self, index: usize, square: u8) {
        debug_assert!(self.cells[index] == EMPTY && square != EMPTY);
        self.cells[index] = square;
        self.tiles += 1;
    }

    /// The board flipped across its main diagonal: rows become columns.
    pub fn transposed(&self) -> Board {
        let mut out = Board { cells: [EMPTY; SQUARES], tiles: self.tiles };
        for row in 0..N {
            for col in 0..N {
                out.cells[col * N + row] = self.cells[row * N + col];
            }
        }
        out
    }

    /// Parse the 15-line text form: `.` empty, upper case a tile, lower case a blank.
    pub fn from_text(text: &str) -> Result<Board, String> {
        let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        if lines.len() != N {
            return Err(format!("expected {N} board lines, found {}", lines.len()));
        }
        let mut board = Board::new();
        for (row, line) in lines.iter().enumerate() {
            let chars: Vec<char> = line.chars().collect();
            if chars.len() != N {
                return Err(format!("board line {} has {} characters, expected {N}", row + 1, chars.len()));
            }
            for (col, &c) in chars.iter().enumerate() {
                match c {
                    '.' => {}
                    'A'..='Z' => board.set(row * N + col, make_square(c as u8 - b'A', false)),
                    'a'..='z' => board.set(row * N + col, make_square(c as u8 - b'a', true)),
                    _ => return Err(format!("unexpected character {c:?} on board line {}", row + 1)),
                }
            }
        }
        Ok(board)
    }

    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(SQUARES + N);
        for row in 0..N {
            for col in 0..N {
                let square = self.get(row, col);
                out.push(match square {
                    EMPTY => '.',
                    s if square_is_blank(s) => (b'a' + square_letter(s)) as char,
                    s => (b'A' + square_letter(s)) as char,
                });
            }
            out.push('\n');
        }
        out
    }
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n{}", self.to_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premium_layout_matches_the_standard_board() {
        let count = |table: &[u8; SQUARES], value: u8| table.iter().filter(|&&m| m == value).count();
        assert_eq!(count(&WORD_MULT, 3), 8, "triple word squares");
        assert_eq!(count(&WORD_MULT, 2), 17, "double word squares, centre included");
        assert_eq!(count(&LETTER_MULT, 3), 12, "triple letter squares");
        assert_eq!(count(&LETTER_MULT, 2), 24, "double letter squares");
        assert_eq!(WORD_MULT[CENTER], 2);
        assert_eq!(WORD_MULT[0], 3);
        assert_eq!(LETTER_MULT[3], 2); // D1
        assert_eq!(LETTER_MULT[N + 5], 3); // F2
        for row in 0..N {
            for col in 0..N {
                let here = row * N + col;
                for other in [col * N + row, row * N + (N - 1 - col), (N - 1 - row) * N + col] {
                    assert_eq!(LETTER_MULT[here], LETTER_MULT[other]);
                    assert_eq!(WORD_MULT[here], WORD_MULT[other]);
                }
            }
        }
    }

    #[test]
    fn squares_encode_letter_blank_and_value() {
        let q = make_square(16, false);
        assert_eq!((square_letter(q), square_is_blank(q), square_value(q), square_tile(q)), (16, false, 10, 16));
        let blank_q = make_square(16, true);
        assert_eq!(
            (square_letter(blank_q), square_is_blank(blank_q), square_value(blank_q), square_tile(blank_q)),
            (16, true, 0, BLANK)
        );
    }

    #[test]
    fn names_and_text_round_trip() {
        assert_eq!(square_name(7, 7), "H8");
        assert_eq!(parse_square("H8"), Some((7, 7)));
        assert_eq!(parse_square("O15"), Some((14, 14)));
        assert_eq!(parse_square("a1"), Some((0, 0)));
        assert_eq!(parse_square("P1"), None);
        assert_eq!(parse_square("A16"), None);

        let mut board = Board::new();
        board.set(CENTER, make_square(2, false));
        board.set(CENTER + 1, make_square(0, true));
        let text = board.to_text();
        assert_eq!(text.lines().nth(7).unwrap(), ".......Ca......");
        assert_eq!(Board::from_text(&text).unwrap(), board);
        assert_eq!(board.transposed().get(8, 7), make_square(0, true));
        assert_eq!(board.tile_count(), 2);
    }
}
