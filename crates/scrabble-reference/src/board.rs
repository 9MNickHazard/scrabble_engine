//! The board and its text form (RULES.md "Canonical forms").

use crate::tiles::{PlacedTile, N};

/// A board position. `cells[row][col]` is `None` for an empty square, or `(letter, is_blank)` with
/// `letter` in `b'A'..=b'Z'` (for a blank, the letter it stands for).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RefBoard {
    pub cells: [[Option<(u8, bool)>; N]; N],
}

impl RefBoard {
    /// A board with no tiles.
    pub fn empty() -> Self {
        RefBoard {
            cells: [[None; N]; N],
        }
    }

    /// Read the text form: 15 lines of 15 characters, `.` for an empty square, an upper-case letter
    /// for a normal tile, a lower-case letter for a blank standing for that letter.
    ///
    /// Leading and trailing whitespace on each line (including `\r`) is ignored, and so are lines
    /// that are empty after trimming. Anything else is an error.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let lines: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        if lines.len() != N {
            return Err(format!(
                "board text has {} non-empty lines, expected {N}",
                lines.len()
            ));
        }
        let mut board = RefBoard::empty();
        for (row, line) in lines.iter().enumerate() {
            let bytes = line.as_bytes();
            if bytes.len() != N {
                return Err(format!(
                    "board line {} {:?} has {} characters, expected {N}",
                    row + 1,
                    line,
                    bytes.len()
                ));
            }
            for (col, &b) in bytes.iter().enumerate() {
                board.cells[row][col] = match b {
                    b'.' => None,
                    b'A'..=b'Z' => Some((b, false)),
                    b'a'..=b'z' => Some((b.to_ascii_uppercase(), true)),
                    _ => {
                        return Err(format!(
                            "board line {} {:?}: character {:?} is not '.', A-Z or a-z",
                            row + 1,
                            line,
                            b as char
                        ))
                    }
                };
            }
        }
        Ok(board)
    }

    /// The text form: 15 lines of 15 characters, each line ending in `\n`.
    pub fn to_text(&self) -> String {
        let mut text = String::with_capacity(N * (N + 1));
        for row in &self.cells {
            for cell in row {
                text.push(match *cell {
                    None => '.',
                    Some((letter, false)) => letter as char,
                    Some((letter, true)) => letter.to_ascii_lowercase() as char,
                });
            }
            text.push('\n');
        }
        text
    }

    /// Put tiles on the board without any checking (an occupied square is overwritten).
    pub fn place(&mut self, tiles: &[PlacedTile]) {
        for tile in tiles {
            self.cells[tile.row as usize][tile.col as usize] = Some((tile.letter, tile.blank));
        }
    }

    /// True when no square holds a tile.
    pub fn is_empty(&self) -> bool {
        self.cells
            .iter()
            .all(|row| row.iter().all(|cell| cell.is_none()))
    }

    /// The same position reflected in the main diagonal: `(row, col)` becomes `(col, row)`.
    /// The premium layout is symmetric under this reflection, so every play transposes to a play
    /// with the same score. Used by the tests.
    pub fn transposed(&self) -> Self {
        let mut out = RefBoard::empty();
        for row in 0..N {
            for col in 0..N {
                out.cells[col][row] = self.cells[row][col];
            }
        }
        out
    }
}
