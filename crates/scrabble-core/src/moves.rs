//! Moves: tile placements, exchanges and passes.

use crate::board::{make_square, square_name, N};
use crate::tiles::{tile_to_char, Rack, BLANK};

/// Flag in [`Play::word`]: this square's tile was already on the board.
pub const PLAYED_THROUGH: u8 = 0x40;
/// Flag in [`Play::word`]: the tile placed here is a blank.
pub const PLACED_BLANK: u8 = 0x80;
const LETTER_BITS: u8 = 0x1F;

/// A tile placement, described by its main word.
///
/// `lane` is the row of an across play or the column of a down play; `start` is where the main
/// word begins along that lane. `word[i]` is the letter (0..=25) on the i-th square of the main
/// word, with [`PLAYED_THROUGH`] set if the tile was already there and [`PLACED_BLANK`] set if a
/// blank is being placed.
///
/// A single placed tile has one canonical form: across if it forms an across word of two or
/// more letters, otherwise down. So two `Play`s are equal exactly when they put the same tiles on
/// the same squares (docs/RULES.md, "the same play").
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Play {
    pub down: bool,
    pub lane: u8,
    pub start: u8,
    pub len: u8,
    pub tiles_played: u8,
    pub score: i16,
    pub word: [u8; N],
}

/// One tile of a placement: board index, letter 0..=25, and whether it is a blank.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Placed {
    pub index: usize,
    pub letter: u8,
    pub blank: bool,
}

impl Placed {
    /// The byte this tile becomes on the board.
    #[inline]
    pub fn square(&self) -> u8 {
        make_square(self.letter, self.blank)
    }

    /// The tile type it takes from the rack.
    #[inline]
    pub fn tile(&self) -> u8 {
        if self.blank {
            BLANK
        } else {
            self.letter
        }
    }
}

impl Play {
    /// Board index of the i-th square of the main word.
    #[inline]
    pub fn index(&self, i: usize) -> usize {
        let along = self.start as usize + i;
        if self.down {
            along * N + self.lane as usize
        } else {
            self.lane as usize * N + along
        }
    }

    /// The newly placed tiles, in board order along the word.
    pub fn placed(&self) -> impl Iterator<Item = Placed> + '_ {
        (0..self.len as usize).filter_map(move |i| {
            let code = self.word[i];
            (code & PLAYED_THROUGH == 0).then(|| Placed {
                index: self.index(i),
                letter: code & LETTER_BITS,
                blank: code & PLACED_BLANK != 0,
            })
        })
    }

    /// The tiles this play takes from the rack.
    pub fn tiles_used(&self) -> Rack {
        let mut used = Rack::new();
        for tile in self.placed() {
            used.add(tile.tile());
        }
        used
    }

    /// The main word as text: lower case for a placed blank, existing tiles in parentheses.
    pub fn word_text(&self) -> String {
        let mut out = String::new();
        let mut through = false;
        for &code in &self.word[..self.len as usize] {
            let is_through = code & PLAYED_THROUGH != 0;
            if is_through != through {
                out.push(if is_through { '(' } else { ')' });
                through = is_through;
            }
            let letter = (b'A' + (code & LETTER_BITS)) as char;
            out.push(if code & PLACED_BLANK != 0 { letter.to_ascii_lowercase() } else { letter });
        }
        if through {
            out.push(')');
        }
        out
    }

    /// Position in the usual notation: row then column (`8D`) for across, column then row
    /// (`D8`) for down.
    pub fn position_text(&self) -> String {
        if self.down {
            square_name(self.start as usize, self.lane as usize)
        } else {
            format!("{}{}", self.lane + 1, (b'A' + self.start) as char)
        }
    }
}

impl std::fmt::Display for Play {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.position_text(), self.word_text(), self.score)
    }
}

impl std::fmt::Debug for Play {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Play({self})")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    Pass,
    /// The tiles returned to the bag.
    Exchange(Rack),
    Play(Play),
}

impl Move {
    /// Points this move scores before any end-of-game adjustment.
    #[inline]
    pub fn score(&self) -> i32 {
        match self {
            Move::Play(play) => play.score as i32,
            _ => 0,
        }
    }

    /// The tiles the move takes out of the rack.
    pub fn tiles_used(&self) -> Rack {
        match self {
            Move::Pass => Rack::new(),
            Move::Exchange(tiles) => *tiles,
            Move::Play(play) => play.tiles_used(),
        }
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Move::Pass => write!(f, "pass"),
            Move::Exchange(tiles) => {
                write!(f, "exchange ")?;
                for tile in tiles.tiles() {
                    write!(f, "{}", tile_to_char(tile))?;
                }
                Ok(())
            }
            Move::Play(play) => write!(f, "{play}"),
        }
    }
}
