//! Independent reference implementation of the placement rules in `docs/RULES.md`, used only to
//! test `scrabble-core`. It is written from the rules document alone and shares no code or method
//! with the engine. Correctness is favoured over speed throughout.
//!
//! - [`check_placement`] judges one placement and scores it, rule by rule.
//! - [`legal_plays`] lists every legal placement for a rack by sliding every lexicon word along
//!   every row and column; `check_placement` has the final say on each candidate.
//! - [`legal_plays_exhaustive`] does the same by brute force over squares and tiles (small racks
//!   only) and exists to cross-check `legal_plays`.
//! - [`parse_cases`] reads the hand-checked scoring cases in `testdata/scoring_cases.txt`
//!   (also built in as [`SCORING_CASES`]).

// Plain `for r in 0..N` loops over board coordinates are deliberate: they read like the rules.
#![allow(clippy::needless_range_loop)]

mod board;
mod cases;
mod exhaustive;
mod generate;
mod judge;
mod lexicon;
mod tiles;

pub use board::RefBoard;
pub use cases::{parse_cases, Case, SCORING_CASES};
pub use exhaustive::legal_plays_exhaustive;
pub use generate::legal_plays;
pub use judge::check_placement;
pub use lexicon::RefLexicon;
pub use tiles::{
    parse_placed_tile, parse_square, placed_tile_text, premium_at, square_name, tile_value,
    PlacedTile, Premium, RefPlay, N,
};
