//! Scrabble engine: NASPA two-player rules (docs/RULES.md), GADDAG move generation and scoring.

pub mod belief;
pub mod bench;
pub mod board;
pub mod encode;
pub mod env;
pub mod gaddag;
pub mod game;
pub mod leave;
pub mod lexicon;
pub mod movegen;
pub mod moves;
pub mod policy;
pub mod rng;
pub mod selfplay;
pub mod tiles;
pub mod validate;
