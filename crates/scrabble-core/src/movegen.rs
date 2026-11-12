//! Move generation: Gordon's GADDAG algorithm with cross-check sets.
//!
//! The generator works one lane (row) at a time and only ever plays across. Down plays come from
//! running the same code on the transposed board. For each *anchor* (an empty square next to a
//! tile, or the centre of an empty board) it places a tile on the anchor, spells leftwards along
//! the reversed-prefix part of the GADDAG, then crosses the separator and spells rightwards.
//!
//! Each play is produced exactly once:
//! - While spelling leftwards, a tile may not be placed on another anchor. That play covers an
//!   anchor further left and is generated from there.
//! - A single placed tile could be described as an across or a down play. The down pass skips it
//!   when it also forms an across word, so it has one canonical form (see [`Play`]).
//!
//! Blanks are handled inside the traversal: at each square the blank tries exactly the letters
//! the GADDAG node and the cross-check allow.

use crate::board::{square_letter, square_value, Board, CENTER, EMPTY, LETTER_MULT, N, SQUARES, WORD_MULT};
use crate::gaddag::{Gaddag, SEPARATOR};
use crate::moves::{Play, PLACED_BLANK, PLAYED_THROUGH};
use crate::tiles::{Rack, ALL_LETTERS, BINGO_BONUS, BLANK, RACK_SIZE, TILE_TYPES, TILE_VALUES};

/// What the tiles above and below each empty square allow there, for plays along the rows.
#[derive(Clone)]
pub struct CrossChecks {
    /// Letters that may be placed: all 26 if the square has no tile above or below, the letters
    /// that complete a word if it has, and 0 for an occupied square.
    pub allowed: [u32; SQUARES],
    /// Total value of the tiles directly above and below (the rest of the cross-word).
    pub sum: [i16; SQUARES],
    /// True if a tile placed here forms a cross-word.
    pub has_word: [bool; SQUARES],
}

impl CrossChecks {
    /// Cross-checks from the *vertical* neighbours of each square of `cells`.
    pub fn vertical(gaddag: &Gaddag, cells: &[u8; SQUARES]) -> CrossChecks {
        let mut out = CrossChecks { allowed: [0; SQUARES], sum: [0; SQUARES], has_word: [false; SQUARES] };
        let (mut before, mut after) = ([0u8; N], [0u8; N]);
        for col in 0..N {
            for row in 0..N {
                let index = row * N + col;
                if cells[index] != EMPTY {
                    continue;
                }
                let mut top = row;
                while top > 0 && cells[(top - 1) * N + col] != EMPTY {
                    top -= 1;
                }
                let mut bottom = row;
                while bottom + 1 < N && cells[(bottom + 1) * N + col] != EMPTY {
                    bottom += 1;
                }
                if top == row && bottom == row {
                    out.allowed[index] = ALL_LETTERS;
                    continue;
                }
                let mut sum = 0;
                for (i, r) in (top..row).enumerate() {
                    before[i] = square_letter(cells[r * N + col]);
                    sum += square_value(cells[r * N + col]);
                }
                for (i, r) in (row + 1..=bottom).enumerate() {
                    after[i] = square_letter(cells[r * N + col]);
                    sum += square_value(cells[r * N + col]);
                }
                out.allowed[index] = gaddag.letters_between(&before[..row - top], &after[..bottom - row]);
                out.sum[index] = sum as i16;
                out.has_word[index] = true;
            }
        }
        out
    }
}

/// Anchor squares: empty squares with a tile next to them, or the centre if the board is empty.
pub fn anchors(cells: &[u8; SQUARES]) -> [bool; SQUARES] {
    let mut out = [false; SQUARES];
    let mut any_tile = false;
    for row in 0..N {
        for col in 0..N {
            let index = row * N + col;
            if cells[index] != EMPTY {
                any_tile = true;
                continue;
            }
            out[index] = (row > 0 && cells[index - N] != EMPTY)
                || (row + 1 < N && cells[index + N] != EMPTY)
                || (col > 0 && cells[index - 1] != EMPTY)
                || (col + 1 < N && cells[index + 1] != EMPTY);
        }
    }
    if !any_tile {
        out[CENTER] = true;
    }
    out
}

/// Append every legal tile placement for `rack` on `board` to `out`.
pub fn generate_plays(gaddag: &Gaddag, board: &Board, rack: &Rack, out: &mut Vec<Play>) {
    if rack.is_empty() {
        return;
    }
    generate_direction(gaddag, &board.cells, rack, false, out);
    generate_direction(gaddag, &board.transposed().cells, rack, true, out);
}

fn generate_direction(gaddag: &Gaddag, cells: &[u8; SQUARES], rack: &Rack, down: bool, out: &mut Vec<Play>) {
    let cross = CrossChecks::vertical(gaddag, cells);
    let anchor_squares = anchors(cells);
    for lane in 0..N {
        let row = lane * N..(lane + 1) * N;
        if !anchor_squares[row.clone()].iter().any(|&a| a) {
            continue;
        }
        let mut generator = LaneGenerator {
            gaddag,
            tile: cells[row.clone()].try_into().unwrap(),
            allowed: cross.allowed[row.clone()].try_into().unwrap(),
            cross_sum: cross.sum[row.clone()].try_into().unwrap(),
            has_cross: cross.has_word[row.clone()].try_into().unwrap(),
            is_anchor: anchor_squares[row.clone()].try_into().unwrap(),
            // The premium layout is symmetric under transposition, so the same table serves both
            // directions.
            letter_mult: LETTER_MULT[row.clone()].try_into().unwrap(),
            word_mult: WORD_MULT[row].try_into().unwrap(),
            placed: [0; N],
            rack: rack.counts,
            rack_letters: rack.letter_mask(),
            tiles_placed: 0,
            anchor: 0,
            down,
            lane: lane as u8,
            out,
        };
        for anchor in 0..N {
            if generator.is_anchor[anchor] {
                generator.anchor = anchor;
                generator.extend_left(anchor, gaddag.root());
            }
        }
    }
}

struct LaneGenerator<'a> {
    gaddag: &'a Gaddag,
    tile: [u8; N],
    allowed: [u32; N],
    cross_sum: [i16; N],
    has_cross: [bool; N],
    is_anchor: [bool; N],
    letter_mult: [u8; N],
    word_mult: [u8; N],
    /// Tiles placed by the play under construction: letter, plus `PLACED_BLANK` for a blank.
    placed: [u8; N],
    rack: [u8; TILE_TYPES],
    rack_letters: u32,
    tiles_placed: u8,
    anchor: usize,
    down: bool,
    lane: u8,
    out: &'a mut Vec<Play>,
}

impl LaneGenerator<'_> {
    /// Handle square `col` (at or left of the anchor) while spelling the word's start backwards.
    fn extend_left(&mut self, col: usize, node: u32) {
        let square = self.tile[col];
        if square != EMPTY {
            self.after_left(col, square_letter(square), node);
            return;
        }
        if col != self.anchor && self.is_anchor[col] {
            return;
        }
        let options = self.allowed[col] & (self.gaddag.children_mask(node) | self.gaddag.terminal_mask(node));
        let mut natural = options & self.rack_letters;
        while natural != 0 {
            let letter = natural.trailing_zeros() as u8;
            natural &= natural - 1;
            self.take(letter);
            self.placed[col] = letter;
            self.after_left(col, letter, node);
            self.put_back(letter);
        }
        if self.rack[BLANK as usize] > 0 {
            self.rack[BLANK as usize] -= 1;
            self.tiles_placed += 1;
            let mut any = options;
            while any != 0 {
                let letter = any.trailing_zeros() as u8;
                any &= any - 1;
                self.placed[col] = letter | PLACED_BLANK;
                self.after_left(col, letter, node);
            }
            self.rack[BLANK as usize] += 1;
            self.tiles_placed -= 1;
        }
    }

    /// `letter` now sits on `col`, the current left end of the word.
    fn after_left(&mut self, col: usize, letter: u8, node: u32) {
        let left_free = col == 0 || self.tile[col - 1] == EMPTY;
        let right_of_anchor_free = self.anchor == N - 1 || self.tile[self.anchor + 1] == EMPTY;
        if left_free && right_of_anchor_free && self.gaddag.terminal_mask(node) & (1 << letter) != 0 {
            self.record(col, self.anchor);
        }
        let next = self.gaddag.child(node, letter);
        if next == 0 {
            return;
        }
        if col > 0 {
            self.extend_left(col - 1, next);
        }
        if left_free && self.anchor < N - 1 {
            let turned = self.gaddag.child(next, SEPARATOR);
            if turned != 0 {
                self.extend_right(self.anchor + 1, turned, col);
            }
        }
    }

    /// Handle square `col` (right of the anchor) for a word that starts at `start`.
    fn extend_right(&mut self, col: usize, node: u32, start: usize) {
        let square = self.tile[col];
        if square != EMPTY {
            self.after_right(col, square_letter(square), node, start);
            return;
        }
        let options = self.allowed[col] & (self.gaddag.children_mask(node) | self.gaddag.terminal_mask(node));
        let mut natural = options & self.rack_letters;
        while natural != 0 {
            let letter = natural.trailing_zeros() as u8;
            natural &= natural - 1;
            self.take(letter);
            self.placed[col] = letter;
            self.after_right(col, letter, node, start);
            self.put_back(letter);
        }
        if self.rack[BLANK as usize] > 0 {
            self.rack[BLANK as usize] -= 1;
            self.tiles_placed += 1;
            let mut any = options;
            while any != 0 {
                let letter = any.trailing_zeros() as u8;
                any &= any - 1;
                self.placed[col] = letter | PLACED_BLANK;
                self.after_right(col, letter, node, start);
            }
            self.rack[BLANK as usize] += 1;
            self.tiles_placed -= 1;
        }
    }

    fn after_right(&mut self, col: usize, letter: u8, node: u32, start: usize) {
        let right_free = col == N - 1 || self.tile[col + 1] == EMPTY;
        if right_free && self.gaddag.terminal_mask(node) & (1 << letter) != 0 {
            self.record(start, col);
        }
        if col < N - 1 {
            let next = self.gaddag.child(node, letter);
            if next != 0 {
                self.extend_right(col + 1, next, start);
            }
        }
    }

    #[inline]
    fn take(&mut self, letter: u8) {
        self.rack[letter as usize] -= 1;
        if self.rack[letter as usize] == 0 {
            self.rack_letters &= !(1 << letter);
        }
        self.tiles_placed += 1;
    }

    #[inline]
    fn put_back(&mut self, letter: u8) {
        self.rack[letter as usize] += 1;
        self.rack_letters |= 1 << letter;
        self.tiles_placed -= 1;
    }

    /// The word on squares `start..=end` of this lane is a legal play: score it and store it.
    fn record(&mut self, start: usize, end: usize) {
        // A lone tile that also makes an across word is recorded by the across pass.
        if self.down && self.tiles_placed == 1 && self.has_cross[self.anchor] {
            return;
        }
        let mut word = [0u8; N];
        let (mut main, mut multiplier, mut cross_words) = (0i32, 1i32, 0i32);
        for col in start..=end {
            let square = self.tile[col];
            if square != EMPTY {
                main += square_value(square);
                word[col - start] = square_letter(square) | PLAYED_THROUGH;
                continue;
            }
            let code = self.placed[col];
            let face = if code & PLACED_BLANK != 0 { 0 } else { TILE_VALUES[(code & 0x1F) as usize] as i32 };
            let value = face * self.letter_mult[col] as i32;
            let word_mult = self.word_mult[col] as i32;
            main += value;
            multiplier *= word_mult;
            if self.has_cross[col] {
                cross_words += (self.cross_sum[col] as i32 + value) * word_mult;
            }
            word[col - start] = code;
        }
        let bingo = if self.tiles_placed as usize == RACK_SIZE { BINGO_BONUS } else { 0 };
        self.out.push(Play {
            down: self.down,
            lane: self.lane,
            start: start as u8,
            len: (end - start + 1) as u8,
            tiles_played: self.tiles_placed,
            score: (main * multiplier + cross_words + bingo) as i16,
            word,
        });
    }
}

/// Append every distinct non-empty set of tiles that could be exchanged from `rack`.
pub fn generate_exchanges(rack: &Rack, out: &mut Vec<Rack>) {
    fn choose(rack: &Rack, tile: usize, chosen: &mut Rack, out: &mut Vec<Rack>) {
        if tile == TILE_TYPES {
            if !chosen.is_empty() {
                out.push(*chosen);
            }
            return;
        }
        choose(rack, tile + 1, chosen, out);
        for _ in 0..rack.counts[tile] {
            chosen.add(tile as u8);
            choose(rack, tile + 1, chosen, out);
        }
        for _ in 0..rack.counts[tile] {
            chosen.remove(tile as u8);
        }
    }
    choose(rack, 0, &mut Rack::new(), out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::make_square;

    fn gaddag() -> Gaddag {
        Gaddag::build(&["AT", "TA", "CAT", "CATS", "ACT", "AS", "SAT", "TAS", "QI", "ZA", "CASTE", "AA"])
    }

    fn plays(board: &Board, rack: &str) -> Vec<String> {
        let mut out = Vec::new();
        generate_plays(&gaddag(), board, &Rack::parse(rack).unwrap(), &mut out);
        let mut text: Vec<String> = out.iter().map(Play::to_string).collect();
        text.sort();
        text
    }

    #[test]
    fn opening_plays_cover_the_centre_in_both_directions() {
        let found = plays(&Board::new(), "AT");
        // AT and TA, each through H8 at two offsets, across and down. 1+1 doubled = 4.
        assert_eq!(
            found,
            ["8G AT 4", "8G TA 4", "8H AT 4", "8H TA 4", "H7 AT 4", "H7 TA 4", "H8 AT 4", "H8 TA 4"]
        );
    }

    #[test]
    fn hooks_extensions_and_parallel_plays() {
        let mut board = Board::new();
        for (i, letter) in [2u8, 0, 19].into_iter().enumerate() {
            board.set(CENTER + i, make_square(letter, false)); // CAT at H8-J8
        }
        let found = plays(&board, "S");
        // CATS by hooking K8 (no premium): 3+1+1+1 = 6.
        // AS down from the A: the S lands on I9, a double letter square: 1 + 1x2 = 3.
        // Nothing else: SC, CS, SA, ST and TS are not words.
        assert_eq!(found, ["8H (CAT)S 6", "I8 (A)S 3"]);
    }

    #[test]
    fn single_tile_forming_two_words_is_one_across_play() {
        // C A T on row 8 and an A above the square after T: placing S there makes CATS and AS.
        let mut board = Board::new();
        for (i, letter) in [2u8, 0, 19].into_iter().enumerate() {
            board.set(CENTER + i, make_square(letter, false));
        }
        board.set(CENTER + 3 - N, make_square(0, false)); // A at K7
        let found = plays(&board, "S");
        // CATS = 3+1+1+1 = 6, AS = 1+1 = 2; K8 has no premium.
        assert!(found.contains(&"8H (CAT)S 8".to_string()), "{found:?}");
        assert_eq!(found.iter().filter(|p| p.ends_with(" 8")).count(), 1, "{found:?}");
    }

    #[test]
    fn blank_can_be_any_letter_and_scores_zero() {
        let found = plays(&Board::new(), "?");
        assert!(found.is_empty(), "one tile cannot open: {found:?}");
        let found = plays(&Board::new(), "Q?");
        // QI with the blank as I: Q on H8 (double word) = 20, or Q on G8 = 20.
        assert!(found.contains(&"8H Qi 20".to_string()), "{found:?}");
        assert!(found.contains(&"8G Qi 20".to_string()), "{found:?}");
        assert!(found.iter().all(|p| p.contains("Qi")), "{found:?}");
    }

    #[test]
    fn natural_tile_and_blank_for_the_same_letter_are_both_generated() {
        let found = plays(&Board::new(), "AT?");
        assert!(found.contains(&"8G AT 4".to_string()));
        assert!(found.contains(&"8G aT 2".to_string()));
        assert!(found.contains(&"8G At 2".to_string()));
        // CAT with the blank as C.
        assert!(found.contains(&"8F cAT 4".to_string()), "{found:?}");
        let unique: std::collections::HashSet<&String> = found.iter().collect();
        assert_eq!(unique.len(), found.len(), "duplicates in {found:?}");
    }

    #[test]
    fn exchanges_are_the_distinct_sub_multisets() {
        let mut out = Vec::new();
        generate_exchanges(&Rack::parse("AAB").unwrap(), &mut out);
        let mut text: Vec<String> = out.iter().map(Rack::to_string).collect();
        text.sort();
        assert_eq!(text, ["A", "AA", "AAB", "AB", "B"]);

        out.clear();
        generate_exchanges(&Rack::parse("ABCDEF?").unwrap(), &mut out);
        assert_eq!(out.len(), 127);
    }
}
