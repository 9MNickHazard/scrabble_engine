//! Compact encoding of afterstates for the value network.
//!
//! An afterstate is the position a move leaves behind, seen by the player who made it, before
//! any tiles are drawn. The engine writes small integer codes; the network expands them into
//! input planes on the GPU. One afterstate is:
//!
//! - `board`: 225 bytes. 0 for an empty square, otherwise `letter + 1` (1..=26), plus
//!   [`CODE_BLANK`] if the tile is a blank and [`CODE_PLACED`] if this move placed it.
//! - `leave`: 27 counts, the tiles the mover keeps.
//! - `unseen`: 27 counts, the tiles the mover cannot see (bag plus opponent's rack, plus anything
//!   the mover just threw back in an exchange).
//! - `meta`: [`META`] small integers, see the `M_*` constants.
//! - `leave_value`: the static value of the leave, the baseline the network adds to.
//!
//! Cross-check masks are not stored with an afterstate: [`cross_masks`] recomputes them from the
//! board codes, for self-play candidates and for replayed positions alike.

use crate::board::{make_square, square_is_blank, square_letter, Board, EMPTY, N, SQUARES};
use crate::gaddag::Gaddag;
use crate::game::Game;
use crate::movegen::CrossChecks;
use crate::moves::Move;
use crate::policy::{bag_after_draw, Candidate};
use crate::tiles::TILE_TYPES;
use rayon::prelude::*;

pub const CODE_LETTER: u8 = 0x1F;
pub const CODE_BLANK: u8 = 0x20;
pub const CODE_PLACED: u8 = 0x40;

/// Number of integers in an afterstate's `meta`.
pub const META: usize = 11;
/// Points the move scored, including the bonus for going out.
pub const M_SCORE: usize = 0;
/// The mover's spread after the move.
pub const M_SPREAD: usize = 1;
/// Tiles left in the bag once the mover has drawn.
pub const M_BAG: usize = 2;
/// Tiles the mover draws.
pub const M_DRAWS: usize = 3;
/// Passes and exchanges in a row after the move.
pub const M_SCORELESS: usize = 4;
/// Tiles on the opponent's rack.
pub const M_OPPONENT_TILES: usize = 5;
/// 0 placement, 1 exchange, 2 pass.
pub const M_KIND: usize = 6;
/// 0 the game goes on, 1 the mover went out, 2 sixth scoreless turn.
pub const M_TERMINAL: usize = 7;
/// The opponent's previous turn: 0 none yet, 1 placement, 2 exchange, 3 pass.
pub const M_OPPONENT_LAST_KIND: usize = 8;
/// Tiles the opponent placed or exchanged on that turn.
pub const M_OPPONENT_LAST_TILES: usize = 9;
/// Points the opponent scored on that turn.
pub const M_OPPONENT_LAST_SCORE: usize = 10;

/// In a cross mask: bits 0..=25 are the letters a perpendicular word allows on the square, and
/// this bit says there *is* a perpendicular word (the square touches a tile in that direction).
pub const CROSS_HAS_WORD: u32 = 1 << 26;
/// Masks per board: one set for plays along rows, one for plays along columns.
pub const CROSS_WORDS: usize = 2 * SQUARES;

/// Board codes for `board` with nothing marked as just placed.
pub fn board_codes(board: &Board) -> [u8; SQUARES] {
    let mut codes = [0u8; SQUARES];
    for (code, &square) in codes.iter_mut().zip(board.cells.iter()) {
        if square != EMPTY {
            *code = (square_letter(square) + 1) | if square_is_blank(square) { CODE_BLANK } else { 0 };
        }
    }
    codes
}

/// Write the afterstate of `candidate` for the player to move in `game`.
///
/// `base` is [`board_codes`] of the current board. `unseen` is `game.unseen(game.to_move)`.
pub fn encode_afterstate(
    game: &Game,
    base: &[u8; SQUARES],
    unseen: &[u8; TILE_TYPES],
    candidate: &Candidate,
    board: &mut [u8],
    leave: &mut [u8],
    unseen_out: &mut [u8],
    meta: &mut [i16],
) {
    let (me, opponent) = (game.to_move, 1 - game.to_move);
    board.copy_from_slice(base);
    leave.copy_from_slice(&candidate.leave.counts);
    unseen_out.copy_from_slice(unseen);
    let (kind, draws, scoreless) = match &candidate.mv {
        Move::Play(play) => {
            for tile in play.placed() {
                board[tile.index] = (tile.letter + 1) | if tile.blank { CODE_BLANK } else { 0 } | CODE_PLACED;
            }
            (0, (play.tiles_played as usize).min(game.bag.len()), 0)
        }
        Move::Exchange(tiles) => {
            // The mover knows what went back into the bag.
            for (count, returned) in unseen_out.iter_mut().zip(tiles.counts) {
                *count += returned;
            }
            (1, tiles.len(), game.scoreless_turns + 1)
        }
        Move::Pass => (2, 0, game.scoreless_turns + 1),
    };
    meta[M_SCORE] = candidate.score as i16;
    meta[M_SPREAD] = (game.spread(me) + candidate.score) as i16;
    meta[M_BAG] = bag_after_draw(game, &candidate.mv) as i16;
    meta[M_DRAWS] = draws as i16;
    meta[M_SCORELESS] = scoreless as i16;
    meta[M_OPPONENT_TILES] = game.racks[opponent].len() as i16;
    meta[M_KIND] = kind;
    meta[M_TERMINAL] = candidate.terminal as i16;
    let theirs = game.last_turn[opponent];
    meta[M_OPPONENT_LAST_KIND] = theirs.kind as i16;
    meta[M_OPPONENT_LAST_TILES] = theirs.tiles as i16;
    meta[M_OPPONENT_LAST_SCORE] = theirs.score;
}

/// Cross-check masks for one coded board: `out[..225]` for plays along rows (what the tiles
/// above and below each empty square allow), `out[225..]` for plays along columns. Occupied
/// squares get 0; an empty square with no neighbour in that direction gets all 26 letters.
pub fn cross_masks(gaddag: &Gaddag, codes: &[u8], out: &mut [u32]) {
    debug_assert!(codes.len() == SQUARES && out.len() == CROSS_WORDS);
    let mut cells = [EMPTY; SQUARES];
    let mut transposed = [EMPTY; SQUARES];
    for index in 0..SQUARES {
        let code = codes[index];
        if code & CODE_LETTER != 0 {
            let square = make_square((code & CODE_LETTER) - 1, code & CODE_BLANK != 0);
            cells[index] = square;
            transposed[index % N * N + index / N] = square;
        }
    }
    let rows = CrossChecks::vertical(gaddag, &cells);
    let columns = CrossChecks::vertical(gaddag, &transposed);
    for index in 0..SQUARES {
        let flipped = index % N * N + index / N;
        out[index] = rows.allowed[index] | if rows.has_word[index] { CROSS_HAS_WORD } else { 0 };
        out[SQUARES + index] = columns.allowed[flipped] | if columns.has_word[flipped] { CROSS_HAS_WORD } else { 0 };
    }
}

/// [`cross_masks`] for many boards at once, in parallel. `codes` holds `n * 225` bytes and `out`
/// `n * 450` words.
pub fn cross_masks_batch(gaddag: &Gaddag, codes: &[u8], out: &mut [u32]) {
    assert_eq!(codes.len() % SQUARES, 0);
    assert_eq!(out.len(), codes.len() / SQUARES * CROSS_WORDS);
    codes
        .par_chunks(SQUARES)
        .zip(out.par_chunks_mut(CROSS_WORDS))
        .with_min_len(64)
        .for_each(|(board, masks)| cross_masks(gaddag, board, masks));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::CENTER;
    use crate::leave::LeaveModel;
    use crate::policy::candidate;
    use crate::tiles::{Rack, ALL_LETTERS, BLANK};

    fn gaddag() -> Gaddag {
        Gaddag::build(&["AT", "TA", "CAT", "CATS", "AS", "QI", "TAS"])
    }

    fn game_with_cat() -> Game {
        let mut board = Board::new();
        for (i, letter) in [2u8, 0, 19].into_iter().enumerate() {
            board.set(CENTER + i, make_square(letter, i == 1)); // C a T: the A is a blank
        }
        let racks = [Rack::parse("S?EE").unwrap(), Rack::parse("QI").unwrap()];
        Game::from_position(board, racks, &[4u8; 10], [20, 5], 0, 1)
    }

    fn encode(game: &Game, mv: Move) -> ([u8; SQUARES], [u8; TILE_TYPES], [u8; TILE_TYPES], [i16; META]) {
        let described = candidate(game, &LeaveModel::zero(), mv);
        let (mut board, mut leave, mut unseen, mut meta) = ([0; SQUARES], [0; TILE_TYPES], [0; TILE_TYPES], [0; META]);
        encode_afterstate(game, &board_codes(&game.board), &game.unseen(0), &described, &mut board, &mut leave, &mut unseen, &mut meta);
        (board, leave, unseen, meta)
    }

    #[test]
    fn play_afterstate_marks_new_tiles_and_keeps_the_unseen_pool() {
        let game = game_with_cat();
        let mut plays = Vec::new();
        game.plays(&gaddag(), &mut plays);
        // C(3) + blank A(0) + T(1) + blank S(0) = 4.
        let cats = plays.iter().find(|p| p.to_string() == "8H (CAT)s 4").copied().expect("CATs with the blank");
        let (board, leave, unseen, meta) = encode(&game, Move::Play(cats));
        assert_eq!(board[CENTER], 3); // C
        assert_eq!(board[CENTER + 1], 1 | CODE_BLANK); // blank A, already there
        assert_eq!(board[CENTER + 3], 19 | CODE_BLANK | CODE_PLACED); // blank S, just placed
        assert_eq!(board.iter().filter(|&&c| c != 0).count(), 4);
        assert_eq!(leave[18], 1, "the real S is kept");
        assert_eq!(leave[BLANK as usize], 0);
        assert_eq!(unseen, game.unseen(0), "a placement does not change what the mover cannot see");
        assert_eq!(meta, [4, 20 - 5 + 4, 9, 1, 0, 2, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn exchange_and_pass_afterstates() {
        let game = game_with_cat();
        let thrown = Rack::parse("EE").unwrap();
        let (board, leave, unseen, meta) = encode(&game, Move::Exchange(thrown));
        assert_eq!(board, board_codes(&game.board));
        assert_eq!(leave.iter().map(|&c| c as usize).sum::<usize>(), 2);
        assert_eq!(unseen[4], game.unseen(0)[4] + 2, "the two E's thrown back are unseen again");
        assert_eq!(meta, [0, 15, 10, 2, 1, 2, 1, 0, 0, 0, 0]);

        let mut late = game_with_cat();
        late.scoreless_turns = 5;
        late.last_turn[1] = crate::game::LastTurn { kind: 2, tiles: 5, score: 0 };
        let (_, _, _, meta) = encode(&late, Move::Pass);
        // The last three entries: the opponent's previous turn was an exchange of five tiles.
        assert_eq!(meta, [0, 15, 10, 0, 6, 2, 2, 2, 2, 5, 0]);
    }

    #[test]
    fn cross_masks_match_the_generator_in_both_directions() {
        let (gaddag, game) = (gaddag(), game_with_cat());
        let mut masks = [0u32; CROSS_WORDS];
        cross_masks(&gaddag, &board_codes(&game.board), &mut masks);
        let letter = |c: char| 1u32 << (c as u8 - b'A');
        // Row plays: the square under the A (I9) must complete A? downwards: AS or AT.
        assert_eq!(masks[CENTER + 1 + N], letter('S') | letter('T') | CROSS_HAS_WORD);
        // Above the T (J7): ?T -> AT.
        assert_eq!(masks[CENTER + 2 - N], letter('A') | CROSS_HAS_WORD);
        // Column plays: the square right of CAT (K8) must complete CAT? -> CATS.
        assert_eq!(masks[SQUARES + CENTER + 3], letter('S') | CROSS_HAS_WORD);
        // Left of CAT (G8): ?CAT is not a word.
        assert_eq!(masks[SQUARES + CENTER - 1], CROSS_HAS_WORD);
        // Occupied squares and far-away squares.
        assert_eq!((masks[CENTER], masks[SQUARES + CENTER]), (0, 0));
        assert_eq!((masks[0], masks[SQUARES]), (ALL_LETTERS, ALL_LETTERS));
        // K8 has no vertical neighbour: unconstrained for row plays.
        assert_eq!(masks[CENTER + 3], ALL_LETTERS);

        let mut batch = vec![0u32; 3 * CROSS_WORDS];
        let codes: Vec<u8> = board_codes(&game.board).iter().copied().cycle().take(3 * SQUARES).collect();
        cross_masks_batch(&gaddag, &codes, &mut batch);
        assert!(batch.chunks(CROSS_WORDS).all(|chunk| chunk == masks));
    }
}
