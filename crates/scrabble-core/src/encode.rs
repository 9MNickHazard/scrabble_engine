//! Compact encoding of afterstates for the value network.
//!
//! An afterstate is the position a move leaves behind, seen by the player who made it, before
//! any tiles are drawn. The engine writes small integer codes; the network expands them into
//! input planes on the GPU. One afterstate is:
//!
//! - `board`: 225 bytes. 0 for an empty square, otherwise `letter + 1` (1..=26), plus
//!   [`CODE_BLANK`] if the tile is a blank, [`CODE_PLACED`] if this move placed it and
//!   [`CODE_OPPONENT`] if the opponent's previous move did.
//! - `leave`: 27 counts, the tiles the mover keeps.
//! - `unseen`: 27 counts, the tiles the mover cannot see (bag plus opponent's rack, plus anything
//!   the mover just threw back in an exchange).
//! - `meta`: [`META`] small integers, see the `M_*` constants.
//! - `leave_value`: the static value of the leave, the baseline the network adds to.
//!
//! Cross-check masks are not stored with an afterstate: [`cross_masks`] recomputes them from the
//! board codes, for self-play candidates and for replayed positions alike. So does
//! [`open_squares`] for the openness bytes, which are worked out from the codes and the masks.

use crate::board::{make_square, square_is_blank, square_letter, Board, EMPTY, N, SQUARES};
use crate::gaddag::Gaddag;
use crate::game::Game;
use crate::movegen::CrossChecks;
use crate::moves::Move;
use crate::policy::{bag_after_draw, Candidate};
use crate::tiles::{ALL_LETTERS, RACK_SIZE, TILE_TYPES};
use rayon::prelude::*;

pub const CODE_LETTER: u8 = 0x1F;
pub const CODE_BLANK: u8 = 0x20;
pub const CODE_PLACED: u8 = 0x40;
/// The tile was placed by the opponent's previous move.
pub const CODE_OPPONENT: u8 = 0x80;

/// Number of integers in an afterstate's `meta`.
pub const META: usize = 13 + TILE_TYPES;
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
/// Turns the opponent has taken since they last placed all seven tiles (0: on their previous
/// turn); -1 if they have not yet.
pub const M_OPPONENT_SINCE_BINGO: usize = 11;
/// 1 if the next [`TILE_TYPES`] entries hold an inference about the opponent's rack, 0 if nothing
/// is known beyond the unseen pool (they are then all zero).
pub const M_BELIEF_INFERRED: usize = 12;
/// First of [`TILE_TYPES`] entries: the expected number of each tile type on the opponent's
/// rack, times `belief::SCALE`.
pub const M_BELIEF: usize = 13;

/// In an openness byte ([`open_squares`]) the low four bits describe placements along the row
/// and the high four bits placements down the column. Within each half, these bits are the
/// fewest tiles a placement needs in order to cover the square (0: no placement can)...
pub const OPEN_TILES: u8 = 0x07;
/// ...and this bit says that a placement of all seven tiles can cover it.
pub const OPEN_SEVEN: u8 = 0x08;

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

/// Board codes for the player to move in `game`: [`board_codes`] with the tiles of the
/// opponent's previous move marked.
pub fn base_codes(game: &Game) -> [u8; SQUARES] {
    let mut codes = board_codes(&game.board);
    if let Some(play) = &game.last_play[1 - game.to_move] {
        for tile in play.placed() {
            codes[tile.index] |= CODE_OPPONENT;
        }
    }
    codes
}

/// Write the afterstate of `candidate` for the player to move in `game`.
///
/// `base` is [`base_codes`] of the game. `unseen` is `game.unseen(game.to_move)`.
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
    meta[M_OPPONENT_SINCE_BINGO] = game.since_bingo[opponent];
    let belief = &game.belief[me];
    meta[M_BELIEF_INFERRED] = belief.inferred as i16;
    meta[M_BELIEF..M_BELIEF + TILE_TYPES].copy_from_slice(&belief.counts);
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

/// How open each empty square of a coded board is: which squares the next player could cover,
/// with how few tiles, and where seven tiles fit. `masks` are the board's [`cross_masks`]; `out`
/// gets one byte per square (see [`OPEN_TILES`] and [`OPEN_SEVEN`]).
///
/// A placement here is any run of squares along a row or column that the rules of placing allow
/// and no cross-check rules out: it fills between one and seven empty squares, every one of them
/// a square some letter fits on; it joins what is on the board (it runs through a tile or forms a
/// cross-word; on an empty board it covers the centre); and the word it makes is at least two
/// squares long unless the single tile forms a cross-word. Whether any real word fits is not
/// asked: this is the shape of the board, which is the same for every rack.
pub fn open_squares(codes: &[u8], masks: &[u32], out: &mut [u8]) {
    debug_assert!(codes.len() == SQUARES && masks.len() == CROSS_WORDS && out.len() == SQUARES);
    out.fill(0);
    let empty_board = codes.iter().all(|&code| code & CODE_LETTER == 0);
    let center = N / 2;
    for direction in 0..2 {
        let shift = 4 * direction;
        for lane in 0..N {
            let index = |j: usize| if direction == 0 { lane * N + j } else { j * N + lane };
            let (mut occupied, mut fits, mut crosses) = ([false; N], [false; N], [false; N]);
            let mut touched = false;
            for j in 0..N {
                let (code, mask) = (codes[index(j)], masks[direction * SQUARES + index(j)]);
                occupied[j] = code & CODE_LETTER != 0;
                fits[j] = !occupied[j] && mask & ALL_LETTERS != 0;
                crosses[j] = !occupied[j] && mask & CROSS_HAS_WORD != 0;
                touched |= occupied[j] || crosses[j];
            }
            let opening = empty_board && lane == center;
            if !touched && !opening {
                continue;
            }
            let (mut fewest, mut seven) = ([0u8; N], [false; N]);
            for first in 0..N {
                if first > 0 && occupied[first - 1] {
                    continue; // a word cannot begin right after a tile
                }
                let (mut tiles, mut joined) = (0usize, false);
                // Placements from `first` only grow as they reach further, so a square's
                // fewest tiles from here come with the first placement that covers it: each
                // square is visited once per starting point.
                let mut uncovered = first;
                for last in first..N {
                    if occupied[last] {
                        joined = true;
                    } else {
                        if !fits[last] || tiles == RACK_SIZE {
                            break;
                        }
                        tiles += 1;
                        joined |= crosses[last] || (opening && last == center);
                    }
                    if last + 1 < N && occupied[last + 1] {
                        continue; // the word runs on through the next tile
                    }
                    if tiles == 0 || !joined || (last == first && !crosses[first]) {
                        continue;
                    }
                    for j in uncovered..=last {
                        if !occupied[j] && (fewest[j] == 0 || tiles as u8 <= fewest[j]) {
                            fewest[j] = tiles as u8;
                        }
                    }
                    uncovered = last + 1;
                    if tiles == RACK_SIZE {
                        // The one seven-tile placement from here (any longer one only runs on
                        // through tiles already down).
                        for j in first..=last {
                            seven[j] |= !occupied[j];
                        }
                        break;
                    }
                }
            }
            for j in 0..N {
                out[index(j)] |= (fewest[j] | if seven[j] { OPEN_SEVEN } else { 0 }) << shift;
            }
        }
    }
}

/// [`open_squares`] for many boards at once, in parallel. `codes` holds `n * 225` bytes, `masks`
/// `n * 450` words and `out` `n * 225` bytes.
pub fn open_squares_batch(codes: &[u8], masks: &[u32], out: &mut [u8]) {
    assert_eq!(codes.len() % SQUARES, 0);
    assert_eq!(masks.len(), codes.len() / SQUARES * CROSS_WORDS);
    assert_eq!(out.len(), codes.len());
    codes
        .par_chunks(SQUARES)
        .zip(masks.par_chunks(CROSS_WORDS))
        .zip(out.par_chunks_mut(SQUARES))
        .with_min_len(64)
        .for_each(|((board, masks), out)| open_squares(board, masks, out));
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
        encode_afterstate(game, &base_codes(game), &game.unseen(0), &described, &mut board, &mut leave, &mut unseen, &mut meta);
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
        assert_eq!(meta[..12], [4, 20 - 5 + 4, 9, 1, 0, 2, 0, 0, 0, 0, 0, -1]);
        assert!(meta[12..].iter().all(|&m| m == 0), "no inference about the opponent's rack");
    }

    #[test]
    fn exchange_and_pass_afterstates() {
        let game = game_with_cat();
        let thrown = Rack::parse("EE").unwrap();
        let (board, leave, unseen, meta) = encode(&game, Move::Exchange(thrown));
        assert_eq!(board, board_codes(&game.board));
        assert_eq!(leave.iter().map(|&c| c as usize).sum::<usize>(), 2);
        assert_eq!(unseen[4], game.unseen(0)[4] + 2, "the two E's thrown back are unseen again");
        assert_eq!(meta[..11], [0, 15, 10, 2, 1, 2, 1, 0, 0, 0, 0]);

        let mut late = game_with_cat();
        late.scoreless_turns = 5;
        late.last_turn[1] = crate::game::LastTurn { kind: 2, tiles: 5, score: 0 };
        let (_, _, _, meta) = encode(&late, Move::Pass);
        // The last three entries: the opponent's previous turn was an exchange of five tiles.
        assert_eq!(meta[..11], [0, 15, 10, 0, 6, 2, 2, 2, 2, 5, 0]);
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

    #[test]
    fn the_opponents_last_placement_is_marked_until_they_move_again() {
        let gaddag = gaddag();
        let racks = [Rack::parse("CAT").unwrap(), Rack::parse("AS").unwrap()];
        let mut game = Game::from_position(Board::new(), racks, &[4u8; 20], [0, 0], 0, 1);
        assert_eq!(base_codes(&game), board_codes(&game.board), "nobody has moved");
        let mut plays = Vec::new();
        game.plays(&gaddag, &mut plays);
        let cat = plays.iter().find(|p| p.to_string().starts_with("8H CAT ")).copied().expect("CAT");
        game.apply(&Move::Play(cat));

        // Player 1 to move: the three tiles player 0 just placed carry the mark.
        let codes = base_codes(&game);
        assert_eq!([codes[CENTER], codes[CENTER + 1], codes[CENTER + 2]], [3 | CODE_OPPONENT, 1 | CODE_OPPONENT, 20 | CODE_OPPONENT]);
        assert_eq!(codes.iter().filter(|&&c| c & CODE_OPPONENT != 0).count(), 3);
        // It reaches every afterstate, next to the mark for the candidate's own tiles.
        plays.clear();
        game.plays(&gaddag, &mut plays);
        let cats = plays.iter().find(|p| p.to_string().starts_with("8H (CAT)S ")).copied().expect("CATS");
        let described = candidate(&game, &LeaveModel::zero(), Move::Play(cats));
        let (mut board, mut leave, mut unseen, mut meta) = ([0; SQUARES], [0; TILE_TYPES], [0; TILE_TYPES], [0; META]);
        encode_afterstate(&game, &codes, &game.unseen(1), &described, &mut board, &mut leave, &mut unseen, &mut meta);
        assert_eq!(board[CENTER + 2], 20 | CODE_OPPONENT);
        assert_eq!(board[CENTER + 3], 19 | CODE_PLACED);
        assert_eq!(meta[M_OPPONENT_SINCE_BINGO], -1);

        // Player 1 plays; now it is player 0's turn and only player 1's tile is marked.
        game.apply(&Move::Play(cats));
        let codes = base_codes(&game);
        assert_eq!(codes.iter().filter(|&&c| c & CODE_OPPONENT != 0).count(), 1);
        assert_eq!(codes[CENTER + 3], 19 | CODE_OPPONENT);
        // A pass by player 0, then by player 1: player 1's last turn was not a placement.
        game.apply(&Move::Pass);
        game.apply(&Move::Pass);
        assert_eq!(base_codes(&game), board_codes(&game.board));
    }

    #[test]
    fn the_inference_about_the_opponents_rack_goes_into_the_metadata() {
        let mut game = game_with_cat();
        let mut counts = [0i16; TILE_TYPES];
        counts[4] = 300;
        counts[BLANK as usize] = 17;
        game.belief[0] = crate::game::Belief { inferred: true, counts };
        game.since_bingo = [3, 5];
        let (_, _, _, meta) = encode(&game, Move::Pass);
        assert_eq!(meta[M_OPPONENT_SINCE_BINGO], 5, "the opponent's count, not the mover's");
        assert_eq!(meta[M_BELIEF_INFERRED], 1);
        assert_eq!(meta[M_BELIEF..M_BELIEF + TILE_TYPES], counts);
        assert_eq!(M_BELIEF + TILE_TYPES, META);
    }

    fn open(codes: &[u8; SQUARES], gaddag: &Gaddag) -> [u8; SQUARES] {
        let mut masks = [0u32; CROSS_WORDS];
        cross_masks(gaddag, codes, &mut masks);
        let mut out = [0xFFu8; SQUARES];
        open_squares(codes, &masks, &mut out);
        out
    }

    #[test]
    fn an_empty_board_is_open_along_the_centre_row_and_column_only() {
        let out = open(&[0; SQUARES], &gaddag());
        for row in 0..N {
            for col in 0..N {
                let (across, down) = (out[row * N + col] & 0x0F, out[row * N + col] >> 4);
                // A first move covers the centre and has at least two tiles, at most seven.
                let expect = |lane: usize, along: usize| {
                    let distance = along.abs_diff(7);
                    if lane != 7 || distance > 6 {
                        0
                    } else {
                        (distance as u8 + 1).max(2) | OPEN_SEVEN
                    }
                };
                assert_eq!(across, expect(row, col), "across at {row},{col}");
                assert_eq!(down, expect(col, row), "down at {row},{col}");
            }
        }
    }

    #[test]
    fn openness_follows_the_tiles_and_the_cross_checks() {
        // CAT on H8-J8 with the test lexicon (AT, TA, CAT, CATS, AS, QI, TAS).
        let (gaddag, game) = (gaddag(), game_with_cat());
        let out = open(&board_codes(&game.board), &gaddag);
        let (across, down) = (|i: usize| out[i] & 0x0F, |i: usize| out[i] >> 4);
        assert_eq!((across(CENTER), down(CENTER)), (0, 0), "occupied squares are not open");
        assert_eq!((across(0), down(0)), (0, 0), "nothing reaches the corner");

        // K8, right of CAT: one tile extends the row; down the column it must be the S of CATS.
        assert_eq!(across(CENTER + 3) & OPEN_TILES, 1);
        assert_eq!(down(CENTER + 3) & OPEN_TILES, 1);
        // G8, left of CAT: the row can be extended (which words exist is not asked), but no
        // letter makes ?CAT, so nothing can be played down through it.
        assert_eq!(across(CENTER - 1) & OPEN_TILES, 1);
        assert_eq!(down(CENTER - 1), 0);

        // The row below. H9 is under the C and no word starts C?: closed, and it blocks the row.
        assert_eq!(across(CENTER + N), 0);
        assert_eq!(across(CENTER + N - 1), 0, "G9 cannot join anything without H9");
        // I9 (under A: AS, AT) and J9 (under T: TA) each take a single tile.
        assert_eq!(across(CENTER + N + 1) & OPEN_TILES, 1);
        assert_eq!(across(CENTER + N + 2) & OPEN_TILES, 1);
        // K9 has no neighbour: it is reached by a play that starts on J9, two tiles.
        assert_eq!(across(CENTER + N + 3) & OPEN_TILES, 2);
        // O9 is the seventh square of a play from I9 (and the sixth from J9).
        assert_eq!(across(CENTER + N + 7) & OPEN_TILES, 6);
        // Seven tiles fit on I9-O9, so every square of that stretch is in a seven-tile lane.
        for col in 8..15 {
            assert_ne!(across(8 * N + col) & OPEN_SEVEN, 0, "column {col}");
        }
        assert_eq!(across(CENTER + N - 1) & OPEN_SEVEN, 0);

        // Down the columns: I7-I1 above the A can be reached from I9 through... no: a down play in
        // column I needs every empty square it fills to fit, and I9 fits A? only as S or T.
        // I9 itself: one tile, forming AS or AT with the row word as its cross-word.
        assert_eq!(down(CENTER + N + 1) & OPEN_TILES, 1);
        // I10: two tiles (I9 and I10), running down from the A.
        assert_eq!(down(CENTER + 2 * N + 1) & OPEN_TILES, 2);
        // The square above the A (I7): ?A... the down word would be [I7] A: no two-letter word
        // ends in A here except TA.
        assert_eq!(down(CENTER - N + 1) & OPEN_TILES, 1);
    }

    #[test]
    fn transposing_the_board_swaps_the_two_halves_of_every_openness_byte() {
        let (gaddag, game) = (gaddag(), game_with_cat());
        let mut codes = board_codes(&game.board);
        codes[CENTER + N + 1] = 19; // an S under the A, so the position is not symmetric
        let out = open(&codes, &gaddag);
        let mut transposed = [0u8; SQUARES];
        for index in 0..SQUARES {
            transposed[index % N * N + index / N] = codes[index];
        }
        let flipped = open(&transposed, &gaddag);
        for index in 0..SQUARES {
            let other = flipped[index % N * N + index / N];
            assert_eq!(out[index], (other << 4) | (other >> 4), "square {index}");
        }
        assert!(out.iter().any(|&b| b != 0));

        let mut masks = vec![0u32; 2 * CROSS_WORDS];
        let boards: Vec<u8> = codes.iter().chain(transposed.iter()).copied().collect();
        cross_masks_batch(&gaddag, &boards, &mut masks);
        let mut batch = vec![0u8; 2 * SQUARES];
        open_squares_batch(&boards, &masks, &mut batch);
        assert_eq!((&batch[..SQUARES], &batch[SQUARES..]), (&out[..], &flipped[..]));
    }

    /// `open_squares` the slow, obvious way: try every run of squares in every lane, and for each
    /// one the rules allow, update every square it covers.
    fn open_squares_reference(codes: &[u8], masks: &[u32]) -> [u8; SQUARES] {
        let mut out = [0u8; SQUARES];
        let empty_board = codes.iter().all(|&code| code & CODE_LETTER == 0);
        for direction in 0..2 {
            for lane in 0..N {
                let index = |j: usize| if direction == 0 { lane * N + j } else { j * N + lane };
                let occupied = |j: usize| codes[index(j)] & CODE_LETTER != 0;
                let mask = |j: usize| masks[direction * SQUARES + index(j)];
                for first in 0..N {
                    for last in first..N {
                        let before = first > 0 && occupied(first - 1);
                        let after = last + 1 < N && occupied(last + 1);
                        let empties: Vec<usize> = (first..=last).filter(|&j| !occupied(j)).collect();
                        let fits = empties.iter().all(|&j| mask(j) & ALL_LETTERS != 0);
                        let through = empties.len() < last - first + 1;
                        let crossing = empties.iter().any(|&j| mask(j) & CROSS_HAS_WORD != 0);
                        let opening = empty_board && lane == 7 && first <= 7 && 7 <= last;
                        let joined = through || crossing || opening;
                        let long_enough = last > first || mask(first) & CROSS_HAS_WORD != 0;
                        if before || after || !fits || empties.is_empty() || empties.len() > 7 || !joined || !long_enough {
                            continue;
                        }
                        for &j in &empties {
                            let half = (out[index(j)] >> (4 * direction)) & 0x0F;
                            let known = half & OPEN_TILES;
                            let fewest = if known == 0 { empties.len() as u8 } else { known.min(empties.len() as u8) };
                            let seven = (half & OPEN_SEVEN) | if empties.len() == 7 { OPEN_SEVEN } else { 0 };
                            out[index(j)] = (out[index(j)] & !(0x0F << (4 * direction))) | ((fewest | seven) << (4 * direction));
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn openness_matches_the_obvious_algorithm_on_random_boards() {
        let mut rng = crate::rng::Rng::new(2026);
        let mut reached = 0usize;
        for trial in 0..400 {
            // Boards from empty to crowded; masks that allow everything, something or nothing.
            let density = trial % 9;
            let mut codes = [0u8; SQUARES];
            let mut masks = [0u32; CROSS_WORDS];
            for code in codes.iter_mut() {
                if rng.below(10) < density as u32 {
                    *code = 1 + rng.below(26) as u8;
                }
            }
            for (i, mask) in masks.iter_mut().enumerate() {
                if codes[i % SQUARES] != 0 {
                    continue; // occupied squares have no mask
                }
                *mask = match rng.below(4) {
                    0 | 1 => ALL_LETTERS,
                    2 => CROSS_HAS_WORD | (1 << rng.below(26)),
                    _ => CROSS_HAS_WORD,
                };
            }
            let mut out = [0u8; SQUARES];
            open_squares(&codes, &masks, &mut out);
            assert_eq!(out, open_squares_reference(&codes, &masks), "trial {trial}");
            reached += out.iter().filter(|&&b| b != 0).count();
        }
        assert!(reached > 10_000, "the random boards exercise the function: {reached}");
    }
}
