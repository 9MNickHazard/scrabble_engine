//! Checking and scoring a move that did not come from the generator.
//!
//! This is a second, separate implementation of the placement rules and the scoring in
//! docs/RULES.md. It builds every word a placement forms and looks each one up, instead of
//! walking the GADDAG with cross-check sets the way the generator does. Human-entered moves go
//! through it, and the tests require it to agree with the generator on every generated play.

use crate::board::{square_letter, square_value, Board, CENTER, EMPTY, LETTER_MULT, N, SQUARES, WORD_MULT};
use crate::gaddag::Gaddag;
use crate::game::Game;
use crate::moves::{Move, Placed, Play, PLACED_BLANK, PLAYED_THROUGH};
use crate::tiles::{BINGO_BONUS, RACK_SIZE};

/// A maximal run of occupied squares through one square, along a row or a column.
struct Run {
    /// Board index of the first square.
    first: usize,
    /// Index step between squares: 1 across, 15 down.
    step: usize,
    len: usize,
}

fn run_through(cells: &[u8; SQUARES], index: usize, down: bool) -> Run {
    let (row, col) = (index / N, index % N);
    let (step, position) = if down { (N, row) } else { (1, col) };
    let mut before = 0;
    while before < position && cells[index - (before + 1) * step] != EMPTY {
        before += 1;
    }
    let mut after = 0;
    while position + after + 1 < N && cells[index + (after + 1) * step] != EMPTY {
        after += 1;
    }
    Run { first: index - before * step, step, len: before + 1 + after }
}

/// Score of the word on `run`, with premiums counting only under squares in `is_new`.
fn word_score(cells: &[u8; SQUARES], is_new: &[bool; SQUARES], run: &Run) -> i32 {
    let (mut total, mut multiplier) = (0, 1);
    for i in 0..run.len {
        let index = run.first + i * run.step;
        let value = square_value(cells[index]);
        if is_new[index] {
            total += value * LETTER_MULT[index] as i32;
            multiplier *= WORD_MULT[index] as i32;
        } else {
            total += value;
        }
    }
    total * multiplier
}

fn is_word(gaddag: &Gaddag, cells: &[u8; SQUARES], run: &Run) -> bool {
    let mut letters = [0u8; N];
    for (i, letter) in letters.iter_mut().enumerate().take(run.len) {
        *letter = square_letter(cells[run.first + i * run.step]);
    }
    gaddag.contains(&letters[..run.len])
}

fn word_text(cells: &[u8; SQUARES], run: &Run) -> String {
    (0..run.len).map(|i| (b'A' + square_letter(cells[run.first + i * run.step])) as char).collect()
}

/// Check a set of tiles against the placement rules and return the play in canonical form,
/// with its score. The rack is not consulted: see [`check_move`].
pub fn check_placement(gaddag: &Gaddag, board: &Board, tiles: &[Placed]) -> Result<Play, String> {
    if tiles.is_empty() {
        return Err("a placement needs at least one tile".into());
    }
    if tiles.len() > RACK_SIZE {
        return Err(format!("{} tiles placed; a rack holds {RACK_SIZE}", tiles.len()));
    }
    let mut cells = board.cells;
    let mut is_new = [false; SQUARES];
    for tile in tiles {
        if tile.index >= SQUARES || tile.letter >= 26 {
            return Err("tile is off the board or not a letter".into());
        }
        if cells[tile.index] != EMPTY {
            return Err("a tile is placed on an occupied square".into());
        }
        cells[tile.index] = tile.square();
        is_new[tile.index] = true;
    }

    let same_row = tiles.iter().all(|t| t.index / N == tiles[0].index / N);
    let same_col = tiles.iter().all(|t| t.index % N == tiles[0].index % N);
    if !same_row && !same_col {
        return Err("the tiles are not all in one row or one column".into());
    }

    if board.is_empty() {
        if !is_new[CENTER] {
            return Err("the first play must cover the centre square".into());
        }
    } else {
        let touches = tiles.iter().any(|t| {
            let (row, col) = (t.index / N, t.index % N);
            (row > 0 && board.cells[t.index - N] != EMPTY)
                || (row + 1 < N && board.cells[t.index + N] != EMPTY)
                || (col > 0 && board.cells[t.index - 1] != EMPTY)
                || (col + 1 < N && board.cells[t.index + 1] != EMPTY)
        });
        if !touches {
            return Err("the play does not touch a tile already on the board".into());
        }
    }

    // A lone tile is described across if it makes an across word, otherwise down.
    let down = if tiles.len() == 1 { run_through(&cells, tiles[0].index, false).len < 2 } else { same_col };
    let main = run_through(&cells, tiles[0].index, down);
    if main.len < 2 {
        return Err("the play forms no word of two or more letters".into());
    }
    let last = main.first + (main.len - 1) * main.step;
    if tiles.iter().any(|t| t.index < main.first || t.index > last) {
        return Err("the placed tiles are not joined into one word (there is a gap)".into());
    }
    if !is_word(gaddag, &cells, &main) {
        return Err(format!("{} is not a word", word_text(&cells, &main)));
    }

    let mut score = word_score(&cells, &is_new, &main);
    for tile in tiles {
        let cross = run_through(&cells, tile.index, !down);
        if cross.len >= 2 {
            if !is_word(gaddag, &cells, &cross) {
                return Err(format!("{} is not a word", word_text(&cells, &cross)));
            }
            score += word_score(&cells, &is_new, &cross);
        }
    }
    if tiles.len() == RACK_SIZE {
        score += BINGO_BONUS;
    }

    let mut word = [0u8; N];
    for (i, code) in word.iter_mut().enumerate().take(main.len) {
        let index = main.first + i * main.step;
        let letter = square_letter(cells[index]);
        *code = if !is_new[index] {
            letter | PLAYED_THROUGH
        } else if cells[index] & crate::board::BLANK_BIT != 0 {
            letter | PLACED_BLANK
        } else {
            letter
        };
    }
    let (row, col) = (main.first / N, main.first % N);
    Ok(Play {
        down,
        lane: if down { col } else { row } as u8,
        start: if down { row } else { col } as u8,
        len: main.len as u8,
        tiles_played: tiles.len() as u8,
        score: score as i16,
        word,
    })
}

/// Check that `mv` is legal for the player to move in `game`: the placement rules, the rack, the
/// exchange limit, and that a play's recorded word and score are right.
pub fn check_move(gaddag: &Gaddag, game: &Game, mv: &Move) -> Result<(), String> {
    if game.over {
        return Err("the game is over".into());
    }
    let rack = &game.racks[game.to_move];
    match mv {
        Move::Pass => Ok(()),
        Move::Exchange(tiles) => {
            if !game.can_exchange() {
                return Err("an exchange needs at least seven tiles in the bag".into());
            }
            if tiles.is_empty() {
                return Err("an exchange must return at least one tile".into());
            }
            if !rack.contains_all(tiles) {
                return Err("the rack does not hold the tiles to exchange".into());
            }
            Ok(())
        }
        Move::Play(play) => {
            if play.len as usize > N || play.start as usize + play.len as usize > N || play.lane as usize >= N {
                return Err("the play runs off the board".into());
            }
            let tiles: Vec<Placed> = play.placed().collect();
            if !rack.contains_all(&play.tiles_used()) {
                return Err("the rack does not hold the tiles for this play".into());
            }
            let checked = check_placement(gaddag, &game.board, &tiles)?;
            if checked != *play {
                return Err(format!("the play is recorded as {play} but is really {checked}"));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{make_square, parse_square};
    use crate::tiles::Rack;

    fn gaddag() -> Gaddag {
        Gaddag::build(&["AT", "TA", "CAT", "CATS", "AS", "OX", "QI", "IT", "RETAINS", "SI"])
    }

    fn tiles(spec: &str) -> Vec<Placed> {
        spec.split_whitespace()
            .map(|part| {
                let (square, letter) = part.split_once('=').unwrap();
                let (row, col) = parse_square(square).unwrap();
                let c = letter.chars().next().unwrap();
                Placed { index: row * N + col, letter: c.to_ascii_uppercase() as u8 - b'A', blank: c.is_ascii_lowercase() }
            })
            .collect()
    }

    fn board_with(spec: &str) -> Board {
        let mut board = Board::new();
        for tile in tiles(spec) {
            board.set(tile.index, make_square(tile.letter, tile.blank));
        }
        board
    }

    fn check(board: &Board, spec: &str) -> Result<String, String> {
        check_placement(&gaddag(), board, &tiles(spec)).map(|p| p.to_string())
    }

    #[test]
    fn the_legacy_engine_bugs_are_rejected_or_scored_correctly() {
        let cat = board_with("H8=C I8=A J8=T");
        // Old engine accepted OX after CAT, leaving CATOX on the board.
        assert!(check(&cat, "K8=O L8=X").unwrap_err().contains("CATOX"));
        // Old engine let a move place no tiles.
        assert!(check_placement(&gaddag(), &cat, &[]).is_err());
        assert!(check(&cat, "H8=C").unwrap_err().contains("occupied"));
        // A blank scores 0 when played...
        assert_eq!(check(&Board::new(), "H8=q I8=I").unwrap(), "8H qI 2");
        // ...and on every later turn: IT down from the I next to a blank Q, then QI re-read.
        let blank_q = board_with("H8=q I8=I");
        assert_eq!(check(&blank_q, "I9=T").unwrap(), "I8 (I)T 3");
        assert_eq!(check(&blank_q, "J8=T").unwrap_err(), "QIT is not a word");
    }

    #[test]
    fn placement_rules() {
        let empty = Board::new();
        assert_eq!(check(&empty, "H8=C I8=A J8=T").unwrap(), "8H CAT 10");
        assert!(check(&empty, "A1=C B1=A C1=T").unwrap_err().contains("centre"));
        assert!(check(&empty, "H8=A").unwrap_err().contains("no word"));
        assert!(check(&empty, "H8=C I8=A K8=T").unwrap_err().contains("gap"));
        assert!(check(&empty, "H8=C I9=A").unwrap_err().contains("one row or one column"));
        assert!(check(&empty, "H8=T I8=C").unwrap_err().contains("TC is not a word"));

        let cat = board_with("H8=C I8=A J8=T");
        assert!(check(&cat, "A1=A B1=T").unwrap_err().contains("touch"));
        assert!(check(&cat, "H10=A I10=T").unwrap_err().contains("touch"));
        // Gap: tiles either side of an empty square in the same row.
        assert!(check(&cat, "G8=S L8=S").is_err());
        // Hook and cross-word together: S on K8 makes CATS (6); with A above at K7 also AS.
        assert_eq!(check(&cat, "K8=S").unwrap(), "8H (CAT)S 6");
        let with_a = board_with("H8=C I8=A J8=T K7=A");
        assert_eq!(check(&with_a, "K8=S").unwrap(), "8H (CAT)S 8");
        // A lone tile that only makes a down word is a down play.
        assert_eq!(check(&cat, "I9=S").unwrap(), "I8 (A)S 3");
        // Invalid cross-word.
        assert!(check(&cat, "H9=A I9=T").unwrap_err().contains("is not a word"));
    }

    #[test]
    fn bingo_and_premiums() {
        // RETAINS from H8 across: R(1) E T A(DL at L8) I N S = 1+1+1+2+1+1+1 = 8, x2 centre, +50.
        assert_eq!(check(&Board::new(), "H8=R I8=E J8=T K8=A L8=I M8=N N8=S").unwrap(), "8H RETAINS 66");
        // Same with a blank for the A: the double letter square doubles zero.
        assert_eq!(check(&Board::new(), "H8=R I8=E J8=T K8=a L8=I M8=N N8=S").unwrap(), "8H RETaINS 64");
    }

    #[test]
    fn check_move_ties_a_play_to_the_rack_and_its_recorded_score() {
        let gaddag = gaddag();
        let racks = [Rack::parse("CAT").unwrap(), Rack::parse("OX").unwrap()];
        let game = Game::from_position(Board::new(), racks, &[0; 20], [0, 0], 0, 1);
        let play = check_placement(&gaddag, &game.board, &tiles("H8=C I8=A J8=T")).unwrap();
        assert!(check_move(&gaddag, &game, &Move::Play(play)).is_ok());

        let mut wrong_score = play;
        wrong_score.score += 1;
        assert!(check_move(&gaddag, &game, &Move::Play(wrong_score)).is_err());

        let not_held = check_placement(&gaddag, &game.board, &tiles("H8=O I8=X")).unwrap();
        assert!(check_move(&gaddag, &game, &Move::Play(not_held)).unwrap_err().contains("rack"));

        assert!(check_move(&gaddag, &game, &Move::Pass).is_ok());
        assert!(check_move(&gaddag, &game, &Move::Exchange(Rack::parse("CA").unwrap())).is_ok());
        assert!(check_move(&gaddag, &game, &Move::Exchange(Rack::parse("ZZ").unwrap())).is_err());
        assert!(check_move(&gaddag, &game, &Move::Exchange(Rack::new())).is_err());
        let small_bag = Game::from_position(Board::new(), racks, &[0; 6], [0, 0], 0, 1);
        assert!(check_move(&gaddag, &small_bag, &Move::Exchange(Rack::parse("C").unwrap())).is_err());
    }
}
