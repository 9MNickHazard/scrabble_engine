//! `check_placement`: one placement judged straight from RULES.md "Placing tiles" and
//! "Scoring a placement". Nothing here is shared with the move finders.

use crate::board::RefBoard;
use crate::lexicon::RefLexicon;
use crate::tiles::{premium_at, square_name, tile_value, PlacedTile, Premium, N};

type Grid = [[Option<(u8, bool)>; N]; N];

/// A maximal run of occupied squares: first square, step `(dr, dc)` (either `(0, 1)` across or
/// `(1, 0)` down), and length.
struct Run {
    row: usize,
    col: usize,
    dr: usize,
    dc: usize,
    len: usize,
}

/// Judge one placement straight from the rules. `Ok(score)` if legal, `Err(reason)` if not.
///
/// The tiles may be given in any order. No rack is involved: this only answers "is putting these
/// tiles on these squares legal, and what does it score".
pub fn check_placement(
    board: &RefBoard,
    tiles: &[PlacedTile],
    lexicon: &RefLexicon,
) -> Result<i32, String> {
    // "A placement puts 1 to 7 tiles from the rack onto empty squares."
    if tiles.is_empty() {
        return Err("no tiles placed".to_string());
    }
    if tiles.len() > 7 {
        return Err(format!(
            "{} tiles placed; a placement uses 1 to 7 tiles",
            tiles.len()
        ));
    }
    for t in tiles {
        if t.row as usize >= N || t.col as usize >= N {
            return Err(format!("tile at ({}, {}) is off the board", t.row, t.col));
        }
        if !t.letter.is_ascii_uppercase() {
            return Err(format!(
                "tile on {} has letter byte {}, not A-Z",
                square_name(t.row, t.col),
                t.letter
            ));
        }
    }
    for i in 0..tiles.len() {
        for j in (i + 1)..tiles.len() {
            if tiles[i].row == tiles[j].row && tiles[i].col == tiles[j].col {
                return Err(format!(
                    "two tiles placed on {}",
                    square_name(tiles[i].row, tiles[i].col)
                ));
            }
        }
    }
    for t in tiles {
        if board.cells[t.row as usize][t.col as usize].is_some() {
            return Err(format!("{} is already occupied", square_name(t.row, t.col)));
        }
    }

    // The board after the placement, and which squares were covered this turn.
    let mut after: Grid = board.cells;
    let mut is_new = [[false; N]; N];
    for t in tiles {
        after[t.row as usize][t.col as usize] = Some((t.letter, t.blank));
        is_new[t.row as usize][t.col as usize] = true;
    }

    // Rules 1 and 2 apply to two or more tiles ("a single placed tile has no line of its own").
    // The step along the line: (0, 1) across, (1, 0) down.
    let mut line_step: Option<(usize, usize)> = None;
    if tiles.len() >= 2 {
        // Rule 1, one line.
        let same_row = tiles.iter().all(|t| t.row == tiles[0].row);
        let same_col = tiles.iter().all(|t| t.col == tiles[0].col);
        line_step = Some(if same_row {
            (0, 1)
        } else if same_col {
            (1, 0)
        } else {
            return Err("the tiles are not all in one row or one column".to_string());
        });

        // Rule 2, no gaps: every square from the first placed tile to the last is occupied.
        // (One of the two ranges below is a single row or column.)
        let first_row = tiles.iter().map(|t| t.row as usize).min().unwrap();
        let last_row = tiles.iter().map(|t| t.row as usize).max().unwrap();
        let first_col = tiles.iter().map(|t| t.col as usize).min().unwrap();
        let last_col = tiles.iter().map(|t| t.col as usize).max().unwrap();
        for r in first_row..=last_row {
            for c in first_col..=last_col {
                if after[r][c].is_none() {
                    // Fixed text, no square name: the brute-force finder hits this rejection
                    // millions of times and formatting the message doubled its run time.
                    return Err("gap between the placed tiles".to_string());
                }
            }
        }
    }

    // Rule 5, connection.
    let board_was_empty = board
        .cells
        .iter()
        .all(|row| row.iter().all(|cell| cell.is_none()));
    if board_was_empty {
        if !tiles.iter().any(|t| t.row == 7 && t.col == 7) {
            return Err("the board is empty and no tile is on H8".to_string());
        }
    } else if !tiles
        .iter()
        .any(|t| next_to_old_tile(board, t.row as usize, t.col as usize))
    {
        return Err("no placed tile is next to a tile already on the board".to_string());
    }

    // The words the placement forms.
    let mut words: Vec<Run> = Vec::new();
    match line_step {
        None => {
            // "A single placed tile has no line of its own": every run of 2+ through it, across and down.
            let (r, c) = (tiles[0].row as usize, tiles[0].col as usize);
            let across = run_through(&after, r, c, 0, 1);
            let down = run_through(&after, r, c, 1, 0);
            if across.len >= 2 {
                words.push(across);
            }
            if down.len >= 2 {
                words.push(down);
            }
            if words.is_empty() {
                return Err(format!(
                    "the single tile on {} forms no word of two or more letters",
                    square_name(tiles[0].row, tiles[0].col)
                ));
            }
        }
        Some((dr, dc)) => {
            // Rule 3, main word: the maximal run along the line through the placed tiles.
            words.push(run_through(
                &after,
                tiles[0].row as usize,
                tiles[0].col as usize,
                dr,
                dc,
            ));

            // Rule 4, cross-words: the maximal run through each placed tile in the other direction.
            for t in tiles {
                let cross = run_through(&after, t.row as usize, t.col as usize, dc, dr);
                if cross.len >= 2 {
                    words.push(cross);
                }
            }
        }
    }

    // Every word formed must be in the lexicon.
    for run in &words {
        let letters = run_letters(&after, run);
        if !lexicon.contains(&letters) {
            return Err(format!(
                "{} is not in the lexicon",
                String::from_utf8_lossy(&letters)
            ));
        }
    }

    // Scoring: every word's score, plus 50 for using 7 tiles.
    let mut score = 0;
    for run in &words {
        score += word_score(&after, &is_new, run);
    }
    if tiles.len() == 7 {
        score += 50;
    }
    Ok(score)
}

/// True when one of the four squares sharing an edge with `(r, c)` held a tile before this turn.
fn next_to_old_tile(board: &RefBoard, r: usize, c: usize) -> bool {
    (r > 0 && board.cells[r - 1][c].is_some())
        || (r + 1 < N && board.cells[r + 1][c].is_some())
        || (c > 0 && board.cells[r][c - 1].is_some())
        || (c + 1 < N && board.cells[r][c + 1].is_some())
}

/// The longest unbroken run of occupied squares through `(r, c)` in direction `(dr, dc)`.
fn run_through(grid: &Grid, r: usize, c: usize, dr: usize, dc: usize) -> Run {
    // Walk back to the first square of the run.
    let (mut sr, mut sc) = (r, c);
    while sr >= dr && sc >= dc && grid[sr - dr][sc - dc].is_some() {
        sr -= dr;
        sc -= dc;
    }
    // Walk forward counting squares.
    let mut len = 0;
    let (mut er, mut ec) = (sr, sc);
    while er < N && ec < N && grid[er][ec].is_some() {
        len += 1;
        er += dr;
        ec += dc;
    }
    Run {
        row: sr,
        col: sc,
        dr,
        dc,
        len,
    }
}

fn run_letters(grid: &Grid, run: &Run) -> Vec<u8> {
    (0..run.len)
        .map(|i| grid[run.row + i * run.dr][run.col + i * run.dc].unwrap().0)
        .collect()
}

/// RULES.md: the sum of the letters' values, a letter on a double or triple letter square covered
/// this turn counting double or triple, then doubled or tripled for each double or triple word
/// square covered this turn in the word. Blanks are worth 0. Old tiles get no premium.
fn word_score(grid: &Grid, is_new: &[[bool; N]; N], run: &Run) -> i32 {
    let mut sum = 0;
    let mut word_multiplier = 1;
    for i in 0..run.len {
        let (r, c) = (run.row + i * run.dr, run.col + i * run.dc);
        let (letter, blank) = grid[r][c].unwrap();
        let value = if blank { 0 } else { tile_value(letter) };
        if is_new[r][c] {
            match premium_at(r as u8, c as u8) {
                Premium::Plain => sum += value,
                Premium::DoubleLetter => sum += 2 * value,
                Premium::TripleLetter => sum += 3 * value,
                Premium::DoubleWord => {
                    sum += value;
                    word_multiplier *= 2;
                }
                Premium::TripleWord => {
                    sum += value;
                    word_multiplier *= 3;
                }
            }
        } else {
            sum += value;
        }
    }
    sum * word_multiplier
}
