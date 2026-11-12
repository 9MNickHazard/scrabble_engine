//! `legal_plays`: find candidates by sliding every lexicon word along every row and column, then
//! let `check_placement` decide legality and score. It shares no logic with `check_placement`; the
//! filters here are only necessary conditions that throw away candidates early.

use std::collections::BTreeSet;

use crate::board::RefBoard;
use crate::judge::check_placement;
use crate::lexicon::{LexWord, RefLexicon};
use crate::tiles::{parse_rack, PlacedTile, RefPlay, N};

/// The rack as letter counts.
struct Rack {
    letters: [u8; 26],
    blanks: u32,
    /// The most tiles one play may use: `min(rack size, 7)`.
    max_tiles: usize,
}

/// One row or column of the board.
struct Line {
    /// The board square of each position along the line.
    squares: [(u8, u8); N],
    /// What is on each square before the play.
    cells: [Option<(u8, bool)>; N],
    /// Whether a new tile on this square could satisfy the connection rule.
    hook: [bool; N],
    /// How many of each letter already lie on this line (blanks counted as their letter).
    letters: [u8; 26],
}

/// Every legal placement that can be made from this rack (upper-case letters, `?` for a blank),
/// sorted, with no duplicates. Different blank choices are different plays.
///
/// Method: for every row and every column, every lexicon word and every start square where it
/// fits, keep it when it matches the tiles already on the line, starts and ends at the edges of a
/// maximal run, covers 1 to 7 empty squares including one that touches an old tile (or H8 on an
/// empty board), and its new letters can come from the rack. Every distinct choice of which new
/// letters are blanks is a candidate, and `check_placement` has the final say on each.
pub fn legal_plays(board: &RefBoard, rack: &str, lexicon: &RefLexicon) -> Vec<RefPlay> {
    let rack_tiles = parse_rack(rack);
    let mut counts = Rack {
        letters: [0; 26],
        blanks: 0,
        max_tiles: rack_tiles.len().min(7),
    };
    for &t in &rack_tiles {
        if t == b'?' {
            counts.blanks += 1;
        } else {
            counts.letters[(t - b'A') as usize] += 1;
        }
    }
    let mut found: BTreeSet<RefPlay> = BTreeSet::new();
    if counts.max_tiles == 0 {
        return Vec::new();
    }

    let hooks = hook_squares(board);
    for line in all_lines(board, &hooks) {
        if !line.hook.iter().any(|&h| h) {
            continue; // no square on this line could make a connected play
        }
        for len in 2..=N {
            for word in lexicon.words_of_len(len) {
                if !rack_and_line_could_supply(word, &counts, &line) {
                    continue;
                }
                for start in 0..=(N - len) {
                    try_word_at(board, lexicon, &counts, &line, word, start, &mut found);
                }
            }
        }
    }
    found.into_iter().collect()
}

/// Squares where a new tile would satisfy the connection rule: H8 on an empty board, otherwise
/// every empty square sharing an edge with a tile.
fn hook_squares(board: &RefBoard) -> [[bool; N]; N] {
    let mut hooks = [[false; N]; N];
    let mut any_tile = false;
    for r in 0..N {
        for c in 0..N {
            if board.cells[r][c].is_some() {
                any_tile = true;
                continue;
            }
            let up = r > 0 && board.cells[r - 1][c].is_some();
            let down = r + 1 < N && board.cells[r + 1][c].is_some();
            let left = c > 0 && board.cells[r][c - 1].is_some();
            let right = c + 1 < N && board.cells[r][c + 1].is_some();
            hooks[r][c] = up || down || left || right;
        }
    }
    if !any_tile {
        hooks[7][7] = true;
    }
    hooks
}

/// The 15 rows (left to right) and 15 columns (top to bottom).
fn all_lines(board: &RefBoard, hooks: &[[bool; N]; N]) -> Vec<Line> {
    let mut lines = Vec::with_capacity(2 * N);
    for fixed in 0..N {
        for across in [true, false] {
            let mut line = Line {
                squares: [(0, 0); N],
                cells: [None; N],
                hook: [false; N],
                letters: [0; 26],
            };
            for i in 0..N {
                let (r, c) = if across { (fixed, i) } else { (i, fixed) };
                line.squares[i] = (r as u8, c as u8);
                line.cells[i] = board.cells[r][c];
                line.hook[i] = hooks[r][c];
                if let Some((letter, _)) = board.cells[r][c] {
                    line.letters[(letter - b'A') as usize] += 1;
                }
            }
            lines.push(line);
        }
    }
    lines
}

/// A quick necessary condition: every letter of the word must come from the rack, the blanks, or a
/// tile already on this line.
fn rack_and_line_could_supply(word: &LexWord, rack: &Rack, line: &Line) -> bool {
    let mut short = 0u32;
    for i in 0..26 {
        let have = rack.letters[i] as u32 + line.letters[i] as u32;
        let need = word.counts[i] as u32;
        if need > have {
            short += need - have;
        }
    }
    short <= rack.blanks
}

/// Try `word` with its first letter at position `start` of `line`.
fn try_word_at(
    board: &RefBoard,
    lexicon: &RefLexicon,
    rack: &Rack,
    line: &Line,
    word: &LexWord,
    start: usize,
    found: &mut BTreeSet<RefPlay>,
) {
    let letters = &word.letters;
    let end = start + letters.len(); // one past the last letter
                                     // The word must be a whole run: nothing directly before or after it.
    if start > 0 && line.cells[start - 1].is_some() {
        return;
    }
    if end < N && line.cells[end].is_some() {
        return;
    }
    // Old tiles must match; empty squares get new tiles.
    let mut new_positions: Vec<usize> = Vec::new();
    let mut touches_hook = false;
    for (i, &letter) in letters.iter().enumerate() {
        let pos = start + i;
        match line.cells[pos] {
            Some((old, _)) => {
                if old != letter {
                    return;
                }
            }
            None => {
                new_positions.push(pos);
                touches_hook |= line.hook[pos];
            }
        }
    }
    if new_positions.is_empty() || new_positions.len() > rack.max_tiles || !touches_hook {
        return;
    }

    // Every way of supplying the new letters: bit j of `blank_mask` set = the j-th new letter is a
    // blank. Each feasible mask is a different play.
    let n = new_positions.len();
    for blank_mask in 0u32..(1u32 << n) {
        if blank_mask.count_ones() > rack.blanks {
            continue;
        }
        let mut real_needed = [0u8; 26];
        for (j, &pos) in new_positions.iter().enumerate() {
            if blank_mask & (1 << j) == 0 {
                real_needed[(letters[pos - start] - b'A') as usize] += 1;
            }
        }
        if (0..26).any(|i| real_needed[i] > rack.letters[i]) {
            continue;
        }
        let mut tiles: Vec<PlacedTile> = new_positions
            .iter()
            .enumerate()
            .map(|(j, &pos)| PlacedTile {
                row: line.squares[pos].0,
                col: line.squares[pos].1,
                letter: letters[pos - start],
                blank: blank_mask & (1 << j) != 0,
            })
            .collect();
        tiles.sort();
        if let Ok(score) = check_placement(board, &tiles, lexicon) {
            found.insert(RefPlay { tiles, score });
        }
    }
}
