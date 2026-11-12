//! `legal_plays_exhaustive`: the dumbest possible move finder. It knows nothing about words; it
//! puts rack tiles on every set of empty squares in a row or column, tries every letter for each
//! blank, and keeps whatever `check_placement` accepts. Only usable for small racks.

use std::collections::BTreeSet;

use crate::board::RefBoard;
use crate::judge::check_placement;
use crate::lexicon::RefLexicon;
use crate::tiles::{parse_rack, PlacedTile, RefPlay, N};

/// Same result as `legal_plays`, by brute force. The work grows like
/// (empty squares per line choose k) x k! x 26^(blanks); keep racks to about 4 tiles.
pub fn legal_plays_exhaustive(board: &RefBoard, rack: &str, lexicon: &RefLexicon) -> Vec<RefPlay> {
    let rack_tiles = parse_rack(rack);
    let max_tiles = rack_tiles.len().min(7);

    // Every set of 1..=max_tiles empty squares lying in one row or one column (each set sorted).
    // A single square turns up from its row and its column; the set keeps one copy.
    let mut square_sets: BTreeSet<Vec<(u8, u8)>> = BTreeSet::new();
    for fixed in 0..N {
        let row_empties: Vec<(u8, u8)> = (0..N)
            .filter(|&c| board.cells[fixed][c].is_none())
            .map(|c| (fixed as u8, c as u8))
            .collect();
        let col_empties: Vec<(u8, u8)> = (0..N)
            .filter(|&r| board.cells[r][fixed].is_none())
            .map(|r| (r as u8, fixed as u8))
            .collect();
        add_subsets(
            &row_empties,
            max_tiles,
            &mut Vec::new(),
            0,
            &mut square_sets,
        );
        add_subsets(
            &col_empties,
            max_tiles,
            &mut Vec::new(),
            0,
            &mut square_sets,
        );
    }

    // For each length m, every distinct ordered sequence of m tiles taken from the rack.
    let sequences: Vec<BTreeSet<Vec<u8>>> = (0..=max_tiles)
        .map(|m| ordered_selections(&rack_tiles, m))
        .collect();

    let mut found: BTreeSet<RefPlay> = BTreeSet::new();
    let mut tiles: Vec<PlacedTile> = Vec::with_capacity(max_tiles);
    for squares in &square_sets {
        for sequence in &sequences[squares.len()] {
            let blank_count = sequence.iter().filter(|&&t| t == b'?').count();
            for code in 0..26usize.pow(blank_count as u32) {
                // Decode `code` as one letter per blank, in base 26.
                let mut rest = code;
                tiles.clear();
                for (i, &tile) in sequence.iter().enumerate() {
                    let (letter, blank) = if tile == b'?' {
                        let letter = b'A' + (rest % 26) as u8;
                        rest /= 26;
                        (letter, true)
                    } else {
                        (tile, false)
                    };
                    tiles.push(PlacedTile {
                        row: squares[i].0,
                        col: squares[i].1,
                        letter,
                        blank,
                    });
                }
                if let Ok(score) = check_placement(board, &tiles, lexicon) {
                    let mut sorted = tiles.clone();
                    sorted.sort();
                    found.insert(RefPlay {
                        tiles: sorted,
                        score,
                    });
                }
            }
        }
    }
    found.into_iter().collect()
}

/// Add every non-empty subset of `items[from..]` extended from `chosen`, up to `max` items.
fn add_subsets(
    items: &[(u8, u8)],
    max: usize,
    chosen: &mut Vec<(u8, u8)>,
    from: usize,
    out: &mut BTreeSet<Vec<(u8, u8)>>,
) {
    if !chosen.is_empty() {
        out.insert(chosen.clone());
    }
    if chosen.len() == max {
        return;
    }
    for i in from..items.len() {
        chosen.push(items[i]);
        add_subsets(items, max, chosen, i + 1, out);
        chosen.pop();
    }
}

/// Every distinct ordered sequence of `m` tiles drawn without replacement from `rack`.
fn ordered_selections(rack: &[u8], m: usize) -> BTreeSet<Vec<u8>> {
    fn extend(
        rack: &[u8],
        used: &mut Vec<bool>,
        current: &mut Vec<u8>,
        m: usize,
        out: &mut BTreeSet<Vec<u8>>,
    ) {
        if current.len() == m {
            out.insert(current.clone());
            return;
        }
        for i in 0..rack.len() {
            if !used[i] {
                used[i] = true;
                current.push(rack[i]);
                extend(rack, used, current, m, out);
                current.pop();
                used[i] = false;
            }
        }
    }
    let mut out = BTreeSet::new();
    extend(
        rack,
        &mut vec![false; rack.len()],
        &mut Vec::new(),
        m,
        &mut out,
    );
    out
}
