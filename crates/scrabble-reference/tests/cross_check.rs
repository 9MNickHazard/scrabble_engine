//! `legal_plays` against the brute-force `legal_plays_exhaustive` on positions built by playing
//! random legal plays with a small made-up lexicon, and against itself on the transposed board.

mod common;

use std::collections::BTreeSet;
use std::time::Instant;

use common::*;
use scrabble_reference::*;

fn describe(board: &RefBoard, rack: &str) -> String {
    format!("rack {rack}, board:\n{}", board.to_text())
}

fn assert_same_plays(board: &RefBoard, rack: &str, fast: &[RefPlay], slow: &[RefPlay]) {
    if fast == slow {
        return;
    }
    let fast_set: BTreeSet<&RefPlay> = fast.iter().collect();
    let slow_set: BTreeSet<&RefPlay> = slow.iter().collect();
    let only_fast: Vec<String> = fast_set
        .difference(&slow_set)
        .map(|p| p.to_string())
        .collect();
    let only_slow: Vec<String> = slow_set
        .difference(&fast_set)
        .map(|p| p.to_string())
        .collect();
    panic!(
        "legal_plays and legal_plays_exhaustive differ on {}\nonly in legal_plays: {:?}\nonly in exhaustive: {:?}",
        describe(board, rack),
        only_fast,
        only_slow
    );
}

/// Sorted by (tiles, score), strictly increasing (so no duplicates), each tile list sorted by
/// square, and each play accepted by check_placement with the same score.
fn assert_well_formed(board: &RefBoard, rack: &str, plays: &[RefPlay], lexicon: &RefLexicon) {
    for pair in plays.windows(2) {
        assert!(
            pair[0] < pair[1],
            "not sorted or duplicated: {} / {}",
            pair[0],
            pair[1]
        );
    }
    let squares: BTreeSet<&Vec<PlacedTile>> = plays.iter().map(|p| &p.tiles).collect();
    assert_eq!(squares.len(), plays.len(), "same tiles with two scores");
    for play in plays {
        for pair in play.tiles.windows(2) {
            assert!(
                (pair[0].row, pair[0].col) < (pair[1].row, pair[1].col),
                "tiles not sorted: {play}"
            );
        }
        assert_eq!(
            check_placement(board, &play.tiles, lexicon),
            Ok(play.score),
            "{play} on {}",
            describe(board, rack)
        );
    }
}

/// `count` positions from seed `seed`: a board after 0-12 random legal plays, then a rack of 1-4
/// tiles. Blanks: up to 2 on racks of 1-3 tiles, up to 1 on racks of 4 (to keep the brute force
/// affordable).
fn cross_check(seed: u64, count: usize) {
    let lexicon = small_lexicon();
    let mut rng = Rng::new(seed);
    let started = Instant::now();
    let mut total_plays = 0;
    let mut blank_boards = 0;
    for _ in 0..count {
        let moves = rng.below(13);
        let boards = play_game(&lexicon, &mut rng, moves, Choice::Random);
        let board = boards.last().unwrap();
        if board
            .cells
            .iter()
            .flatten()
            .any(|c| matches!(c, Some((_, true))))
        {
            blank_boards += 1;
        }
        let size = 1 + rng.below(4);
        let max_blanks = if size == 4 { 1 } else { size.min(2) };
        let blanks = rng.below(max_blanks + 1);
        let rack = random_rack(&mut rng, size - blanks, blanks);

        let fast = legal_plays(board, &rack, &lexicon);
        let brute_started = Instant::now();
        let slow = legal_plays_exhaustive(board, &rack, &lexicon);
        let brute_time = brute_started.elapsed();
        assert_same_plays(board, &rack, &fast, &slow);
        assert_well_formed(board, &rack, &fast, &lexicon);
        total_plays += fast.len();
        let tiles_on_board = board.cells.iter().flatten().filter(|c| c.is_some()).count();
        println!(
            "  seed {seed}: rack {rack:<4} {tiles_on_board:>3} tiles on board, {:>4} plays, brute force {brute_time:.1?}",
            fast.len()
        );
    }
    println!(
        "seed {seed}: {count} positions ({blank_boards} with a blank on the board), {total_plays} plays, all equal, {:.1?}",
        started.elapsed()
    );
}

#[test]
fn legal_plays_matches_exhaustive_1() {
    cross_check(1, 10);
}

#[test]
fn legal_plays_matches_exhaustive_2() {
    cross_check(2, 10);
}

#[test]
fn legal_plays_matches_exhaustive_3() {
    cross_check(3, 10);
}

#[test]
fn legal_plays_matches_exhaustive_4() {
    cross_check(4, 10);
}

/// Racks of 1-2 tiles where every tile may be a blank, on the opening board and later boards.
#[test]
fn legal_plays_matches_exhaustive_blank_racks() {
    let lexicon = small_lexicon();
    let mut rng = Rng::new(99);
    for rack in ["?", "??", "A?", "S?"] {
        for moves in [0, 1, 4, 9] {
            let boards = play_game(&lexicon, &mut rng, moves, Choice::Random);
            let board = boards.last().unwrap();
            let fast = legal_plays(board, rack, &lexicon);
            let slow = legal_plays_exhaustive(board, rack, &lexicon);
            assert_same_plays(board, rack, &fast, &slow);
            assert_well_formed(board, rack, &fast, &lexicon);
        }
    }
}

/// Reflecting the board in the main diagonal must reflect the set of plays and keep every score
/// (the premium layout is symmetric). Uses full 7-tile racks, too big for the brute force.
#[test]
fn transposed_board_gives_transposed_plays() {
    let lexicon = small_lexicon();
    let mut rng = Rng::new(5);
    for i in 0..30 {
        let moves = rng.below(15);
        let boards = play_game(&lexicon, &mut rng, moves, Choice::Random);
        let board = boards.last().unwrap();
        let blanks = i % 3;
        let rack = random_rack(&mut rng, 7 - blanks, blanks);
        let plays = legal_plays(board, &rack, &lexicon);
        assert_well_formed(board, &rack, &plays, &lexicon);
        let mut reflected: Vec<RefPlay> = plays
            .iter()
            .map(|p| {
                let mut tiles: Vec<PlacedTile> = p
                    .tiles
                    .iter()
                    .map(|t| PlacedTile {
                        row: t.col,
                        col: t.row,
                        ..*t
                    })
                    .collect();
                tiles.sort();
                RefPlay {
                    tiles,
                    score: p.score,
                }
            })
            .collect();
        reflected.sort();
        let on_transposed = legal_plays(&board.transposed(), &rack, &lexicon);
        assert_eq!(on_transposed, reflected, "{}", describe(board, &rack));
    }
}

#[test]
fn empty_rack_and_empty_lexicon_give_no_plays() {
    let lexicon = small_lexicon();
    assert!(legal_plays(&RefBoard::empty(), "", &lexicon).is_empty());
    assert!(legal_plays_exhaustive(&RefBoard::empty(), "", &lexicon).is_empty());
    let none = RefLexicon::from_words(Vec::<String>::new());
    assert!(legal_plays(&RefBoard::empty(), "AEINRST", &none).is_empty());
}

/// A rack longer than 7 still gives plays of at most 7 tiles.
#[test]
fn plays_never_use_more_than_seven_tiles() {
    let lexicon = RefLexicon::from_words(["CATALOGS", "CATALOG", "CAT"]);
    let plays = legal_plays(&RefBoard::empty(), "CATALOGS", &lexicon);
    assert!(!plays.is_empty());
    assert!(plays.iter().all(|p| p.tiles.len() <= 7));
    assert!(plays.iter().any(|p| p.tiles.len() == 7));
}

#[test]
#[should_panic]
fn bad_rack_character_panics() {
    legal_plays(&RefBoard::empty(), "ab", &small_lexicon());
}
