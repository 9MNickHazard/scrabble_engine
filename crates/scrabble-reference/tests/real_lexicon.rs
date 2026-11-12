//! Tests with the real word list. Skipped (with a printed note) when the file is missing.
//! Path: the SCRABBLE_LEXICON environment variable, or data/lexicon/NSWL2023.txt under the
//! workspace root.

mod common;

use std::sync::OnceLock;
use std::time::Instant;

use common::*;
use scrabble_reference::*;

fn real_lexicon() -> Option<&'static RefLexicon> {
    static LEXICON: OnceLock<Option<RefLexicon>> = OnceLock::new();
    LEXICON
        .get_or_init(|| {
            let path = real_lexicon_path();
            match RefLexicon::from_file(&path) {
                Ok(lexicon) => {
                    println!("loaded {} words from {}", lexicon.len(), path.display());
                    Some(lexicon)
                }
                Err(e) => {
                    println!(
                        "SKIPPED: no lexicon at {} ({e}); set SCRABBLE_LEXICON",
                        path.display()
                    );
                    None
                }
            }
        })
        .as_ref()
}

fn tiles(text: &str) -> Vec<PlacedTile> {
    let mut tiles: Vec<PlacedTile> = text
        .split_whitespace()
        .map(|t| parse_placed_tile(t).unwrap())
        .collect();
    tiles.sort();
    tiles
}

#[test]
fn real_lexicon_opening_known_answers() {
    let Some(lexicon) = real_lexicon() else {
        return;
    };
    let empty = RefBoard::empty();

    // AEINRST: every 7-letter anagram (RETAINS, NASTIER, ...) is a bingo of seven 1-point
    // letters. Across H8 a 7-letter word covers D8 or L8 (DL) when it starts on B8-D8 or F8-H8:
    // (7 + 1) x2 + 50 = 66. Starting on E8 it covers no letter premium: 7 x2 + 50 = 64. Down is
    // the same by symmetry. Nothing beats 66: a 7-tile play is worth at most that, and plays of
    // fewer tiles get no 50.
    assert!(lexicon.contains(b"RETAINS"));
    let plays = legal_plays(&empty, "AEINRST", lexicon);
    let best = plays.iter().map(|p| p.score).max().unwrap();
    assert_eq!(best, 66);
    assert!(plays
        .iter()
        .filter(|p| p.score == 66)
        .all(|p| p.tiles.len() == 7));
    assert!(plays.contains(&RefPlay {
        tiles: tiles("D8=R E8=E F8=T G8=A H8=I I8=N J8=S"),
        score: 66
    }));
    assert!(plays.contains(&RefPlay {
        tiles: tiles("E8=R F8=E G8=T H8=A I8=I J8=N K8=S"),
        score: 64
    }));
    assert!(plays.contains(&RefPlay {
        tiles: tiles("H2=R H3=E H4=T H5=A H6=I H7=N H8=S"),
        score: 66
    }));
    println!("AEINRST opening: {} plays, best {best}", plays.len());

    // CAT: CAT across H8-J8 = (3+1+1) x2 = 10; brute force agrees on the whole list.
    let plays = legal_plays(&empty, "CAT", lexicon);
    assert!(plays.contains(&RefPlay {
        tiles: tiles("H8=C I8=A J8=T"),
        score: 10
    }));
    assert_eq!(plays, legal_plays_exhaustive(&empty, "CAT", lexicon));
    println!("CAT opening: {} plays", plays.len());
}

/// `legal_plays` against the brute force with the full word list, on the opening and on positions
/// from a greedy game, with racks of 2 and 3 tiles (some with a blank).
#[test]
fn real_lexicon_matches_exhaustive_on_small_racks() {
    let Some(lexicon) = real_lexicon() else {
        return;
    };
    let mut rng = Rng::new(2024);
    let boards = play_game(lexicon, &mut rng, 8, Choice::Greedy);
    let mut compared = 0;
    let mut plays = 0;
    for board in &boards {
        for (letters, blanks) in [(2, 0), (3, 0), (1, 1), (2, 1)] {
            let rack = random_rack(&mut rng, letters, blanks);
            let fast = legal_plays(board, &rack, lexicon);
            let slow = legal_plays_exhaustive(board, &rack, lexicon);
            assert_eq!(fast, slow, "rack {rack}, board:\n{}", board.to_text());
            for play in &fast {
                assert_eq!(check_placement(board, &play.tiles, lexicon), Ok(play.score));
            }
            compared += 1;
            plays += fast.len();
        }
    }
    println!("{compared} positions compared with the full lexicon, {plays} plays, all equal");
}

/// Prints how long `legal_plays` takes with the full lexicon (test profile: opt-level 3, no LTO).
/// For release-mode numbers run the `time_legal_plays` example.
#[test]
fn real_lexicon_timing() {
    let Some(lexicon) = real_lexicon() else {
        return;
    };
    let mut rng = Rng::new(31337);
    let boards = play_game(lexicon, &mut rng, 12, Choice::Greedy);
    for (label, blanks) in [("no blank", 0), ("one blank", 1), ("two blanks", 2)] {
        let mut times = Vec::new();
        let mut plays = 0;
        for board in &boards {
            let rack = random_rack(&mut rng, 7 - blanks, blanks);
            let started = Instant::now();
            plays += legal_plays(board, &rack, lexicon).len();
            times.push(started.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "legal_plays, 7-tile rack, {label}: {} positions, mean {:.3} s, median {:.3} s, max {:.3} s, {:.0} plays/position",
            times.len(),
            mean,
            times[times.len() / 2],
            times[times.len() - 1],
            plays as f64 / times.len() as f64
        );
    }
}
