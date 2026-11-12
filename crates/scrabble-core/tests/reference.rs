//! The engine against the independent reference (`crates/scrabble-reference`).
//!
//! The reference was written by a different author from docs/RULES.md alone, with a different
//! method (sliding every lexicon word along every row and column, then judging each candidate
//! rule by rule). These tests require the GADDAG generator to produce exactly the same set of
//! plays with exactly the same scores.
//!
//!     cargo test -p scrabble-core --test reference                    # quick: ~300 positions
//!     cargo test -p scrabble-core --test reference -- --ignored --nocapture   # full: 6,000+

mod common;

use rayon::prelude::*;
use scrabble_core::board::{Board, N};
use scrabble_core::gaddag::Gaddag;
use scrabble_core::game::Game;
use scrabble_core::movegen::generate_plays;
use scrabble_core::moves::{Move, Placed, Play};
use scrabble_core::rng::Rng;
use scrabble_core::tiles::{Rack, BLANK};
use scrabble_core::validate::check_placement;
use scrabble_reference::{legal_plays, parse_cases, PlacedTile, RefBoard, RefLexicon, RefPlay, SCORING_CASES};
use std::collections::BTreeSet;

type Canonical = (Vec<(u8, u8, u8, bool)>, i32);

fn canonical_engine(play: &Play) -> Canonical {
    let mut tiles: Vec<(u8, u8, u8, bool)> =
        play.placed().map(|t| ((t.index / N) as u8, (t.index % N) as u8, b'A' + t.letter, t.blank)).collect();
    tiles.sort_unstable();
    (tiles, play.score as i32)
}

fn canonical_reference(play: &RefPlay) -> Canonical {
    let mut tiles: Vec<(u8, u8, u8, bool)> = play.tiles.iter().map(|t| (t.row, t.col, t.letter, t.blank)).collect();
    tiles.sort_unstable();
    (tiles, play.score)
}

fn describe(play: &Canonical) -> String {
    let tiles: Vec<String> = play
        .0
        .iter()
        .map(|&(row, col, letter, blank)| {
            let letter = if blank { letter.to_ascii_lowercase() } else { letter } as char;
            format!("{}{}={letter}", (b'A' + col) as char, row + 1)
        })
        .collect();
    format!("{} ({})", tiles.join(" "), play.1)
}

/// Compare the two generators on one position. Returns the number of plays, or a report.
fn compare(gaddag: &Gaddag, lexicon: &RefLexicon, board: &Board, rack: &Rack) -> Result<usize, String> {
    let mut plays = Vec::new();
    generate_plays(gaddag, board, rack, &mut plays);
    let engine: BTreeSet<Canonical> = plays.iter().map(canonical_engine).collect();
    if engine.len() != plays.len() {
        return Err(format!("the engine generated {} duplicate plays", plays.len() - engine.len()));
    }
    let reference_board = RefBoard::from_text(&board.to_text()).expect("board text");
    let reference: BTreeSet<Canonical> =
        legal_plays(&reference_board, &rack.to_string(), lexicon).iter().map(canonical_reference).collect();
    if engine == reference {
        return Ok(engine.len());
    }
    let missed: Vec<String> = reference.difference(&engine).take(5).map(describe).collect();
    let extra: Vec<String> = engine.difference(&reference).take(5).map(describe).collect();
    Err(format!(
        "rack {rack}\n{}engine {} plays, reference {}\n  in the reference only: {missed:?}\n  in the engine only: {extra:?}",
        board.to_text(),
        engine.len(),
        reference.len()
    ))
}

/// A rack of the same size drawn from the game's bag with `blanks` blanks forced in.
fn rack_with_blanks(game: &Game, blanks: usize, rng: &mut Rng) -> Rack {
    let mut pool: Vec<u8> = game.bag.tiles().iter().copied().filter(|&t| t != BLANK).collect();
    pool.extend(game.racks[game.to_move].tiles().filter(|&t| t != BLANK));
    rng.shuffle(&mut pool);
    let size = game.racks[game.to_move].len();
    let mut rack = Rack::new();
    for _ in 0..blanks.min(size) {
        rack.add(BLANK);
    }
    for &tile in pool.iter().take(size - rack.len()) {
        rack.add(tile);
    }
    rack
}

#[derive(Default, Clone, Copy)]
struct Tally {
    positions: [usize; 3], // by number of blanks on the rack
    plays: usize,
}

/// Play `games` seeded games with a mix of random and top-scoring moves. At every turn compare
/// the generators on the real rack; every `blank_every` turns also on racks with one and two
/// blanks forced in.
fn sweep(gaddag: &Gaddag, lexicon: &RefLexicon, seeds: std::ops::Range<u64>, blank_every: u16) -> Tally {
    seeds
        .into_par_iter()
        .map(|seed| {
            let mut tally = Tally::default();
            let mut game = Game::new(seed);
            let mut rng = Rng::new(seed ^ 0x5EED);
            let mut plays = Vec::new();
            while !game.over {
                let rack = game.racks[game.to_move];
                let check = |rack: &Rack, tally: &mut Tally| {
                    let blanks = (rack.count(BLANK) as usize).min(2);
                    match compare(gaddag, lexicon, &game.board, rack) {
                        Ok(count) => {
                            tally.positions[blanks] += 1;
                            tally.plays += count;
                        }
                        Err(report) => panic!("seed {seed}, turn {}: {report}", game.turn),
                    }
                };
                check(&rack, &mut tally);
                if blank_every > 0 && game.turn % blank_every == 0 {
                    for blanks in 1..=2 {
                        let forced = rack_with_blanks(&game, blanks, &mut rng);
                        check(&forced, &mut tally);
                    }
                }
                plays.clear();
                game.plays(gaddag, &mut plays);
                let mv = if plays.is_empty() {
                    if game.can_exchange() { Move::Exchange(rack) } else { Move::Pass }
                } else if rng.below(3) == 0 {
                    Move::Play(plays[rng.below(plays.len() as u32) as usize])
                } else {
                    Move::Play(*plays.iter().max_by_key(|p| p.score).unwrap())
                };
                game.apply(&mv);
            }
            tally
        })
        .reduce(Tally::default, |a, b| Tally {
            positions: [a.positions[0] + b.positions[0], a.positions[1] + b.positions[1], a.positions[2] + b.positions[2]],
            plays: a.plays + b.plays,
        })
}

#[test]
fn hand_checked_scoring_cases() {
    let cases = parse_cases(SCORING_CASES).expect("case file parses");
    assert!(cases.len() >= 40, "{} cases", cases.len());
    let (mut legal, mut illegal, mut generated) = (0, 0, 0);
    for case in &cases {
        let gaddag = Gaddag::build(&case.words.iter().filter(|w| w.len() >= 2).collect::<Vec<_>>());
        let board = Board::from_text(&case.board.to_text()).expect("board text");
        let tiles: Vec<Placed> = case
            .tiles
            .iter()
            .map(|t: &PlacedTile| Placed { index: t.row as usize * N + t.col as usize, letter: t.letter.wrapping_sub(b'A'), blank: t.blank })
            .collect();
        let checked = check_placement(&gaddag, &board, &tiles);
        match (case.expect, &checked) {
            (Some(score), Ok(play)) => {
                assert_eq!(play.score as i32, score, "case {}: validator score", case.name);
                legal += 1;
                // The generator, given exactly these tiles as its rack, must find the same play
                // with the same score. (Cases on boards that cannot arise in a game, with tiles
                // down but the centre empty, are exercised through the validator only.)
                if tiles.len() <= 7 {
                    let mut rack = Rack::new();
                    for tile in &tiles {
                        rack.add(tile.tile());
                    }
                    let mut plays = Vec::new();
                    generate_plays(&gaddag, &board, &rack, &mut plays);
                    let found = plays.iter().find(|p| canonical_engine(p).0 == canonical_engine(play).0);
                    let found = found.unwrap_or_else(|| panic!("case {}: the generator does not find the play", case.name));
                    assert_eq!(found.score as i32, score, "case {}: generator score", case.name);
                    assert_eq!(found, play, "case {}: canonical form", case.name);
                    generated += 1;
                }
            }
            (None, Err(_)) => illegal += 1,
            (Some(score), Err(reason)) => panic!("case {}: expected {score}, validator says illegal: {reason}", case.name),
            (None, Ok(play)) => panic!("case {}: expected illegal, validator accepts it as {play}", case.name),
        }
    }
    println!("{legal} legal and {illegal} illegal hand-checked cases agree; {generated} also found by the generator");
}

#[test]
fn generator_matches_reference_quick() {
    let (Some(gaddag), true) = (common::real_gaddag(), common::lexicon_path().exists()) else { return };
    let lexicon = RefLexicon::from_file(&common::lexicon_path()).expect("word list");
    let tally = sweep(gaddag, &lexicon, 0..10, 6);
    println!("{:?} positions by blanks, {} plays compared", tally.positions, tally.plays);
    assert!(tally.positions.iter().sum::<usize>() >= 250);
}

/// The acceptance run from docs/PLAN.md: at least 5,000 positions, 500 with one blank and 100
/// with two. Takes a few minutes on all cores.
#[test]
#[ignore = "long: run with --ignored"]
fn generator_matches_reference_full() {
    let (Some(gaddag), true) = (common::real_gaddag(), common::lexicon_path().exists()) else { return };
    let lexicon = RefLexicon::from_file(&common::lexicon_path()).expect("word list");
    let start = std::time::Instant::now();
    let tally = sweep(gaddag, &lexicon, 1_000..1_220, 5);
    println!(
        "positions: {} no blank, {} one blank, {} two blanks; {} plays compared, 0 differences, {:.0} s",
        tally.positions[0],
        tally.positions[1],
        tally.positions[2],
        tally.plays,
        start.elapsed().as_secs_f64()
    );
    assert!(tally.positions.iter().sum::<usize>() >= 5_000);
    assert!(tally.positions[1] >= 500 && tally.positions[2] >= 100);
}
