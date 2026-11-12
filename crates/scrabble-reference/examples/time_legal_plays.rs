//! Time `legal_plays` with the full lexicon on positions from greedy self-play games.
//!
//! cargo run --release -p scrabble-reference --example time_legal_plays -- [games] [seed]
//!
//! Every position of every game (the board before each move) is timed with three fresh 7-tile
//! racks: no blank, one blank, two blanks. One thread.

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::Instant;

use common::*;
use scrabble_reference::{legal_plays, RefLexicon};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: usize = args.get(1).map(|a| a.parse().expect("games")).unwrap_or(3);
    let seed: u64 = args.get(2).map(|a| a.parse().expect("seed")).unwrap_or(1);

    let path = real_lexicon_path();
    let started = Instant::now();
    let lexicon =
        RefLexicon::from_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    println!(
        "{} words from {} in {:.2} s",
        lexicon.len(),
        path.display(),
        started.elapsed().as_secs_f64()
    );

    let mut rng = Rng::new(seed);
    let mut boards = Vec::new();
    let started = Instant::now();
    for _ in 0..games {
        let game = play_game(&lexicon, &mut rng, 40, Choice::Greedy);
        boards.extend(game);
    }
    println!(
        "{games} greedy games, {} positions, built in {:.1} s",
        boards.len(),
        started.elapsed().as_secs_f64()
    );

    for (label, blanks) in [("no blank", 0), ("one blank", 1), ("two blanks", 2)] {
        let mut times = Vec::new();
        let mut plays = 0usize;
        for board in &boards {
            let rack = random_rack(&mut rng, 7 - blanks, blanks);
            let started = Instant::now();
            plays += legal_plays(board, &rack, &lexicon).len();
            times.push(started.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = times.len();
        let mean = times.iter().sum::<f64>() / n as f64;
        println!(
            "{label:>10}: n={n}  mean {:.3} s  median {:.3} s  p90 {:.3} s  max {:.3} s  min {:.3} s  ({:.0} plays/position)",
            mean,
            times[n / 2],
            times[(n * 9) / 10],
            times[n - 1],
            times[0],
            plays as f64 / n as f64
        );
    }
}
