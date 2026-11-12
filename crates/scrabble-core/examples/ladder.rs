//! Fit leave values and play the baseline ladder, all in Rust.
//!
//!     cargo run --release -p scrabble-core --example ladder -- [fit_games] [pairs] [rounds]
//!
//! Prints the speed of each step and, for each rung, player A's mean spread per game over
//! seat-swapped pairs with a 95% confidence interval. `uv run scrabble-ai eval` is the supported
//! way to run evaluations; this example exists to exercise the engine without Python.

use scrabble_core::gaddag::Gaddag;
use scrabble_core::leave::LeaveModel;
use scrabble_core::lexicon::Lexicon;
use scrabble_core::policy::Bot;
use scrabble_core::selfplay::{fit_leaves, play_pairs, Player};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<u64> = std::env::args().skip(1).map(|a| a.parse().expect("numeric argument")).collect();
    let fit_games = args.first().copied().unwrap_or(200_000);
    let pairs = args.get(1).copied().unwrap_or(2_000);
    let rounds = args.get(2).copied().unwrap_or(2);

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let lexicon_dir = root.join("data").join("lexicon");
    let lexicon = Lexicon::from_file("NSWL2023", &lexicon_dir.join("NSWL2023.txt")).expect("word list");
    let gaddag = Gaddag::load_or_build(&lexicon, &lexicon_dir).expect("GADDAG");
    println!("lexicon {} ({} words), {} threads", lexicon.id(), lexicon.words.len(), rayon::current_num_threads());

    let zero = LeaveModel::zero();
    let mut leaves = LeaveModel::zero();
    for round in 0..rounds {
        let bot = if round == 0 { Bot::Greedy } else { Bot::GreedyLeave };
        let seeds: Vec<u64> = (0..fit_games).map(|i| 1_000_000_000 * (round + 1) + i).collect();
        let start = Instant::now();
        let (fitted, report) = fit_leaves(&gaddag, bot, &leaves, &seeds, 50.0);
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "fit round {round}: {fit_games} {} self-play games in {seconds:.1} s ({:.0} games/s)",
            bot.name(),
            fit_games as f64 / seconds
        );
        for (stage, r) in report.iter().enumerate() {
            println!("  stage {stage}: {} samples, target mean {:+.1}, sd {:.1}", r.samples, r.target_mean, r.target_sd);
        }
        leaves = fitted;
    }

    let rack = |text: &str| scrabble_core::tiles::Rack::parse(text).unwrap();
    println!("sample leave values with a full bag (points):");
    for text in ["", "S", "?", "Q", "QU", "ERS", "AEINRT", "UUV", "VVW", "ZX", "AEIOU", "RST", "??"] {
        println!("  {:>7}  {:+6.1}", if text.is_empty() { "(none)" } else { text }, leaves.value(&rack(text), 60));
    }

    let seeds: Vec<u64> = (0..pairs).map(|i| 9_000_000_000 + i).collect();
    let players = [
        ("random", Player { bot: Bot::Random, leaves: &zero }),
        ("greedy", Player { bot: Bot::Greedy, leaves: &zero }),
        ("greedy-leave", Player { bot: Bot::GreedyLeave, leaves: &leaves }),
    ];
    for (a, b) in [(1, 0), (2, 1), (2, 0)] {
        let start = Instant::now();
        let results = play_pairs(&gaddag, players[a].1, players[b].1, &seeds);
        let seconds = start.elapsed().as_secs_f64();
        let n = results.len() as f64;
        // One value per pair: A's mean spread over its two games, so seat and tile luck cancel.
        let per_pair: Vec<f64> = results.iter().map(|r| (r.spread_first + r.spread_second) as f64 / 2.0).collect();
        let mean = per_pair.iter().sum::<f64>() / n;
        let sd = (per_pair.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let half_width = 1.96 * sd / n.sqrt();
        let wins: f64 = results
            .iter()
            .flat_map(|r| [r.spread_first, r.spread_second])
            .map(|s| if s > 0 { 1.0 } else if s == 0 { 0.5 } else { 0.0 })
            .sum::<f64>()
            / (2.0 * n);
        let score_a = results.iter().map(|r| (r.score_a[0] + r.score_a[1]) as f64).sum::<f64>() / (2.0 * n);
        let score_b = results.iter().map(|r| (r.score_b[0] + r.score_b[1]) as f64).sum::<f64>() / (2.0 * n);
        let turns = results.iter().map(|r| (r.turns[0] + r.turns[1]) as f64).sum::<f64>() / (2.0 * n);
        println!(
            "{:>12} vs {:<12} {pairs} pairs: spread {mean:+.1} +/- {half_width:.1}, wins {:.1}%, \
             scores {score_a:.0}-{score_b:.0}, {turns:.1} turns/game, {:.0} games/s",
            players[a].0,
            players[b].0,
            wins * 100.0,
            2.0 * n / seconds
        );
    }
}
