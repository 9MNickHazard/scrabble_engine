//! Timing of the move generator on positions from real games, for `scrabble-ai bench`.

use crate::game::Game;
use crate::movegen::generate_plays;
use crate::moves::{Move, Play};
use crate::rng::Rng;
use crate::tiles::{Rack, BLANK};
use crate::gaddag::Gaddag;
use std::time::Instant;

/// Timings of `generate_plays` for racks with a given number of blanks.
#[derive(Clone, Debug, Default)]
pub struct Timing {
    pub positions: usize,
    pub median_micros: f64,
    pub p99_micros: f64,
    pub max_micros: f64,
    pub mean_micros: f64,
    pub mean_plays: f64,
}

/// A rack of the same size as the mover's, drawn from the mover's rack and the bag, with exactly
/// `blanks` blanks.
fn rack_with_blanks(game: &Game, blanks: usize, rng: &mut Rng) -> Rack {
    let mut pool: Vec<u8> = game.racks[game.to_move].tiles().chain(game.bag.tiles().iter().copied()).filter(|&t| t != BLANK).collect();
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

/// Play one greedy game per seed on a single thread. At every turn, time move generation for a
/// rack with no blank, one blank and two blanks on that board. Returns the three timings.
pub fn movegen_timings(gaddag: &Gaddag, seeds: &[u64]) -> [Timing; 3] {
    let mut micros: [Vec<f64>; 3] = Default::default();
    let mut play_counts = [0usize; 3];
    let mut plays: Vec<Play> = Vec::new();
    for &seed in seeds {
        let mut game = Game::new(seed);
        let mut rng = Rng::new(seed ^ 0xBE7C);
        while !game.over {
            for blanks in 0..3 {
                let rack = rack_with_blanks(&game, blanks, &mut rng);
                plays.clear();
                let start = Instant::now();
                generate_plays(gaddag, &game.board, &rack, &mut plays);
                micros[blanks].push(start.elapsed().as_secs_f64() * 1e6);
                play_counts[blanks] += plays.len();
            }
            plays.clear();
            game.plays(gaddag, &mut plays);
            let mv = plays.iter().max_by_key(|p| p.score).map_or(Move::Pass, |p| Move::Play(*p));
            game.apply(&mv);
        }
    }
    std::array::from_fn(|blanks| {
        let times = &mut micros[blanks];
        times.sort_by(f64::total_cmp);
        let n = times.len();
        if n == 0 {
            return Timing::default();
        }
        Timing {
            positions: n,
            median_micros: times[n / 2],
            p99_micros: times[(n * 99 / 100).min(n - 1)],
            max_micros: times[n - 1],
            mean_micros: times.iter().sum::<f64>() / n as f64,
            mean_plays: play_counts[blanks] as f64 / n as f64,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timings_cover_all_three_rack_kinds() {
        let gaddag = Gaddag::build(&["AT", "TA", "CAT", "QI", "ZA", "AA", "TEA", "EAT", "ATE", "ON", "NO", "IN", "IT", "IS"]);
        let timings = movegen_timings(&gaddag, &[1, 2, 3]);
        assert!(timings.iter().all(|t| t.positions > 0 && t.positions == timings[0].positions));
        assert!(timings.iter().all(|t| t.median_micros <= t.p99_micros && t.p99_micros <= t.max_micros));
        // Blanks can only add plays.
        assert!(timings[2].mean_plays >= timings[0].mean_plays);
    }
}
