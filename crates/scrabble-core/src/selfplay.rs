//! Bot-versus-bot games, run in parallel: paired matches for the baseline ladder, and the
//! self-play that leave values are fitted from.

use crate::gaddag::Gaddag;
use crate::game::Game;
use crate::leave::{features, stage, Accumulator, LeaveModel, MAX_ACTIVE, STAGES};
use crate::policy::{bag_after_draw, candidate, choose, Bot, Scratch};
use crate::rng::Rng;
use rayon::prelude::*;

/// A bot together with the leave values it plays with.
#[derive(Clone, Copy)]
pub struct Player<'a> {
    pub bot: Bot,
    pub leaves: &'a LeaveModel,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GameResult {
    pub seed: u64,
    /// Final scores by seat (seat 0 moved first).
    pub scores: [i32; 2],
    pub turns: u16,
}

/// Play one game. `players[seat]` controls that seat.
pub fn play_game(gaddag: &Gaddag, players: [Player; 2], seed: u64, scratch: &mut Scratch) -> GameResult {
    let mut game = Game::new(seed);
    // The bots' own randomness is separate from the bag's, but still fixed by the seed.
    let mut rng = Rng::new(seed ^ 0xB07_5EED);
    while !game.over {
        let player = players[game.to_move];
        let mv = choose(player.bot, &game, gaddag, player.leaves, &mut rng, scratch);
        game.apply(&mv);
    }
    GameResult { seed, scores: game.scores, turns: game.turn }
}

/// One seed played twice, with the two players swapping seats.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PairResult {
    pub seed: u64,
    /// Player A's spread when moving first, and when moving second.
    pub spread_first: i32,
    pub spread_second: i32,
    pub turns: [u16; 2],
    /// Player A's own score in each of the two games.
    pub score_a: [i32; 2],
    pub score_b: [i32; 2],
}

/// Play every seed twice (A first, then B first) across all cores. Results are in seed order and
/// do not depend on the number of threads.
pub fn play_pairs(gaddag: &Gaddag, a: Player, b: Player, seeds: &[u64]) -> Vec<PairResult> {
    seeds
        .par_iter()
        .map_init(Scratch::default, |scratch, &seed| {
            let first = play_game(gaddag, [a, b], seed, scratch);
            let second = play_game(gaddag, [b, a], seed, scratch);
            PairResult {
                seed,
                spread_first: first.scores[0] - first.scores[1],
                spread_second: second.scores[1] - second.scores[0],
                turns: [first.turns, second.turns],
                score_a: [first.scores[0], second.scores[1]],
                score_b: [first.scores[1], second.scores[0]],
            }
        })
        .collect()
}

/// What a leave fit saw, per stage of the game.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageReport {
    pub samples: u64,
    /// Mean and standard deviation of the target (remaining spread for the mover).
    pub target_mean: f64,
    pub target_sd: f64,
}

/// Fit leave values from self-play.
///
/// Both seats are played by `bot` using `current` leave values. For every turn the sample is
/// (features of the leave kept, spread the mover gained from then to the end of the game).
/// Turns that end the game by going out are left out: nothing is kept and the value is 0.
pub fn fit_leaves(
    gaddag: &Gaddag,
    bot: Bot,
    current: &LeaveModel,
    seeds: &[u64],
    ridge: f64,
) -> (LeaveModel, [StageReport; STAGES]) {
    struct Turn {
        stage: usize,
        active: [u16; MAX_ACTIVE],
        count: usize,
        mover: usize,
        spread_after: i32,
    }

    let sums = seeds
        .par_iter()
        .fold(
            || (std::array::from_fn::<Accumulator, STAGES, _>(|_| Accumulator::default()), Scratch::default(), Vec::<Turn>::new()),
            |(mut sums, mut scratch, mut turns), &seed| {
                let mut game = Game::new(seed);
                let mut rng = Rng::new(seed ^ 0xB07_5EED);
                turns.clear();
                while !game.over {
                    let mover = game.to_move;
                    let mv = choose(bot, &game, gaddag, current, &mut rng, &mut scratch);
                    let described = candidate(&game, current, mv);
                    if described.terminal != 1 {
                        let bag = bag_after_draw(&game, &mv);
                        let mut active = [0u16; MAX_ACTIVE];
                        let count = features(&described.leave, bag, &mut active);
                        turns.push(Turn {
                            stage: stage(bag),
                            active,
                            count,
                            mover,
                            spread_after: game.spread(mover) + described.score,
                        });
                    }
                    game.apply(&mv);
                }
                for turn in &turns {
                    let remaining = game.spread(turn.mover) - turn.spread_after;
                    sums[turn.stage].add(&turn.active[..turn.count], remaining as f64);
                }
                (sums, scratch, turns)
            },
        )
        .map(|(sums, _, _)| sums)
        .reduce(
            || std::array::from_fn(|_| Accumulator::default()),
            |mut left, right| {
                for (l, r) in left.iter_mut().zip(&right) {
                    l.merge(r);
                }
                left
            },
        );

    let model = LeaveModel { weights: std::array::from_fn(|s| sums[s].solve(ridge)) };
    let report = std::array::from_fn(|s| {
        let n = sums[s].samples.max(1) as f64;
        let mean = sums[s].target_sum / n;
        StageReport {
            samples: sums[s].samples,
            target_mean: mean,
            target_sd: (sums[s].target_squares / n - mean * mean).max(0.0).sqrt(),
        }
    });
    (model, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaddag() -> Gaddag {
        // Enough short words that random racks usually have a play.
        Gaddag::build(&[
            "AA", "AB", "AD", "AE", "AG", "AH", "AI", "AL", "AM", "AN", "AR", "AS", "AT", "AW", "AX", "AY", "BA", "BE",
            "BI", "BO", "BY", "DA", "DE", "DO", "ED", "EF", "EH", "EL", "EM", "EN", "ER", "ES", "EX", "FA", "FE", "GO",
            "HA", "HE", "HI", "HM", "HO", "ID", "IF", "IN", "IS", "IT", "JO", "KA", "KI", "LA", "LI", "LO", "MA", "ME",
            "MI", "MM", "MO", "MU", "MY", "NA", "NE", "NO", "NU", "OD", "OE", "OF", "OH", "OI", "OM", "ON", "OP", "OR",
            "OS", "OW", "OX", "OY", "PA", "PE", "PI", "QI", "RE", "SH", "SI", "SO", "TA", "TI", "TO", "UH", "UM", "UN",
            "UP", "US", "UT", "WE", "WO", "XI", "XU", "YA", "YE", "YO", "ZA", "CAT", "DOG", "TEA", "EAT", "ATE", "RAT",
            "TAR", "ART", "NET", "TEN", "ONE", "EON", "SEA", "SIT", "ITS", "TIE", "TOE", "RETAINS", "STAINER",
        ])
    }

    #[test]
    fn games_are_reproducible_and_pairs_swap_seats() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let greedy = Player { bot: Bot::Greedy, leaves: &leaves };
        let random = Player { bot: Bot::Random, leaves: &leaves };
        let mut scratch = Scratch::default();
        let one = play_game(&gaddag, [greedy, random], 7, &mut scratch);
        assert_eq!(one, play_game(&gaddag, [greedy, random], 7, &mut scratch));
        assert_ne!(one, play_game(&gaddag, [greedy, random], 8, &mut scratch));

        let seeds: Vec<u64> = (0..40).collect();
        let pairs = play_pairs(&gaddag, greedy, random, &seeds);
        assert_eq!(pairs.len(), 40);
        assert_eq!(pairs[7].seed, 7);
        assert_eq!(pairs[7].spread_first, one.scores[0] - one.scores[1]);
        assert_eq!(pairs, play_pairs(&gaddag, greedy, random, &seeds), "independent of thread scheduling");
        // A bot paired against itself: the two games of a pair are the same game seen from each side.
        for pair in play_pairs(&gaddag, greedy, greedy, &seeds) {
            assert_eq!(pair.spread_first, -pair.spread_second);
        }
        let mean = |f: fn(&PairResult) -> i32| pairs.iter().map(f).sum::<i32>() as f64 / pairs.len() as f64;
        assert!(mean(|p| p.spread_first + p.spread_second) > 0.0, "greedy should outscore random");
    }

    #[test]
    fn leave_fit_runs_and_counts_one_sample_per_non_final_turn() {
        let gaddag = gaddag();
        let seeds: Vec<u64> = (0..60).collect();
        let (model, report) = fit_leaves(&gaddag, Bot::Greedy, &LeaveModel::zero(), &seeds, 10.0);
        let samples: u64 = report.iter().map(|r| r.samples).sum();
        let leaves = LeaveModel::zero();
        let greedy = Player { bot: Bot::Greedy, leaves: &leaves };
        let mut scratch = Scratch::default();
        let turns: u64 = seeds.iter().map(|&s| play_game(&gaddag, [greedy, greedy], s, &mut scratch).turns as u64).sum();
        assert!(samples <= turns && samples + seeds.len() as u64 >= turns, "{samples} samples, {turns} turns");
        assert!(model.weights.iter().flatten().all(|w| w.is_finite()));
        assert!(report[0].samples > 0 && report[0].target_sd > 0.0);
    }
}
