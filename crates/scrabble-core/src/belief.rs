//! What a placement says about the tiles its player kept.
//!
//! A player who puts down QUIZ for 48 was probably not holding a blank and an S as well: with
//! those they would more likely have found something better. This module turns that argument
//! into numbers. When a player places tiles, the other player can ask, for each set of tiles the
//! mover might have kept, how good the move actually played would have been with that rack, and
//! weigh the possibilities accordingly (Richards and Amir, "Opponent Modeling in Scrabble",
//! IJCAI 2007).
//!
//! The inference here:
//!
//! 1. Sample possible leaves uniformly from the tiles the observer could not see, less the tiles
//!    just played.
//! 2. For each, put the played tiles back with it to make the rack the mover would have had, and
//!    find the best static value (score + leave value) that rack had on the board as it was.
//! 3. Weigh the sample by `exp(-(best - value of the move played) / SHARPNESS)`: a mover who
//!    mostly, not always, plays close to the best static move.
//! 4. The weighted mean of the sampled leaves is the expected leave. The tiles drawn to replace
//!    the ones played come uniformly from what is then left, so their expectation follows.
//!
//! The result is the expected number of each tile type on the mover's new rack. It costs one
//! full move generation per sample, which is why the number of samples is a setting and 0 turns
//! the inference off. Exchanges and passes are not interpreted: nothing is inferred from them.

use crate::game::{Applied, Belief, Game};
use crate::gaddag::Gaddag;
use crate::leave::LeaveModel;
use crate::moves::{Move, Play};
use crate::policy::{candidate, candidates, Candidate, Scratch};
use crate::rng::Rng;
use crate::tiles::{TILE_TYPES, TOTAL_TILES};

/// Expected tile counts are stored as integers: count times this.
pub const SCALE: i16 = 256;
/// Points of static value that make a move e times less likely to be chosen. The static bot
/// always takes the best (0 here would say so), but self-play explores and a network disagrees
/// with the static ranking, so the inference is kept soft.
pub const SHARPNESS: f32 = 6.0;

/// Expected tile counts (times [`SCALE`]) on the mover's rack once `play` has been made and the
/// replacement tiles drawn, as the other player can infer them. `None` when there is nothing to
/// infer: no tiles were kept, or the bag is empty (the unseen pool is then exactly the rack).
///
/// Call this *before* applying the move: it needs the board as the mover saw it.
pub fn infer(
    game: &Game,
    play: &Play,
    gaddag: &Gaddag,
    leaves: &LeaveModel,
    samples: usize,
    rng: &mut Rng,
    scratch: &mut Scratch,
) -> Option<[i16; TILE_TYPES]> {
    let (mover, observer) = (game.to_move, 1 - game.to_move);
    let used = play.tiles_used();
    let kept = game.racks[mover].len() - used.len();
    let bag = game.bag.len();
    if samples == 0 || kept == 0 || bag == 0 {
        return None;
    }
    // What the observer cannot see once the played tiles are on the board: the bag and the
    // leave. (In a legal game this is `game.unseen(observer)` less the tiles played; adding it
    // up from the bag and the rack keeps it right in positions set up by hand as well.)
    let _ = observer;
    let mut pool = game.bag.counts();
    let mut tiles = [0u8; TOTAL_TILES];
    let mut total = 0;
    for (tile, ((count, held), played)) in pool.iter_mut().zip(game.racks[mover].counts).zip(used.counts).enumerate() {
        *count += held - played;
        for _ in 0..*count {
            tiles[total] = tile as u8;
            total += 1;
        }
    }
    debug_assert_eq!(total, bag + kept);

    let mut probe = game.clone();
    let mut expected = [0f64; TILE_TYPES];
    let mut weight_sum = 0f64;
    for _ in 0..samples {
        // The first `kept` entries after a partial shuffle are a uniform sample without replacement.
        let mut rack = used;
        for i in 0..kept {
            tiles.swap(i, i + rng.below((total - i) as u32) as usize);
            rack.add(tiles[i]);
        }
        probe.racks[mover] = rack;
        candidates(&probe, gaddag, leaves, scratch);
        let best = scratch.candidates.iter().map(Candidate::static_value).fold(f32::NEG_INFINITY, f32::max);
        let played = candidate(&probe, leaves, Move::Play(*play)).static_value();
        let weight = (((played - best).min(0.0) / SHARPNESS) as f64).exp();
        weight_sum += weight;
        for &tile in &tiles[..kept] {
            expected[tile as usize] += weight;
        }
    }
    if !(weight_sum > 1e-12) {
        return None;
    }
    let draws = used.len().min(bag) as f64;
    let mut counts = [0i16; TILE_TYPES];
    for tile in 0..TILE_TYPES {
        let leave = expected[tile] / weight_sum;
        let in_bag = pool[tile] as f64 - leave;
        counts[tile] = ((leave + draws * in_bag / bag as f64) * SCALE as f64).round() as i16;
    }
    Some(counts)
}

/// Apply `mv` and record what the other player can infer from it about the mover's new rack.
/// With `samples` 0 this is exactly [`Game::apply`].
pub fn apply_observed(
    game: &mut Game,
    mv: &Move,
    gaddag: &Gaddag,
    leaves: &LeaveModel,
    samples: usize,
    rng: &mut Rng,
    scratch: &mut Scratch,
) -> Applied {
    let observer = 1 - game.to_move;
    let inferred = match mv {
        Move::Play(play) if samples > 0 => infer(game, play, gaddag, leaves, samples, rng, scratch),
        _ => None,
    };
    let applied = game.apply(mv);
    if let Some(counts) = inferred {
        game.belief[observer] = Belief { inferred: true, counts };
    }
    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::tiles::{Rack, BLANK};

    fn gaddag() -> Gaddag {
        Gaddag::build(&["AT", "TA", "CAT", "CATS", "CAST", "ACT", "ACTS", "AS", "TAS", "SAT", "QI", "ZA", "ZAS", "TE", "EE", "TEE", "TEES", "EAT", "EATS", "ATE", "SEAT", "SEA", "ES"])
    }

    /// A game on an empty board where the mover holds `rack` and the pool beyond the two racks
    /// is `bag`.
    fn game(rack: &str, bag: &str) -> Game {
        let racks = [Rack::parse(rack).unwrap(), Rack::parse("EEEEEEE").unwrap()];
        let bag: Vec<u8> = Rack::parse(bag).unwrap().tiles().collect();
        Game::from_position(Board::new(), racks, &bag, [0, 0], 0, 1)
    }

    fn play_of(game: &Game, gaddag: &Gaddag, text: &str) -> Play {
        let mut plays = Vec::new();
        game.plays(gaddag, &mut plays);
        *plays.iter().find(|p| p.to_string().starts_with(text)).unwrap_or_else(|| panic!("no play {text}"))
    }

    #[test]
    fn nothing_is_inferred_without_samples_a_leave_or_a_bag() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (mut rng, mut scratch) = (Rng::new(1), Scratch::default());
        let with_bag = game("CATE", "AAASSSTTT");
        let cat = play_of(&with_bag, &gaddag, "8F CAT ");
        assert!(infer(&with_bag, &cat, &gaddag, &leaves, 0, &mut rng, &mut scratch).is_none(), "no samples");
        let empty_bag = game("CATE", "");
        let cat = play_of(&empty_bag, &gaddag, "8F CAT ");
        assert!(infer(&empty_bag, &cat, &gaddag, &leaves, 8, &mut rng, &mut scratch).is_none(), "empty bag");
        let all_played = game("CAT", "AAASSSTTT");
        let cat = play_of(&all_played, &gaddag, "8F CAT ");
        assert!(infer(&all_played, &cat, &gaddag, &leaves, 8, &mut rng, &mut scratch).is_none(), "nothing kept");
    }

    #[test]
    fn the_expected_counts_add_up_to_the_new_rack_and_stay_inside_the_pool() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (mut rng, mut scratch) = (Rng::new(7), Scratch::default());
        // The mover holds CAT plus four tiles the observer cannot tell from the nine in the bag.
        let game = game("CATAEST", "AASSTTEEZ");
        let cat = play_of(&game, &gaddag, "8F CAT ");
        let counts = infer(&game, &cat, &gaddag, &leaves, 64, &mut rng, &mut scratch).expect("an inference");
        // Four tiles kept and three drawn: seven on the new rack.
        let total: i32 = counts.iter().map(|&c| c as i32).sum();
        assert!((total - 7 * SCALE as i32).abs() <= TILE_TYPES as i32, "{total}");
        // What the observer cannot see once CAT is down: the leave AEST and the bag AASSTTEEZ.
        let pool = Rack::parse("AESTAASSTTEEZ").unwrap().counts;
        for (tile, (&expected, &available)) in counts.iter().zip(pool.iter()).enumerate() {
            assert!(expected >= 0 && expected as i32 <= available as i32 * SCALE as i32, "tile {tile}: {expected} of {available}");
        }
        assert_eq!(counts[BLANK as usize], 0, "there is no blank to hold");
        assert_eq!(counts[2], 0, "the only C is on the board");
    }

    #[test]
    fn a_weaker_move_than_the_tiles_allowed_makes_those_tiles_unlikely() {
        // The mover plays AT for 4 points and keeps one tile, which is a Z or an E as far as the
        // observer knows (three of each unseen besides the play). With a Z they had ZA for 22,
        // so the Z is much the less likely tile to have been kept: its weight is exp(-18 / 6)
        // against 1 for the E, a posterior of about 0.047.
        let gaddag = Gaddag::build(&["AT", "TA", "ZA", "EE"]);
        let leaves = LeaveModel::zero();
        let (mut rng, mut scratch) = (Rng::new(3), Scratch::default());
        let game = game("ATE", "ZZZEE");
        let at = play_of(&game, &gaddag, "8G AT ");
        assert_eq!(at.score, 4);
        let counts = infer(&game, &at, &gaddag, &leaves, 2000, &mut rng, &mut scratch).expect("an inference");
        let (z, e) = (counts[25] as f64 / SCALE as f64, counts[4] as f64 / SCALE as f64);
        assert!((z + e - 3.0).abs() < 0.02, "one kept and two drawn: {z} + {e}");
        // Kept: 0.047 of a Z. Drawn: two of the five tiles left, which hold the other 2.953 Z's.
        let kept = (-3.0f64).exp() / (1.0 + (-3.0f64).exp());
        let expected = kept + 2.0 * (3.0 - kept) / 5.0;
        assert!((z - expected).abs() < 0.03, "Z {z}, expected {expected}");
        assert!(z < 1.5 - 0.2, "well below the 1.5 of a rack drawn blind");
    }

    #[test]
    fn apply_observed_records_the_inference_for_the_other_player_only() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (mut rng, mut scratch) = (Rng::new(5), Scratch::default());
        let start = game("CATAEST", "AASSTTEEZ");
        let cat = Move::Play(play_of(&start, &gaddag, "8F CAT "));

        let mut plain = start.clone();
        let applied_plain = apply_observed(&mut plain, &cat, &gaddag, &leaves, 0, &mut rng, &mut scratch);
        assert_eq!(plain.belief, [Belief::default(); 2]);

        let mut observed = start.clone();
        let applied = apply_observed(&mut observed, &cat, &gaddag, &leaves, 16, &mut rng, &mut scratch);
        assert_eq!(applied, applied_plain);
        assert_eq!((observed.scores, observed.racks, observed.to_move), (plain.scores, plain.racks, plain.to_move), "the game itself is unchanged");
        assert!(observed.belief[1].inferred, "the player who did not move has an inference");
        assert_eq!(observed.belief[0], Belief::default());

        // The observer's own move leaves their inference alone; the first player's next move replaces it.
        observed.apply(&Move::Pass);
        assert!(observed.belief[1].inferred);
        observed.apply(&Move::Pass);
        assert_eq!(observed.belief[1], Belief::default(), "a pass says nothing, and the old inference is out of date");
    }

    #[test]
    fn the_same_seed_gives_the_same_inference() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let mut scratch = Scratch::default();
        let game = game("CATAEST", "AASSTTEEZ");
        let cat = play_of(&game, &gaddag, "8F CAT ");
        let a = infer(&game, &cat, &gaddag, &leaves, 12, &mut Rng::new(9), &mut scratch);
        let b = infer(&game, &cat, &gaddag, &leaves, 12, &mut Rng::new(9), &mut scratch);
        let c = infer(&game, &cat, &gaddag, &leaves, 12, &mut Rng::new(10), &mut scratch);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
