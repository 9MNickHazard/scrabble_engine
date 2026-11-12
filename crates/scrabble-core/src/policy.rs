//! Static move evaluation and the baseline bots.
//!
//! The static value of a move is its score plus the value of the leave it keeps. Three bots use
//! it: `Random` ignores it, `Greedy` uses the score alone, and `GreedyLeave` uses the whole
//! thing. They are the baseline ladder the network is measured against, the source of warm-start
//! data, and the ranking that prunes candidates before the network evaluates them.

use crate::gaddag::Gaddag;
use crate::game::Game;
use crate::leave::LeaveModel;
use crate::movegen::{generate_exchanges, generate_plays};
use crate::moves::{Move, Play};
use crate::rng::Rng;
use crate::tiles::Rack;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bot {
    /// A uniformly random tile placement; exchanges or passes only when there is none.
    Random,
    /// The highest-scoring placement.
    Greedy,
    /// The move with the highest score + leave value, exchanges included.
    GreedyLeave,
}

impl Bot {
    pub fn parse(name: &str) -> Option<Bot> {
        match name {
            "random" => Some(Bot::Random),
            "greedy" => Some(Bot::Greedy),
            "greedy-leave" | "greedy_leave" | "leave" => Some(Bot::GreedyLeave),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Bot::Random => "random",
            Bot::Greedy => "greedy",
            Bot::GreedyLeave => "greedy-leave",
        }
    }
}

/// A candidate move with everything known about it before the tiles are drawn.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub mv: Move,
    /// Points scored, including the going-out bonus when the move ends the game.
    pub score: i32,
    /// The tiles kept.
    pub leave: Rack,
    /// Leave value from the model (0 for a move that ends the game by going out).
    pub leave_value: f32,
    /// 0: the game goes on. 1: the mover goes out. 2: sixth scoreless turn.
    pub terminal: u8,
}

impl Candidate {
    /// Score plus leave value: the static evaluation.
    #[inline]
    pub fn static_value(&self) -> f32 {
        self.score as f32 + self.leave_value
    }
}

/// Reusable buffers for move generation, so a long self-play run does not allocate per turn.
#[derive(Default)]
pub struct Scratch {
    pub plays: Vec<Play>,
    pub exchanges: Vec<Rack>,
    pub candidates: Vec<Candidate>,
}

/// Tiles in the bag after the mover of `mv` has drawn.
#[inline]
pub fn bag_after_draw(game: &Game, mv: &Move) -> usize {
    match mv {
        Move::Play(play) => game.bag.len().saturating_sub(play.tiles_played as usize),
        _ => game.bag.len(),
    }
}

/// Describe `mv` for the player to move.
pub fn candidate(game: &Game, leaves: &LeaveModel, mv: Move) -> Candidate {
    let (me, opponent) = (game.to_move, 1 - game.to_move);
    let mut leave = game.racks[me];
    leave.remove_all(&mv.tiles_used());
    let mut score = mv.score();
    let terminal = match mv {
        Move::Play(_) if leave.is_empty() && game.bag.is_empty() => {
            score += 2 * game.racks[opponent].value();
            1
        }
        Move::Play(_) => 0,
        _ if game.scoreless_turns + 1 >= crate::game::SCORELESS_LIMIT => 2,
        _ => 0,
    };
    let leave_value = if terminal == 1 { 0.0 } else { leaves.value(&leave, bag_after_draw(game, &mv)) };
    Candidate { mv, score, leave, leave_value, terminal }
}

/// Fill `scratch.candidates` with the moves worth considering for the player to move:
/// every placement, every exchange if the bag allows one, and pass when it is the only move or
/// exchanging is no longer possible.
pub fn candidates(game: &Game, gaddag: &Gaddag, leaves: &LeaveModel, scratch: &mut Scratch) {
    scratch.plays.clear();
    scratch.exchanges.clear();
    scratch.candidates.clear();
    if game.over {
        return;
    }
    generate_plays(gaddag, &game.board, &game.racks[game.to_move], &mut scratch.plays);
    scratch.candidates.extend(scratch.plays.iter().map(|&play| candidate(game, leaves, Move::Play(play))));
    if game.can_exchange() {
        generate_exchanges(&game.racks[game.to_move], &mut scratch.exchanges);
        scratch.candidates.extend(scratch.exchanges.iter().map(|&tiles| candidate(game, leaves, Move::Exchange(tiles))));
    }
    if scratch.plays.is_empty() || !game.can_exchange() {
        scratch.candidates.push(candidate(game, leaves, Move::Pass));
    }
}

/// The move `bot` makes for the player to move.
pub fn choose(bot: Bot, game: &Game, gaddag: &Gaddag, leaves: &LeaveModel, rng: &mut Rng, scratch: &mut Scratch) -> Move {
    match bot {
        Bot::Random | Bot::Greedy => {
            scratch.plays.clear();
            generate_plays(gaddag, &game.board, &game.racks[game.to_move], &mut scratch.plays);
            if scratch.plays.is_empty() {
                // Nothing to place: throw the whole rack back if allowed, otherwise pass.
                return if game.can_exchange() { Move::Exchange(game.racks[game.to_move]) } else { Move::Pass };
            }
            let play = if bot == Bot::Random {
                scratch.plays[rng.below(scratch.plays.len() as u32) as usize]
            } else {
                best_by(&scratch.plays, |play| play.score as f32)
            };
            Move::Play(play)
        }
        Bot::GreedyLeave => {
            candidates(game, gaddag, leaves, scratch);
            best_by(&scratch.candidates, Candidate::static_value).mv
        }
    }
}

/// The first item with the highest key. Ties go to the earliest, so the choice does not depend
/// on anything but the generation order.
fn best_by<T: Copy>(items: &[T], key: impl Fn(&T) -> f32) -> T {
    let mut best = items[0];
    let mut best_key = key(&best);
    for item in &items[1..] {
        let k = key(item);
        if k > best_key {
            best = *item;
            best_key = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::leave::{features, FEATURES, MAX_ACTIVE};

    fn gaddag() -> Gaddag {
        Gaddag::build(&["AT", "TA", "CAT", "QI", "ZA", "AA", "TAT"])
    }

    fn game(rack: &str, bag: usize) -> Game {
        let racks = [Rack::parse(rack).unwrap(), Rack::parse("EEEEEEE").unwrap()];
        Game::from_position(Board::new(), racks, &vec![4u8; bag], [0, 0], 0, 1)
    }

    #[test]
    fn greedy_takes_the_top_score_and_random_takes_any_play() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let game = game("CATQI", 20);
        let (mut rng, mut scratch) = (Rng::new(1), Scratch::default());
        let mv = choose(Bot::Greedy, &game, &gaddag, &leaves, &mut rng, &mut scratch);
        // QI with the Q doubled twice... Q on H8 (double word): (10+1) x 2 = 22.
        assert_eq!(mv.score(), 22, "{mv}");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(choose(Bot::Random, &game, &gaddag, &leaves, &mut rng, &mut scratch));
        }
        assert!(seen.len() > 5 && seen.iter().all(|m| matches!(m, Move::Play(_))));
    }

    #[test]
    fn with_no_play_the_simple_bots_exchange_if_they_can_else_pass() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (mut rng, mut scratch) = (Rng::new(1), Scratch::default());
        for bot in [Bot::Random, Bot::Greedy] {
            let stuck = game("VVWWXYZ", 20);
            assert_eq!(choose(bot, &stuck, &gaddag, &leaves, &mut rng, &mut scratch), Move::Exchange(stuck.racks[0]));
            let stuck_late = game("VVWWXYZ", 6);
            assert_eq!(choose(bot, &stuck_late, &gaddag, &leaves, &mut rng, &mut scratch), Move::Pass);
        }
    }

    #[test]
    fn leave_values_change_the_choice() {
        let gaddag = gaddag();
        let game = game("CATQI", 20);
        let (mut rng, mut scratch) = (Rng::new(1), Scratch::default());
        // A model that hates keeping a Q on the rack by 100 points agrees with greedy here (QI).
        // One that loves keeping the Q prefers CAT (10) and its Q-I leave.
        let mut loves_q = LeaveModel::zero();
        let mut active = [0u16; MAX_ACTIVE];
        let n = features(&Rack::parse("Q").unwrap(), 20, &mut active);
        assert!(n >= 1 && (active[0] as usize) < FEATURES);
        loves_q.weights[0][active[0] as usize] = 100.0;
        let mv = choose(Bot::GreedyLeave, &game, &gaddag, &loves_q, &mut rng, &mut scratch);
        assert_eq!(mv.tiles_used().count(16), 0, "should keep the Q: {mv}");
        assert_eq!(mv.score(), 10, "CAT: {mv}");
        let zero = choose(Bot::GreedyLeave, &game, &gaddag, &LeaveModel::zero(), &mut rng, &mut scratch);
        assert_eq!(zero.score(), 22);
    }

    #[test]
    fn candidates_cover_plays_exchanges_and_pass_by_the_rules() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let mut scratch = Scratch::default();
        let count = |scratch: &Scratch, pred: fn(&Move) -> bool| scratch.candidates.iter().filter(|c| pred(&c.mv)).count();

        candidates(&game("AT", 20), &gaddag, &leaves, &mut scratch);
        assert_eq!(count(&scratch, |m| matches!(m, Move::Exchange(_))), 3); // A, T, AT
        assert_eq!(count(&scratch, |m| matches!(m, Move::Pass)), 0);
        assert!(count(&scratch, |m| matches!(m, Move::Play(_))) > 0);

        candidates(&game("AT", 3), &gaddag, &leaves, &mut scratch);
        assert_eq!(count(&scratch, |m| matches!(m, Move::Exchange(_))), 0);
        assert_eq!(count(&scratch, |m| matches!(m, Move::Pass)), 1);

        // Going out: bag empty and the play uses the whole rack. Bonus = 2 x seven E's = 14.
        candidates(&game("AT", 0), &gaddag, &leaves, &mut scratch);
        let out = scratch.candidates.iter().find(|c| c.terminal == 1).expect("a going-out play");
        assert_eq!((out.score, out.leave.len(), out.leave_value), (4 + 14, 0, 0.0));

        // Sixth scoreless turn.
        let mut passed = game("VVWWXYZ", 3);
        passed.scoreless_turns = 5;
        candidates(&passed, &gaddag, &leaves, &mut scratch);
        assert_eq!(scratch.candidates.len(), 1);
        assert_eq!(scratch.candidates[0].terminal, 2);
    }
}
