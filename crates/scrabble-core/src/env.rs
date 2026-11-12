//! Many games stepped together, for self-play and evaluation with a network in the loop.
//!
//! Each seat of each game is controlled either by a built-in bot or externally. One cycle is:
//!
//! 1. [`BatchEnv::prepare`] plays every bot turn that is due and then, for each game waiting on
//!    an external decision, picks the candidate moves and writes their afterstates into the
//!    caller's buffers. Games run in parallel across cores.
//! 2. The caller scores the candidates (one network forward pass over all games) and picks one
//!    per game.
//! 3. [`BatchEnv::apply`] plays the picks.
//!
//! Candidates are the top placements by static value (score + leave value), the best few
//! exchanges, and pass when it is forced or exchanging is no longer possible. They are written
//! in descending static value, so index 0 is always the greedy+leave move.

use crate::encode::{board_codes, encode_afterstate, META};
use crate::gaddag::Gaddag;
use crate::game::Game;
use crate::leave::LeaveModel;
use crate::moves::Move;
use crate::policy::{candidates, choose, Bot, Candidate, Scratch};
use crate::rng::Rng;
use crate::tiles::TILE_TYPES;
use crate::board::SQUARES;
use rayon::prelude::*;

/// Who decides a seat's moves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Controller {
    /// The caller, through `prepare` / `apply`.
    External,
    Bot(Bot),
}

/// What a slot is doing after `prepare`.
pub const STATUS_EMPTY: u8 = 0;
/// Waiting for the caller to pick one of the candidates written for it.
pub const STATUS_DECISION: u8 = 1;
/// The game has ended; its final scores are in the `scores` output. Reset the slot to reuse it.
pub const STATUS_FINISHED: u8 = 2;

struct Slot {
    game: Game,
    controllers: [Controller; 2],
    rng: Rng,
    status: u8,
    /// Candidates offered for the pending external decision, in the order they were written.
    offered: Vec<Candidate>,
}

/// The caller's buffers for one `prepare` call. With `n` slots and at most `k` candidates each:
pub struct Buffers<'a> {
    /// `n * k * 225` board codes.
    pub board: &'a mut [u8],
    /// `n * k * 27` leave counts.
    pub leave: &'a mut [u8],
    /// `n * k * 27` unseen-pool counts.
    pub unseen: &'a mut [u8],
    /// `n * k * META` integers.
    pub meta: &'a mut [i16],
    /// `n * k` static leave values.
    pub leave_value: &'a mut [f32],
    /// `n`: number of candidates written for each slot (0 unless its status is `DECISION`).
    pub count: &'a mut [i32],
    /// `n`: seat to move in each slot with a pending decision.
    pub mover: &'a mut [i8],
    /// `n`: `STATUS_*`.
    pub status: &'a mut [u8],
    /// `n * 2`: current scores by seat (final scores once the status is `FINISHED`).
    pub scores: &'a mut [i32],
}

pub struct BatchEnv {
    slots: Vec<Slot>,
    max_candidates: usize,
    max_exchanges: usize,
}

impl BatchEnv {
    /// `slots` empty game slots offering at most `max_candidates` moves per decision, of which at
    /// most `max_exchanges` are exchanges.
    pub fn new(slots: usize, max_candidates: usize, max_exchanges: usize) -> BatchEnv {
        assert!(max_candidates >= 2 && max_exchanges < max_candidates);
        let empty = || Slot {
            game: Game::new(0),
            controllers: [Controller::External; 2],
            rng: Rng::new(0),
            status: STATUS_EMPTY,
            offered: Vec::new(),
        };
        BatchEnv { slots: (0..slots).map(|_| empty()).collect(), max_candidates, max_exchanges }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn max_candidates(&self) -> usize {
        self.max_candidates
    }

    /// Start a new game in `slot`. The seed fixes the bag and the bots' random choices.
    pub fn reset(&mut self, slot: usize, seed: u64, controllers: [Controller; 2]) {
        let s = &mut self.slots[slot];
        s.game = Game::new(seed);
        s.controllers = controllers;
        s.rng = Rng::new(seed ^ 0xB07_5EED);
        s.status = STATUS_DECISION; // settled by the next `prepare`
        s.offered.clear();
    }

    /// Empty `slot` so it is skipped until it is reset.
    pub fn clear(&mut self, slot: usize) {
        self.slots[slot].status = STATUS_EMPTY;
        self.slots[slot].offered.clear();
    }

    pub fn game(&self, slot: usize) -> &Game {
        &self.slots[slot].game
    }

    /// The move behind candidate `index` of the pending decision in `slot`.
    pub fn offered(&self, slot: usize, index: usize) -> Option<&Candidate> {
        self.slots[slot].offered.get(index)
    }

    /// Advance every game to its next external decision (or its end) and write the candidates.
    pub fn prepare(&mut self, gaddag: &Gaddag, leaves: &LeaveModel, out: Buffers<'_>) {
        let (n, k) = (self.slots.len(), self.max_candidates);
        assert_eq!(out.board.len(), n * k * SQUARES);
        assert_eq!(out.leave.len(), n * k * TILE_TYPES);
        assert_eq!(out.unseen.len(), n * k * TILE_TYPES);
        assert_eq!(out.meta.len(), n * k * META);
        assert_eq!(out.leave_value.len(), n * k);
        assert!(out.count.len() == n && out.mover.len() == n && out.status.len() == n && out.scores.len() == 2 * n);
        let max_exchanges = self.max_exchanges;

        self.slots
            .par_iter_mut()
            .zip(out.board.par_chunks_mut(k * SQUARES))
            .zip(out.leave.par_chunks_mut(k * TILE_TYPES))
            .zip(out.unseen.par_chunks_mut(k * TILE_TYPES))
            .zip(out.meta.par_chunks_mut(k * META))
            .zip(out.leave_value.par_chunks_mut(k))
            .zip(out.count.par_iter_mut())
            .zip(out.mover.par_iter_mut())
            .zip(out.status.par_iter_mut())
            .zip(out.scores.par_chunks_mut(2))
            .for_each_init(
                Scratch::default,
                |scratch, (((((((((slot, board), leave), unseen), meta), leave_value), count), mover), status), scores)| {
                    *count = 0;
                    *mover = -1;
                    if slot.status != STATUS_EMPTY {
                        // Bots move until an external seat is to move or the game is over.
                        while !slot.game.over {
                            match slot.controllers[slot.game.to_move] {
                                Controller::External => break,
                                Controller::Bot(bot) => {
                                    let mv = choose(bot, &slot.game, gaddag, leaves, &mut slot.rng, scratch);
                                    slot.game.apply(&mv);
                                }
                            }
                        }
                        if slot.game.over {
                            slot.status = STATUS_FINISHED;
                            slot.offered.clear();
                        } else {
                            slot.status = STATUS_DECISION;
                            select(&slot.game, gaddag, leaves, scratch, k, max_exchanges, &mut slot.offered);
                            let base = board_codes(&slot.game.board);
                            let pool = slot.game.unseen(slot.game.to_move);
                            for (i, candidate) in slot.offered.iter().enumerate() {
                                encode_afterstate(
                                    &slot.game,
                                    &base,
                                    &pool,
                                    candidate,
                                    &mut board[i * SQUARES..(i + 1) * SQUARES],
                                    &mut leave[i * TILE_TYPES..(i + 1) * TILE_TYPES],
                                    &mut unseen[i * TILE_TYPES..(i + 1) * TILE_TYPES],
                                    &mut meta[i * META..(i + 1) * META],
                                );
                                leave_value[i] = candidate.leave_value;
                            }
                            *count = slot.offered.len() as i32;
                            *mover = slot.game.to_move as i8;
                        }
                    }
                    *status = slot.status;
                    scores.copy_from_slice(&slot.game.scores);
                },
            );
    }

    /// Play `actions[slot]` (an index into the candidates written by the last `prepare`) in every
    /// slot with a pending decision. Other entries are ignored.
    pub fn apply(&mut self, actions: &[i32]) -> Result<(), String> {
        assert_eq!(actions.len(), self.slots.len());
        for (index, (slot, &action)) in self.slots.iter_mut().zip(actions).enumerate() {
            if slot.status != STATUS_DECISION || slot.offered.is_empty() {
                continue;
            }
            let Some(candidate) = usize::try_from(action).ok().and_then(|a| slot.offered.get(a)) else {
                return Err(format!("slot {index}: action {action} is not one of {} candidates", slot.offered.len()));
            };
            let mv = candidate.mv;
            slot.game.apply(&mv);
            slot.offered.clear();
        }
        Ok(())
    }
}

/// Choose which moves to offer: the best placements by static value, up to `max_exchanges`
/// exchanges, and pass when the rules of [`candidates`] include it. Sorted best first.
fn select(
    game: &Game,
    gaddag: &Gaddag,
    leaves: &LeaveModel,
    scratch: &mut Scratch,
    max_candidates: usize,
    max_exchanges: usize,
    offered: &mut Vec<Candidate>,
) {
    candidates(game, gaddag, leaves, scratch);
    // Stable sort: equal values keep generation order, so the selection is reproducible.
    scratch.candidates.sort_by(|a, b| b.static_value().total_cmp(&a.static_value()));
    offered.clear();
    let is_exchange = |c: &&Candidate| matches!(c.mv, Move::Exchange(_));
    let pass = scratch.candidates.iter().find(|c| matches!(c.mv, Move::Pass)).copied();
    let exchanges = scratch.candidates.iter().filter(is_exchange).count().min(max_exchanges);
    let room = max_candidates - exchanges - pass.is_some() as usize;
    // Placements first, as the bot considers them, so that ties resolve the same way it does.
    offered.extend(scratch.candidates.iter().filter(|c| matches!(c.mv, Move::Play(_))).take(room).copied());
    offered.extend(scratch.candidates.iter().filter(is_exchange).take(exchanges).copied());
    offered.extend(pass);
    offered.sort_by(|a, b| b.static_value().total_cmp(&a.static_value()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{M_KIND, M_SCORE, M_SPREAD, M_TERMINAL};
    use crate::selfplay::{play_game, Player};

    fn gaddag() -> Gaddag {
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

    struct Arrays {
        board: Vec<u8>,
        leave: Vec<u8>,
        unseen: Vec<u8>,
        meta: Vec<i16>,
        leave_value: Vec<f32>,
        count: Vec<i32>,
        mover: Vec<i8>,
        status: Vec<u8>,
        scores: Vec<i32>,
    }

    impl Arrays {
        fn new(n: usize, k: usize) -> Arrays {
            Arrays {
                board: vec![0; n * k * SQUARES],
                leave: vec![0; n * k * TILE_TYPES],
                unseen: vec![0; n * k * TILE_TYPES],
                meta: vec![0; n * k * META],
                leave_value: vec![0.0; n * k],
                count: vec![0; n],
                mover: vec![0; n],
                status: vec![0; n],
                scores: vec![0; 2 * n],
            }
        }

        fn buffers(&mut self) -> Buffers<'_> {
            Buffers {
                board: &mut self.board,
                leave: &mut self.leave,
                unseen: &mut self.unseen,
                meta: &mut self.meta,
                leave_value: &mut self.leave_value,
                count: &mut self.count,
                mover: &mut self.mover,
                status: &mut self.status,
                scores: &mut self.scores,
            }
        }
    }

    /// Run every slot to the end, always picking candidate `pick(count)`.
    fn run(env: &mut BatchEnv, gaddag: &Gaddag, leaves: &LeaveModel, arrays: &mut Arrays, pick: fn(i32) -> i32) {
        for _ in 0..500 {
            env.prepare(gaddag, leaves, arrays.buffers());
            if arrays.status.iter().all(|&s| s != STATUS_DECISION) {
                return;
            }
            let actions: Vec<i32> = arrays.count.iter().map(|&c| pick(c)).collect();
            env.apply(&actions).unwrap();
        }
        panic!("games did not finish");
    }

    #[test]
    fn picking_candidate_zero_reproduces_the_greedy_leave_bot() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (n, k) = (6, 16);
        let mut env = BatchEnv::new(n, k, 3);
        for slot in 0..n {
            env.reset(slot, 100 + slot as u64, [Controller::External; 2]);
        }
        let mut arrays = Arrays::new(n, k);
        run(&mut env, &gaddag, &leaves, &mut arrays, |_| 0);
        let bot = Player { bot: Bot::GreedyLeave, leaves: &leaves };
        let mut scratch = Scratch::default();
        for slot in 0..n {
            let expected = play_game(&gaddag, [bot, bot], 100 + slot as u64, &mut scratch);
            assert_eq!(arrays.status[slot], STATUS_FINISHED);
            assert_eq!(&arrays.scores[2 * slot..2 * slot + 2], &expected.scores, "slot {slot}");
        }
    }

    #[test]
    fn bot_seats_are_played_inside_the_environment() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let mut env = BatchEnv::new(2, 8, 2);
        env.reset(0, 5, [Controller::External, Controller::Bot(Bot::Greedy)]);
        env.reset(1, 5, [Controller::Bot(Bot::Greedy), Controller::External]);
        let mut arrays = Arrays::new(2, 8);
        env.prepare(&gaddag, &leaves, arrays.buffers());
        assert_eq!(arrays.mover, [0, 1], "each slot stops at its external seat");
        assert_eq!(env.game(0).turn, 0);
        assert_eq!(env.game(1).turn, 1, "the bot already moved in slot 1");
        run(&mut env, &gaddag, &leaves, &mut arrays, |_| 0);
        let (greedy, leave) = (Player { bot: Bot::Greedy, leaves: &leaves }, Player { bot: Bot::GreedyLeave, leaves: &leaves });
        let mut scratch = Scratch::default();
        assert_eq!(&arrays.scores[0..2], &play_game(&gaddag, [leave, greedy], 5, &mut scratch).scores);
        assert_eq!(&arrays.scores[2..4], &play_game(&gaddag, [greedy, leave], 5, &mut scratch).scores);
    }

    #[test]
    fn candidates_are_sorted_bounded_and_consistent_with_their_metadata() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (n, k) = (4, 12);
        let mut env = BatchEnv::new(n, k, 3);
        for slot in 0..n {
            env.reset(slot, 40 + slot as u64, [Controller::External; 2]);
        }
        let mut arrays = Arrays::new(n, k);
        for _ in 0..12 {
            env.prepare(&gaddag, &leaves, arrays.buffers());
            for slot in 0..n {
                if arrays.status[slot] != STATUS_DECISION {
                    continue;
                }
                let count = arrays.count[slot] as usize;
                assert!((1..=k).contains(&count));
                let meta = |i: usize, field: usize| arrays.meta[(slot * k + i) * META + field];
                let game = env.game(slot);
                let mut exchanges = 0;
                for i in 0..count {
                    let offered = env.offered(slot, i).unwrap();
                    assert_eq!(meta(i, M_SCORE) as i32, offered.score);
                    assert_eq!(meta(i, M_SPREAD) as i32, game.spread(game.to_move) + offered.score);
                    assert_eq!(meta(i, M_TERMINAL) as u8, offered.terminal);
                    exchanges += (meta(i, M_KIND) == 1) as usize;
                    if i > 0 {
                        assert!(offered.static_value() <= env.offered(slot, i - 1).unwrap().static_value());
                    }
                    // Tiles are conserved in every afterstate: board + leave + unseen = 100.
                    let on_board = arrays.board[(slot * k + i) * SQUARES..][..SQUARES].iter().filter(|&&c| c != 0).count();
                    let held: usize = arrays.leave[(slot * k + i) * TILE_TYPES..][..TILE_TYPES].iter().map(|&c| c as usize).sum();
                    let hidden: usize = arrays.unseen[(slot * k + i) * TILE_TYPES..][..TILE_TYPES].iter().map(|&c| c as usize).sum();
                    assert_eq!(on_board + held + hidden, 100);
                }
                assert!(exchanges <= 3);
            }
            let actions: Vec<i32> = arrays.count.iter().map(|&c| (c - 1).max(0)).collect();
            env.apply(&actions).unwrap();
        }
        // An action that is not one of the candidates is an error wherever a decision is pending.
        env.prepare(&gaddag, &leaves, arrays.buffers());
        let pending = arrays.status.iter().any(|&s| s == STATUS_DECISION);
        assert_eq!(env.apply(&vec![99; n]).is_err(), pending);
    }

    #[test]
    fn empty_slots_are_skipped_and_can_be_reused() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let mut env = BatchEnv::new(3, 8, 2);
        env.reset(1, 9, [Controller::Bot(Bot::Greedy); 2]);
        let mut arrays = Arrays::new(3, 8);
        env.prepare(&gaddag, &leaves, arrays.buffers());
        assert_eq!(arrays.status, [STATUS_EMPTY, STATUS_FINISHED, STATUS_EMPTY], "a bot-only game runs to its end");
        assert_eq!(arrays.count, [0, 0, 0]);
        env.clear(1);
        env.reset(2, 9, [Controller::External; 2]);
        env.prepare(&gaddag, &leaves, arrays.buffers());
        assert_eq!(arrays.status, [STATUS_EMPTY, STATUS_EMPTY, STATUS_DECISION]);
    }
}
