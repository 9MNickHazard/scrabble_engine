//! Properties that must hold in every game: the invariants of docs/RULES.md under random play,
//! and that nothing the network sees depends on hidden information.
//!
//!     cargo test -p scrabble-core --test properties                          # 6,000 games
//!     cargo test -p scrabble-core --test properties -- --ignored --nocapture # 100,000 games

mod common;

use rayon::prelude::*;
use scrabble_core::belief::{apply_observed, infer};
use scrabble_core::board::SQUARES;
use scrabble_core::encode::{base_codes, encode_afterstate, META, M_BELIEF_INFERRED, M_OPPONENT_TILES};
use scrabble_core::gaddag::Gaddag;
use scrabble_core::game::{Game, SCORELESS_LIMIT};
use scrabble_core::leave::LeaveModel;
use scrabble_core::moves::Move;
use scrabble_core::policy::{candidates, Scratch};
use scrabble_core::rng::Rng;
use scrabble_core::tiles::{Rack, TILE_COUNTS, TILE_TYPES};
use scrabble_core::validate::check_move;

#[derive(Default, Clone, Copy)]
struct Fuzz {
    games: u64,
    turns: u64,
    plays: u64,
    exchanges: u64,
    passes: u64,
    went_out: u64,
    six_scoreless: u64,
    validated: u64,
}

/// Play one game choosing uniformly at random among *all* legal moves (so exchanges and passes
/// happen often, unlike in sensible play), checking the invariants after every turn.
fn fuzz_game(gaddag: &Gaddag, seed: u64, validate: bool) -> Fuzz {
    let mut out = Fuzz { games: 1, ..Fuzz::default() };
    let mut game = Game::new(seed);
    let mut rng = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 1);
    let mut tally = [0i32; 2];
    // Some games lean heavily on passing so the six-turn rule is exercised.
    let pass_heavy = seed % 5 == 0;
    while !game.over {
        let moves = game.legal_moves(gaddag);
        assert!(matches!(moves.last(), Some(Move::Pass)), "pass is always legal");
        let has_exchange = moves.iter().any(|m| matches!(m, Move::Exchange(_)));
        assert_eq!(has_exchange, game.bag.len() >= 7, "exchange availability, seed {seed}");
        let mv = if pass_heavy && rng.below(2) == 0 { Move::Pass } else { moves[rng.below(moves.len() as u32) as usize] };
        if validate {
            if let Err(reason) = check_move(gaddag, &game, &mv) {
                panic!("seed {seed} turn {}: {mv} rejected: {reason}\n{:?}", game.turn, game.board);
            }
            out.validated += 1;
        }
        let (mover, scoreless_before) = (game.to_move, game.scoreless_turns);
        let applied = game.apply(&mv);
        tally[mover] += applied.score;
        tally[0] += applied.adjustment[0];
        tally[1] += applied.adjustment[1];
        out.turns += 1;
        match mv {
            Move::Play(_) => {
                out.plays += 1;
                assert_eq!(game.scoreless_turns, 0);
            }
            Move::Exchange(_) => {
                out.exchanges += 1;
                assert_eq!(game.scoreless_turns, scoreless_before + 1);
            }
            Move::Pass => {
                out.passes += 1;
                assert_eq!(game.scoreless_turns, scoreless_before + 1);
            }
        }
        assert_eq!(game.tile_census(), TILE_COUNTS, "tiles not conserved, seed {seed}");
        assert_eq!(game.scores, tally, "score accounting, seed {seed}");
        assert_eq!(game.to_move, 1 - mover);
        if !game.over && !game.bag.is_empty() {
            assert_eq!((game.racks[0].len(), game.racks[1].len()), (7, 7), "racks not refilled, seed {seed}");
        }
        if game.over {
            if game.scoreless_turns >= SCORELESS_LIMIT {
                out.six_scoreless += 1;
                assert_eq!(applied.adjustment, [-game.racks[0].value(), -game.racks[1].value()]);
            } else {
                out.went_out += 1;
                assert!(game.racks[mover].is_empty(), "a game that ends on a play ends by going out");
                assert!(game.bag.is_empty());
                assert_eq!(applied.adjustment[mover], 2 * game.racks[1 - mover].value());
                assert_eq!(applied.adjustment[1 - mover], 0);
            }
            assert!(game.legal_moves(gaddag).is_empty());
        }
        assert!(game.turn < 1_000, "game does not end, seed {seed}");
    }
    out
}

fn fuzz(gaddag: &Gaddag, seeds: std::ops::Range<u64>, validate_every: u64) -> Fuzz {
    seeds
        .into_par_iter()
        .map(|seed| fuzz_game(gaddag, seed, seed % validate_every == 0))
        .reduce(Fuzz::default, |a, b| Fuzz {
            games: a.games + b.games,
            turns: a.turns + b.turns,
            plays: a.plays + b.plays,
            exchanges: a.exchanges + b.exchanges,
            passes: a.passes + b.passes,
            went_out: a.went_out + b.went_out,
            six_scoreless: a.six_scoreless + b.six_scoreless,
            validated: a.validated + b.validated,
        })
}

fn report(f: &Fuzz, seconds: f64) {
    println!(
        "{} random games in {seconds:.0} s: {} turns ({} plays, {} exchanges, {} passes); {} ended by going out, \
         {} by six scoreless turns; {} moves re-checked by the validator; 0 invariant violations",
        f.games, f.turns, f.plays, f.exchanges, f.passes, f.went_out, f.six_scoreless, f.validated
    );
}

#[test]
fn random_games_keep_the_invariants_mini_lexicon() {
    let gaddag = common::mini_gaddag();
    let start = std::time::Instant::now();
    let result = fuzz(&gaddag, 0..2_000, 1);
    report(&result, start.elapsed().as_secs_f64());
    assert!(result.six_scoreless > 0 && result.exchanges > 0 && result.plays > 0);
}

#[test]
fn random_games_keep_the_invariants() {
    let Some(gaddag) = common::real_gaddag() else { return };
    let start = std::time::Instant::now();
    let result = fuzz(gaddag, 0..4_000, 10);
    report(&result, start.elapsed().as_secs_f64());
    assert!(result.six_scoreless > 0 && result.exchanges > 0 && result.plays > 0);
}

/// The acceptance run from docs/PLAN.md.
#[test]
#[ignore = "long: run with --ignored"]
fn random_games_keep_the_invariants_full() {
    let Some(gaddag) = common::real_gaddag() else { return };
    let start = std::time::Instant::now();
    let result = fuzz(gaddag, 1_000_000..1_100_000, 50);
    report(&result, start.elapsed().as_secs_f64());
    assert_eq!(result.games, 100_000);
}

type Encoded = Vec<(Move, [u8; SQUARES], [u8; TILE_TYPES], [u8; TILE_TYPES], [i16; META], f32)>;

fn encode_all(game: &Game, gaddag: &Gaddag, leaves: &LeaveModel) -> Encoded {
    let mut scratch = Scratch::default();
    candidates(game, gaddag, leaves, &mut scratch);
    let base = base_codes(game);
    let pool = game.unseen(game.to_move);
    scratch
        .candidates
        .iter()
        .map(|candidate| {
            let (mut board, mut leave, mut unseen, mut meta) = ([0; SQUARES], [0; TILE_TYPES], [0; TILE_TYPES], [0; META]);
            encode_afterstate(game, &base, &pool, candidate, &mut board, &mut leave, &mut unseen, &mut meta);
            (candidate.mv, board, leave, unseen, meta, candidate.leave_value)
        })
        .collect()
}

/// While tiles remain in the bag the opponent's rack is hidden. Swapping it for other tiles from
/// the bag must change nothing the mover's network is given: not the candidate list, not one
/// byte of any afterstate. The old project built its training targets from the opponent's rack.
#[test]
fn nothing_the_network_sees_depends_on_the_hidden_rack() {
    let (hidden, _) = hidden_rack_check(&common::mini_gaddag(), 0..60);
    assert!(hidden > 200, "{hidden} positions checked");
}

/// The same with the real word list, where games reach the empty-bag endgame.
#[test]
fn nothing_the_network_sees_depends_on_the_hidden_rack_real_lexicon() {
    let Some(gaddag) = common::real_gaddag() else { return };
    let (hidden, open) = hidden_rack_check(gaddag, 0..40);
    assert!(hidden > 200 && open > 200, "{hidden} hidden-rack positions, {open} endgame afterstates");
}

fn hidden_rack_check(gaddag: &Gaddag, seeds: std::ops::Range<u64>) -> (usize, usize) {
    let mut leaves = LeaveModel::zero();
    for (i, weight) in leaves.weights[0].iter_mut().enumerate() {
        *weight = (i % 7) as f32 - 3.0; // arbitrary non-zero leave values
    }
    let (mut hidden_checked, mut open_checked, mut with_inference) = (0, 0, 0);
    let mut scratch = Scratch::default();
    for seed in seeds {
        let mut game = Game::new(seed);
        let mut rng = Rng::new(seed + 99);
        let mut belief_rng = Rng::new(seed + 7);
        while !game.over {
            let (me, opponent) = (game.to_move, 1 - game.to_move);
            if game.bag.len() >= 7 {
                // Deal the opponent a different rack from the same hidden pool.
                let mut pool: Vec<u8> = game.bag.tiles().to_vec();
                pool.extend(game.racks[opponent].tiles());
                rng.shuffle(&mut pool);
                let mut other_rack = Rack::new();
                for &tile in &pool[..game.racks[opponent].len()] {
                    other_rack.add(tile);
                }
                let mut racks = game.racks;
                racks[opponent] = other_rack;
                let mut twin = Game::from_position(game.board.clone(), racks, &pool[other_rack.len()..], game.scores, me, 5);
                twin.scoreless_turns = game.scoreless_turns;
                // Public: what each side did last turn and where, how long since each last placed
                // seven tiles, and what each has inferred from the other's play (which is worked
                // out from public facts alone: see the test of the inference below).
                twin.last_turn = game.last_turn;
                twin.last_play = game.last_play;
                twin.since_bingo = game.since_bingo;
                twin.belief = game.belief;
                assert_eq!(twin.tile_census(), TILE_COUNTS);
                if other_rack != game.racks[opponent] {
                    let encoded = encode_all(&game, gaddag, &leaves);
                    assert!(encoded == encode_all(&twin, gaddag, &leaves), "seed {seed}");
                    hidden_checked += 1;
                    with_inference += (encoded[0].4[M_BELIEF_INFERRED] == 1) as usize;
                }
            } else if game.bag.is_empty() {
                // With the bag empty the unseen pool IS the opponent's rack, and that is public.
                for (mv, _, _, unseen, meta, _) in &encode_all(&game, gaddag, &leaves) {
                    if !matches!(mv, Move::Exchange(_)) {
                        assert_eq!(unseen, &game.racks[opponent].counts);
                    }
                    assert_eq!(meta[M_OPPONENT_TILES] as usize, game.racks[opponent].len());
                    open_checked += 1;
                }
            }
            let moves = game.legal_moves(gaddag);
            let plays: Vec<&Move> = moves.iter().filter(|m| matches!(m, Move::Play(_))).collect();
            let mv = if plays.is_empty() { moves[moves.len() - 1] } else { *plays[rng.below(plays.len() as u32) as usize] };
            apply_observed(&mut game, &mv, gaddag, &leaves, 2, &mut belief_rng, &mut scratch);
        }
    }
    assert!(with_inference > hidden_checked / 2, "most of the positions carried an inference: {with_inference} of {hidden_checked}");
    println!("{hidden_checked} positions with a hidden rack swapped, {open_checked} endgame afterstates checked");
    (hidden_checked, open_checked)
}

/// What a placement lets the other player infer about the mover's rack must be worked out from
/// what that player can see: the board, the move, and the pool of unseen tiles as a whole. So if
/// the tiles the mover actually kept are exchanged for others from the bag, with the same move
/// played, the inference must not change by one count.
#[test]
fn the_inference_about_a_rack_does_not_depend_on_the_tiles_actually_kept() {
    let gaddag = common::mini_gaddag();
    let mut leaves = LeaveModel::zero();
    for (i, weight) in leaves.weights[0].iter_mut().enumerate() {
        *weight = (i % 7) as f32 - 3.0;
    }
    let mut scratch = Scratch::default();
    let (mut checked, mut inferred) = (0, 0);
    for seed in 0..60u64 {
        let mut game = Game::new(seed);
        let mut rng = Rng::new(seed + 31);
        while !game.over {
            let me = game.to_move;
            let moves = game.legal_moves(&gaddag);
            let plays: Vec<&Move> = moves.iter().filter(|m| matches!(m, Move::Play(_))).collect();
            let mv = if plays.is_empty() { moves[moves.len() - 1] } else { *plays[rng.below(plays.len() as u32) as usize] };
            if let Move::Play(play) = &mv {
                let used = play.tiles_used();
                let mut kept = game.racks[me];
                kept.remove_all(&used);
                // The same hidden pool (the bag and the kept tiles), dealt differently.
                let mut pool: Vec<u8> = game.bag.tiles().to_vec();
                pool.extend(kept.tiles());
                rng.shuffle(&mut pool);
                let mut other = used;
                for &tile in &pool[..kept.len()] {
                    other.add(tile);
                }
                let mut racks = game.racks;
                racks[me] = other;
                let mut twin = Game::from_position(game.board.clone(), racks, &pool[kept.len()..], game.scores, me, 5);
                twin.scoreless_turns = game.scoreless_turns;
                assert_eq!(twin.tile_census(), TILE_COUNTS);
                if other != game.racks[me] {
                    let stream = seed * 1000 + game.turn as u64;
                    let ours = infer(&game, play, &gaddag, &leaves, 5, &mut Rng::new(stream), &mut scratch);
                    let theirs = infer(&twin, play, &gaddag, &leaves, 5, &mut Rng::new(stream), &mut scratch);
                    assert_eq!(ours, theirs, "seed {seed}, turn {}", game.turn);
                    checked += 1;
                    inferred += ours.is_some() as usize;
                }
            }
            game.apply(&mv);
        }
    }
    println!("{checked} placements checked, {inferred} with an inference");
    assert!(checked > 300 && inferred > 200, "{checked} placements checked, {inferred} with an inference");
}
