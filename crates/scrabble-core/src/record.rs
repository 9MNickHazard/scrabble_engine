//! Games saved as the moves that were played, and training positions read back out of them.
//!
//! Training from saved games needs a great many games (tens of millions) and one position from
//! each at a time, a different one on every pass. So a game is stored as its moves, a few
//! hundred bytes, and a position is rebuilt on demand by replaying the game up to a chosen turn:
//! replaying needs no move generation, only the moves themselves.
//!
//! One record:
//!
//! ```text
//! seed            u64   the game is Game::new(seed) with these moves applied in order
//! scores          2 x i16   final scores by seat
//! flags           u8    bit 0: the last move went out
//! moves           u8    number of moves
//! then per move:
//!   0                          pass
//!   1, n, n tiles              exchange of n tiles
//!   2 or 3, lane, start, len,  placement across (2) or down (3), as `Play` describes it:
//!     tiles played, score i16,   `len` bytes of `Play::word`
//!     len bytes
//! ```
//!
//! Records are concatenated in one file; a second file holds the byte offset of each record and,
//! last, the total length (`GameFile`).

use crate::belief::apply_observed;
use crate::board::{N, SQUARES};
use crate::encode::{base_codes, encode_afterstate, META, M_SPREAD};
use crate::gaddag::Gaddag;
use crate::game::Game;
use crate::leave::LeaveModel;
use crate::moves::{Move, Play};
use crate::policy::{candidate, candidates, choose, Bot, Candidate, Scratch};
use crate::rng::Rng;
use crate::tiles::{Rack, RACK_SIZE, TILE_TYPES};
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

const HEADER: usize = 14;
const FLAG_WENT_OUT: u8 = 1;

/// Numbers written for each position besides the afterstate itself.
pub const TARGETS: usize = 8;
/// 1 if the mover went on to win the game, 0.5 for a draw, 0 for a loss.
pub const T_WIN: usize = 0;
/// Points of spread the mover went on to gain after this move.
pub const T_SPREAD: usize = 1;
/// Points the opponent's next move scored, whether it placed all seven tiles, and whether there
/// was a next move at all.
pub const T_OPP_SCORE: usize = 2;
pub const T_OPP_BINGO: usize = 3;
pub const T_HAS_OPP: usize = 4;
/// The same for the mover's own next move, two turns on.
pub const T_OWN_SCORE: usize = 5;
pub const T_OWN_BINGO: usize = 6;
pub const T_HAS_OWN: usize = 7;

/// How the first moves of a recorded game are varied, so that the games are not all the same
/// few openings. Each game draws a number of plies between 0 and `max_plies`; on those plies the
/// move is sampled from the best `top` by static value, with probability proportional to
/// `exp(static value / temperature)`. Every later move is the static evaluator's first choice.
#[derive(Clone, Copy, Debug)]
pub struct Opening {
    pub max_plies: u8,
    pub temperature: f32,
    pub top: usize,
}

impl Default for Opening {
    fn default() -> Opening {
        Opening { max_plies: 4, temperature: 6.0, top: 16 }
    }
}

/// A decoded record.
pub struct Recorded {
    pub seed: u64,
    pub scores: [i32; 2],
    pub went_out: bool,
    pub moves: Vec<Move>,
}

/// Play one game by the static evaluator (score + leave value), with a varied opening.
pub fn play_recorded(gaddag: &Gaddag, leaves: &LeaveModel, seed: u64, opening: &Opening, scratch: &mut Scratch) -> Recorded {
    let mut game = Game::new(seed);
    let mut rng = Rng::new(seed ^ 0x0BE9_1265);
    let varied = if opening.max_plies == 0 { 0 } else { rng.below(opening.max_plies as u32 + 1) };
    let mut moves = Vec::with_capacity(32);
    let mut went_out = false;
    while !game.over && moves.len() < u8::MAX as usize {
        let mv = if (moves.len() as u32) < varied && opening.temperature > 0.0 {
            candidates(&game, gaddag, leaves, scratch);
            scratch.candidates.sort_by(|a, b| b.static_value().total_cmp(&a.static_value()));
            let pool = &scratch.candidates[..scratch.candidates.len().min(opening.top.max(1))];
            let best = pool[0].static_value();
            let weights: Vec<f64> = pool.iter().map(|c| (((c.static_value() - best) / opening.temperature) as f64).exp()).collect();
            let mut draw = rng.unit() * weights.iter().sum::<f64>();
            let mut chosen = pool.len() - 1;
            for (i, weight) in weights.iter().enumerate() {
                if draw < *weight {
                    chosen = i;
                    break;
                }
                draw -= weight;
            }
            pool[chosen].mv
        } else {
            choose(Bot::GreedyLeave, &game, gaddag, leaves, &mut rng, scratch)
        };
        game.apply(&mv);
        went_out = game.over && matches!(mv, Move::Play(_));
        moves.push(mv);
    }
    Recorded { seed, scores: game.scores, went_out, moves }
}

/// Append the record of `game` to `out`.
pub fn encode(game: &Recorded, out: &mut Vec<u8>) {
    assert!(game.moves.len() <= u8::MAX as usize);
    out.extend_from_slice(&game.seed.to_le_bytes());
    for score in game.scores {
        out.extend_from_slice(&(score.clamp(i16::MIN as i32, i16::MAX as i32) as i16).to_le_bytes());
    }
    out.push(if game.went_out { FLAG_WENT_OUT } else { 0 });
    out.push(game.moves.len() as u8);
    for mv in &game.moves {
        match mv {
            Move::Pass => out.push(0),
            Move::Exchange(tiles) => {
                out.push(1);
                out.push(tiles.len() as u8);
                out.extend(tiles.tiles());
            }
            Move::Play(play) => {
                out.push(if play.down { 3 } else { 2 });
                out.extend_from_slice(&[play.lane, play.start, play.len, play.tiles_played]);
                out.extend_from_slice(&play.score.to_le_bytes());
                out.extend_from_slice(&play.word[..play.len as usize]);
            }
        }
    }
}

/// Read one record back.
pub fn decode(bytes: &[u8]) -> Result<Recorded, String> {
    let short = || "a game record is cut short".to_string();
    if bytes.len() < HEADER {
        return Err(short());
    }
    let seed = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let score = |at: usize| i16::from_le_bytes([bytes[at], bytes[at + 1]]) as i32;
    let (scores, went_out, count) = ([score(8), score(10)], bytes[12] & FLAG_WENT_OUT != 0, bytes[13] as usize);
    let mut moves = Vec::with_capacity(count);
    let mut at = HEADER;
    let mut take = |n: usize| -> Result<&[u8], String> {
        let slice = bytes.get(at..at + n).ok_or_else(short)?;
        at += n;
        Ok(slice)
    };
    for _ in 0..count {
        let mv = match take(1)?[0] {
            0 => Move::Pass,
            1 => {
                let n = take(1)?[0] as usize;
                let mut rack = Rack::new();
                for &tile in take(n)? {
                    if tile as usize >= TILE_TYPES || rack.len() == RACK_SIZE {
                        return Err("a game record holds an impossible exchange".into());
                    }
                    rack.add(tile);
                }
                Move::Exchange(rack)
            }
            kind @ (2 | 3) => {
                let head = take(6)?;
                let (lane, start, len, tiles_played) = (head[0], head[1], head[2], head[3]);
                let score = i16::from_le_bytes([head[4], head[5]]);
                if lane as usize >= N || len as usize > N || start as usize + len as usize > N {
                    return Err("a game record holds a placement off the board".into());
                }
                let mut word = [0u8; N];
                word[..len as usize].copy_from_slice(take(len as usize)?);
                Move::Play(Play { down: kind == 3, lane, start, len, tiles_played, score, word })
            }
            other => return Err(format!("a game record holds an unknown move kind {other}")),
        };
        moves.push(mv);
    }
    Ok(Recorded { seed, scores, went_out, moves })
}

/// What `generate` reports about the games it played.
#[derive(Clone, Copy, Default, Debug)]
pub struct Generated {
    pub games: u64,
    pub moves: u64,
    pub score_sum: i64,
    pub first_player_wins: f64,
    pub bingos: u64,
}

/// Play `count` games with seeds from `first_seed` on, in parallel. Returns the records one
/// after another, the length of each, and totals. The same arguments always give the same bytes.
pub fn generate(gaddag: &Gaddag, leaves: &LeaveModel, first_seed: u64, count: usize, opening: &Opening) -> (Vec<u8>, Vec<u32>, Generated) {
    let records: Vec<(Vec<u8>, Generated)> = (0..count as u64)
        .into_par_iter()
        .map_init(Scratch::default, |scratch, offset| {
            let game = play_recorded(gaddag, leaves, first_seed + offset, opening, scratch);
            let mut bytes = Vec::with_capacity(512);
            encode(&game, &mut bytes);
            let stats = Generated {
                games: 1,
                moves: game.moves.len() as u64,
                score_sum: (game.scores[0] + game.scores[1]) as i64,
                first_player_wins: match game.scores[0].cmp(&game.scores[1]) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Less => 0.0,
                },
                bingos: game.moves.iter().filter(|m| matches!(m, Move::Play(p) if p.tiles_played as usize == RACK_SIZE)).count() as u64,
            };
            (bytes, stats)
        })
        .collect();
    let mut bytes = Vec::with_capacity(records.iter().map(|(b, _)| b.len()).sum());
    let mut lengths = Vec::with_capacity(count);
    let mut totals = Generated::default();
    for (record, stats) in &records {
        bytes.extend_from_slice(record);
        lengths.push(record.len() as u32);
        totals.games += stats.games;
        totals.moves += stats.moves;
        totals.score_sum += stats.score_sum;
        totals.first_player_wins += stats.first_player_wins;
        totals.bingos += stats.bingos;
    }
    (bytes, lengths, totals)
}

/// Where one position's arrays go.
pub struct Slot<'a> {
    pub board: &'a mut [u8],
    pub leave: &'a mut [u8],
    pub unseen: &'a mut [u8],
    pub meta: &'a mut [i16],
    pub leave_value: &'a mut f32,
    pub targets: &'a mut [f32],
    /// 450 bytes: 1 on each square the opponent's next move places a tile, then the same for the
    /// mover's own next move.
    pub squares: &'a mut [u8],
}

/// Rebuild one position of `game` and write it to `out`: the afterstate of the move played on
/// the turn that `pick` (in `[0, 1)`) selects, uniformly among the turns that leave something
/// to predict (a move that goes out ends the game, so it is never chosen). Returns the turn.
///
/// `belief_samples` > 0 also infers what the mover can tell about the opponent's rack from the
/// opponent's previous move, as the environment does during play (`belief.rs`).
pub fn position(
    game: &Recorded,
    pick: f32,
    gaddag: &Gaddag,
    leaves: &LeaveModel,
    belief_samples: usize,
    scratch: &mut Scratch,
    out: Slot<'_>,
) -> Result<usize, String> {
    let eligible = game.moves.len() - game.went_out as usize;
    if eligible == 0 {
        return Err(format!("game {} has no position to learn from", game.seed));
    }
    let turn = ((pick.clamp(0.0, 0.999_999) as f64 * eligible as f64) as usize).min(eligible - 1);
    let mut state = Game::new(game.seed);
    for (index, mv) in game.moves[..turn].iter().enumerate() {
        if state.over {
            return Err(format!("game {} carries on after its end", game.seed));
        }
        if index + 1 == turn && belief_samples > 0 {
            // Only the opponent's last move matters to what the mover has inferred.
            let mut rng = Rng::new(game.seed ^ ((turn as u64) << 32) ^ 0xBE11_EF5);
            apply_observed(&mut state, mv, gaddag, leaves, belief_samples, &mut rng, scratch);
        } else {
            state.apply(mv);
        }
    }
    if state.over {
        return Err(format!("game {} carries on after its end", game.seed));
    }
    let mover = state.to_move;
    let described: Candidate = candidate(&state, leaves, game.moves[turn]);
    encode_afterstate(&state, &base_codes(&state), &state.unseen(mover), &described, out.board, out.leave, out.unseen, out.meta);
    *out.leave_value = described.leave_value;

    let final_spread = game.scores[mover] - game.scores[1 - mover];
    out.targets[T_WIN] = match final_spread.cmp(&0) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Equal => 0.5,
        std::cmp::Ordering::Less => 0.0,
    };
    out.targets[T_SPREAD] = (final_spread - out.meta[M_SPREAD] as i32) as f32;
    out.squares.fill(0);
    for (ahead, (score, bingo, has)) in [(1, (T_OPP_SCORE, T_OPP_BINGO, T_HAS_OPP)), (2, (T_OWN_SCORE, T_OWN_BINGO, T_HAS_OWN))] {
        let next = game.moves.get(turn + ahead);
        out.targets[has] = next.is_some() as u8 as f32;
        out.targets[score] = next.map_or(0, Move::score) as f32;
        out.targets[bingo] = 0.0;
        if let Some(Move::Play(play)) = next {
            out.targets[bingo] = (play.tiles_played as usize == RACK_SIZE) as u8 as f32;
            for tile in play.placed() {
                out.squares[(ahead - 1) * SQUARES + tile.index] = 1;
            }
        }
    }
    Ok(turn)
}

/// A file of game records with its offsets, opened for reading at random.
pub struct GameFile {
    file: File,
    offsets: Vec<u64>,
}

impl GameFile {
    /// Open `games.bin` and `index.bin` in `directory`. The index holds the offset of every
    /// record and then the length of the file, as little-endian u64.
    ///
    /// With `games` given, only the first that many games are read, and it is an error if the
    /// files hold fewer. The files can hold more than a set records, or end in a torn entry, when
    /// the process writing a block was killed before it recorded the block; what lies past the
    /// recorded games is then ignored.
    pub fn open(directory: &Path, games: Option<usize>) -> Result<GameFile, String> {
        let index = std::fs::read(directory.join("index.bin")).map_err(|e| format!("cannot read the game index: {e}"))?;
        let mut offsets: Vec<u64> = index.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        match games {
            Some(games) => {
                if offsets.len() < games + 1 {
                    return Err(format!("the game files hold fewer than the {games} games the set records"));
                }
                offsets.truncate(games + 1);
            }
            None => {
                if index.len() % 8 != 0 || offsets.is_empty() {
                    return Err("the game index is damaged".into());
                }
            }
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err("the game index is not in order".into());
        }
        let file = File::open(directory.join("games.bin")).map_err(|e| format!("cannot open the games file: {e}"))?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        if *offsets.last().unwrap() > size {
            return Err("the game index points past the end of the games file".into());
        }
        Ok(GameFile { file, offsets })
    }

    /// Number of games.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The record of game `index`.
    pub fn read(&self, index: usize) -> Result<Recorded, String> {
        if index >= self.len() {
            return Err(format!("there is no game {index} among {}", self.len()));
        }
        let (start, end) = (self.offsets[index], self.offsets[index + 1]);
        let mut bytes = vec![0u8; (end - start) as usize];
        read_at(&self.file, &mut bytes, start).map_err(|e| format!("cannot read game {index}: {e}"))?;
        decode(&bytes)
    }

    /// Positions from many games at once, in parallel: `games[i]` and `picks[i]` choose the game
    /// and the turn ([`position`]), and row `i` of every output array receives the result. With
    /// `n` positions the arrays hold `n * 225`, `n * 27`, `n * 27`, `n * META`, `n`,
    /// `n * TARGETS`, `n * 450` and `n` entries.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        gaddag: &Gaddag,
        leaves: &LeaveModel,
        games: &[u64],
        picks: &[f32],
        belief_samples: usize,
        board: &mut [u8],
        leave: &mut [u8],
        unseen: &mut [u8],
        meta: &mut [i16],
        leave_value: &mut [f32],
        targets: &mut [f32],
        squares: &mut [u8],
        turns: &mut [i32],
    ) -> Result<(), String> {
        let n = games.len();
        let sizes_ok = picks.len() == n
            && board.len() == n * SQUARES
            && leave.len() == n * TILE_TYPES
            && unseen.len() == n * TILE_TYPES
            && meta.len() == n * META
            && leave_value.len() == n
            && targets.len() == n * TARGETS
            && squares.len() == n * 2 * SQUARES
            && turns.len() == n;
        if !sizes_ok {
            return Err(format!("array shapes do not match {n} positions"));
        }
        games
            .par_iter()
            .zip(picks.par_iter())
            .zip(board.par_chunks_mut(SQUARES))
            .zip(leave.par_chunks_mut(TILE_TYPES))
            .zip(unseen.par_chunks_mut(TILE_TYPES))
            .zip(meta.par_chunks_mut(META))
            .zip(leave_value.par_iter_mut())
            .zip(targets.par_chunks_mut(TARGETS))
            .zip(squares.par_chunks_mut(2 * SQUARES))
            .zip(turns.par_iter_mut())
            .with_min_len(16)
            .try_for_each_init(
                Scratch::default,
                |scratch, (((((((((game, pick), board), leave), unseen), meta), leave_value), targets), squares), turn)| {
                    let recorded = self.read(*game as usize)?;
                    let slot = Slot { board, leave, unseen, meta, leave_value, targets, squares };
                    *turn = position(&recorded, *pick, gaddag, leaves, belief_samples, scratch, slot)? as i32;
                    Ok(())
                },
            )
    }
}

#[cfg(windows)]
fn read_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buffer.is_empty() {
        match file.seek_read(buffer, offset)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buffer = &mut buffer[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buffer, offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{CODE_LETTER, CODE_PLACED, M_BELIEF_INFERRED, M_KIND, M_SCORE, M_TERMINAL};

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

    fn arrays() -> ([u8; SQUARES], [u8; TILE_TYPES], [u8; TILE_TYPES], [i16; META], f32, [f32; TARGETS], [u8; 2 * SQUARES]) {
        ([0; SQUARES], [0; TILE_TYPES], [0; TILE_TYPES], [0; META], 0.0, [0.0; TARGETS], [0; 2 * SQUARES])
    }

    #[test]
    fn a_record_round_trips_and_replays_to_the_same_game() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let mut scratch = Scratch::default();
        for seed in 0..40 {
            let game = play_recorded(&gaddag, &leaves, seed, &Opening::default(), &mut scratch);
            let mut bytes = Vec::new();
            encode(&game, &mut bytes);
            let back = decode(&bytes).unwrap();
            assert_eq!((back.seed, back.scores, back.went_out), (game.seed, game.scores, game.went_out));
            assert_eq!(back.moves, game.moves);
            // Applying the moves to a fresh game of that seed ends exactly where the game did.
            let mut replay = Game::new(seed);
            for mv in &back.moves {
                assert!(!replay.over);
                replay.apply(mv);
            }
            assert!(replay.over && replay.scores == game.scores, "seed {seed}");
            // Cut anywhere, a record is refused, not misread.
            assert!(decode(&bytes[..bytes.len() - 1]).is_err());
            assert!(decode(&bytes[..HEADER - 1]).is_err());
        }
    }

    #[test]
    fn games_differ_in_their_openings_and_then_follow_the_static_evaluator() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let mut scratch = Scratch::default();
        let (mut greedy_first, mut other_first) = (0, 0);
        for seed in 0..60 {
            let varied = play_recorded(&gaddag, &leaves, seed, &Opening::default(), &mut scratch);
            let plain = play_recorded(&gaddag, &leaves, seed, &Opening { max_plies: 0, ..Opening::default() }, &mut scratch);
            // With no varied plies the game is the greedy-leave bot against itself.
            let bot = crate::selfplay::Player { bot: Bot::GreedyLeave, leaves: &leaves };
            assert_eq!(crate::selfplay::play_game(&gaddag, [bot, bot], seed, &mut scratch).scores, plain.scores);
            if varied.moves[0] == plain.moves[0] {
                greedy_first += 1;
            } else {
                other_first += 1;
            }
            // Whatever the opening, once a game has gone four plies it is the static choice.
            let mut state = Game::new(seed);
            for (turn, mv) in varied.moves.iter().enumerate() {
                if turn >= 4 {
                    let mut rng = Rng::new(0);
                    assert_eq!(*mv, choose(Bot::GreedyLeave, &state, &gaddag, &leaves, &mut rng, &mut scratch), "seed {seed} turn {turn}");
                }
                state.apply(mv);
            }
        }
        assert!(greedy_first >= 5 && other_first > 10, "{greedy_first} games open with the static move, {other_first} with another");
    }

    #[test]
    fn generation_is_deterministic_and_in_seed_order() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (bytes, lengths, totals) = generate(&gaddag, &leaves, 500, 64, &Opening::default());
        let again = generate(&gaddag, &leaves, 500, 64, &Opening::default());
        assert_eq!((&bytes, &lengths), (&again.0, &again.1));
        assert_eq!((lengths.len(), totals.games), (64, 64));
        assert_eq!(lengths.iter().map(|&l| l as usize).sum::<usize>(), bytes.len());
        let mut at = 0;
        let mut moves = 0;
        for (offset, &length) in lengths.iter().enumerate() {
            let game = decode(&bytes[at..at + length as usize]).unwrap();
            assert_eq!(game.seed, 500 + offset as u64);
            moves += game.moves.len() as u64;
            at += length as usize;
        }
        assert_eq!(moves, totals.moves);
        assert!(totals.score_sum > 0 && totals.first_player_wins > 10.0);
    }

    #[test]
    fn a_position_is_the_afterstate_of_the_move_played_with_what_followed() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let mut scratch = Scratch::default();
        let mut checked = 0;
        for seed in 0..25 {
            let game = play_recorded(&gaddag, &leaves, seed, &Opening::default(), &mut scratch);
            let eligible = game.moves.len() - game.went_out as usize;
            for turn in 0..eligible {
                let pick = (turn as f32 + 0.5) / eligible as f32;
                let (mut board, mut leave, mut unseen, mut meta, mut leave_value, mut targets, mut squares) = arrays();
                let slot = Slot { board: &mut board, leave: &mut leave, unseen: &mut unseen, meta: &mut meta, leave_value: &mut leave_value, targets: &mut targets, squares: &mut squares };
                let got = position(&game, pick, &gaddag, &leaves, 0, &mut scratch, slot).unwrap();
                assert_eq!(got, turn, "the pick is spread evenly over the turns");

                // The same thing by hand.
                let mut state = Game::new(seed);
                for mv in &game.moves[..turn] {
                    state.apply(mv);
                }
                let mover = state.to_move;
                let mv = game.moves[turn];
                assert_eq!(meta[M_SCORE] as i32, candidate(&state, &leaves, mv).score);
                assert_eq!(meta[M_KIND], match mv { Move::Play(_) => 0, Move::Exchange(_) => 1, Move::Pass => 2 });
                assert_ne!(meta[M_TERMINAL], 1, "a move that goes out is never a training position");
                assert_eq!(meta[M_BELIEF_INFERRED], 0);
                let placed = board.iter().filter(|&&c| c & CODE_PLACED != 0).count();
                assert_eq!(placed, match mv { Move::Play(p) => p.tiles_played as usize, _ => 0 });
                assert_eq!(board.iter().filter(|&&c| c & CODE_LETTER != 0).count(), state.board.tile_count() + placed);
                // Tiles are conserved: board + leave + unseen = 100.
                let total: usize = board.iter().filter(|&&c| c & CODE_LETTER != 0).count()
                    + leave.iter().map(|&c| c as usize).sum::<usize>() + unseen.iter().map(|&c| c as usize).sum::<usize>();
                assert_eq!(total, 100);

                let final_spread = game.scores[mover] - game.scores[1 - mover];
                assert_eq!(targets[T_WIN], if final_spread > 0 { 1.0 } else if final_spread == 0 { 0.5 } else { 0.0 });
                assert_eq!(targets[T_SPREAD], (final_spread - (state.spread(mover) + candidate(&state, &leaves, mv).score)) as f32);
                for (ahead, score, bingo, has) in [(1, T_OPP_SCORE, T_OPP_BINGO, T_HAS_OPP), (2, T_OWN_SCORE, T_OWN_BINGO, T_HAS_OWN)] {
                    let map = &squares[(ahead - 1) * SQUARES..ahead * SQUARES];
                    match game.moves.get(turn + ahead) {
                        None => assert!(targets[has] == 0.0 && targets[score] == 0.0 && map.iter().all(|&s| s == 0)),
                        Some(next) => {
                            assert_eq!((targets[has], targets[score]), (1.0, next.score() as f32));
                            let expected: Vec<usize> = match next { Move::Play(p) => p.placed().map(|t| t.index).collect(), _ => Vec::new() };
                            assert_eq!(map.iter().filter(|&&s| s == 1).count(), expected.len());
                            assert!(expected.iter().all(|&i| map[i] == 1));
                            assert_eq!(targets[bingo] == 1.0, expected.len() == 7);
                        }
                    }
                }
                checked += 1;
            }
            // Asking for the inference changes only the metadata that carries it.
            if eligible > 3 {
                let (mut board, mut leave, mut unseen, mut meta, mut leave_value, mut targets, mut squares) = arrays();
                let slot = Slot { board: &mut board, leave: &mut leave, unseen: &mut unseen, meta: &mut meta, leave_value: &mut leave_value, targets: &mut targets, squares: &mut squares };
                let turn = position(&game, 0.5, &gaddag, &leaves, 4, &mut scratch, slot).unwrap();
                let (mut b2, mut l2, mut u2, mut m2, mut v2, mut t2, mut s2) = arrays();
                let slot = Slot { board: &mut b2, leave: &mut l2, unseen: &mut u2, meta: &mut m2, leave_value: &mut v2, targets: &mut t2, squares: &mut s2 };
                assert_eq!(position(&game, 0.5, &gaddag, &leaves, 0, &mut scratch, slot).unwrap(), turn);
                assert_eq!((board, leave, unseen, targets, squares), (b2, l2, u2, t2, s2));
                assert_eq!(meta[..M_BELIEF_INFERRED], m2[..M_BELIEF_INFERRED]);
            }
        }
        assert!(checked > 300, "{checked} positions checked");
    }

    #[test]
    fn a_file_of_games_is_read_back_at_random_and_in_parallel() {
        let (gaddag, leaves) = (gaddag(), LeaveModel::zero());
        let (bytes, lengths, _) = generate(&gaddag, &leaves, 9000, 200, &Opening::default());
        let directory = std::env::temp_dir().join(format!("scrabble-record-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut index = vec![0u64];
        for length in &lengths {
            index.push(index.last().unwrap() + *length as u64);
        }
        std::fs::write(directory.join("games.bin"), &bytes).unwrap();
        std::fs::write(directory.join("index.bin"), index.iter().flat_map(|o| o.to_le_bytes()).collect::<Vec<u8>>()).unwrap();

        let file = GameFile::open(&directory, None).unwrap();
        assert_eq!(file.len(), 200);
        assert_eq!(file.read(57).unwrap().seed, 9057);
        assert!(file.read(200).is_err());

        // A set that records fewer games than its files hold reads only those, also when a
        // kill left a torn entry at the end of the index; one that records more is refused.
        assert_eq!(GameFile::open(&directory, Some(120)).unwrap().len(), 120);
        assert!(GameFile::open(&directory, Some(201)).is_err());
        let whole = std::fs::read(directory.join("index.bin")).unwrap();
        std::fs::write(directory.join("index.bin"), [&whole[..], &[1u8, 2, 3][..]].concat()).unwrap();
        assert!(GameFile::open(&directory, None).is_err());
        let recorded = GameFile::open(&directory, Some(200)).unwrap();
        assert_eq!((recorded.len(), recorded.read(199).unwrap().seed), (200, 9199));
        std::fs::write(directory.join("index.bin"), &whole).unwrap();

        let n = 96;
        let games: Vec<u64> = (0..n as u64).map(|i| (i * 37) % 200).collect();
        let picks: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37) % 1.0).collect();
        let run = |samples: usize| {
            let (mut board, mut leave, mut unseen, mut meta) = (vec![0u8; n * SQUARES], vec![0u8; n * TILE_TYPES], vec![0u8; n * TILE_TYPES], vec![0i16; n * META]);
            let (mut leave_value, mut targets, mut squares, mut turns) = (vec![0f32; n], vec![0f32; n * TARGETS], vec![0u8; n * 2 * SQUARES], vec![0i32; n]);
            file.sample(&gaddag, &leaves, &games, &picks, samples, &mut board, &mut leave, &mut unseen, &mut meta, &mut leave_value, &mut targets, &mut squares, &mut turns).unwrap();
            (board, meta, targets, turns)
        };
        let (board, meta, targets, turns) = run(0);
        // Row by row it is what `position` gives for that game and pick.
        let mut scratch = Scratch::default();
        for i in 0..n {
            let (mut b, mut l, mut u, mut m, mut v, mut t, mut s) = arrays();
            let slot = Slot { board: &mut b, leave: &mut l, unseen: &mut u, meta: &mut m, leave_value: &mut v, targets: &mut t, squares: &mut s };
            let turn = position(&file.read(games[i] as usize).unwrap(), picks[i], &gaddag, &leaves, 0, &mut scratch, slot).unwrap();
            assert_eq!(turns[i] as usize, turn);
            assert_eq!(&board[i * SQUARES..(i + 1) * SQUARES], &b[..]);
            assert_eq!(&meta[i * META..(i + 1) * META], &m[..]);
            assert_eq!(&targets[i * TARGETS..(i + 1) * TARGETS], &t[..]);
        }
        assert_eq!(run(0), (board, meta, targets, turns), "the same request gives the same positions");
        let inferred = run(3);
        assert!(inferred.1.chunks(META).any(|m| m[M_BELIEF_INFERRED] == 1));
        assert_eq!(run(3).1, inferred.1, "and the same inferences, whatever the threads did");

        // A wrong shape or a game that is not there is an error, not a crash.
        let mut short = vec![0u8; SQUARES];
        let (mut l, mut u, mut m, mut v, mut t, mut s, mut tu) = (vec![0u8; n * TILE_TYPES], vec![0u8; n * TILE_TYPES], vec![0i16; n * META], vec![0f32; n], vec![0f32; n * TARGETS], vec![0u8; n * 2 * SQUARES], vec![0i32; n]);
        assert!(file.sample(&gaddag, &leaves, &games, &picks, 0, &mut short, &mut l, &mut u, &mut m, &mut v, &mut t, &mut s, &mut tu).is_err());
        let mut b = vec![0u8; n * SQUARES];
        let beyond = vec![999u64; n];
        assert!(file.sample(&gaddag, &leaves, &beyond, &picks, 0, &mut b, &mut l, &mut u, &mut m, &mut v, &mut t, &mut s, &mut tu).is_err());
        drop(file);
        std::fs::remove_dir_all(&directory).ok();
    }
}
