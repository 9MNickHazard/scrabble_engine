//! `scrabble_ai._core`: the Rust engine exposed to Python.
//!
//! Everything heavy runs with the GIL released. Batch calls read and write NumPy arrays the
//! caller allocated, so nothing is copied or allocated per step.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scrabble_core::belief;
use scrabble_core::board::{Board, LETTER_MULT, SQUARES, WORD_MULT};
use scrabble_core::encode;
use scrabble_core::env::{self, Buffers, Controller};
use scrabble_core::gaddag::Gaddag;
use scrabble_core::game::Game as CoreGame;
use scrabble_core::leave;
use scrabble_core::lexicon::Lexicon;
use scrabble_core::moves::{Move, Placed, Play};
use scrabble_core::policy::{self, Bot, Candidate, Scratch};
use scrabble_core::record;
use scrabble_core::rng::Rng;
use scrabble_core::validate;
use scrabble_core::selfplay::{self, Player};
use scrabble_core::tiles::{Rack, TILE_COUNTS, TILE_TYPES, TILE_VALUES};
use std::path::PathBuf;
use std::sync::Arc;

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn parse_bot(name: &str) -> PyResult<Bot> {
    Bot::parse(name).ok_or_else(|| value_error(format!("unknown bot {name:?}: use random, greedy or greedy-leave")))
}

fn parse_controller(name: &str) -> PyResult<Controller> {
    if name == "external" {
        Ok(Controller::External)
    } else {
        parse_bot(name).map(Controller::Bot)
    }
}

/// A lexicon compiled for move generation.
#[pyclass(frozen)]
struct Engine {
    gaddag: Arc<Gaddag>,
    lexicon_id: String,
    word_count: usize,
}

#[pymethods]
impl Engine {
    /// Load the word list at `path` and its compiled GADDAG, building and caching the GADDAG in
    /// `cache_dir` the first time.
    #[new]
    fn new(py: Python<'_>, name: &str, path: PathBuf, cache_dir: PathBuf) -> PyResult<Engine> {
        py.detach(|| {
            let lexicon = Lexicon::from_file(name, &path)?;
            let gaddag = Gaddag::load_or_build(&lexicon, &cache_dir)?;
            Ok(Engine { gaddag: Arc::new(gaddag), lexicon_id: lexicon.id(), word_count: lexicon.words.len() })
        })
        .map_err(value_error)
    }

    #[getter]
    fn lexicon_id(&self) -> &str {
        &self.lexicon_id
    }

    #[getter]
    fn word_count(&self) -> usize {
        self.word_count
    }

    #[getter]
    fn gaddag_bytes(&self) -> usize {
        self.gaddag.memory_bytes()
    }

    fn is_word(&self, word: &str) -> bool {
        self.gaddag.contains_str(word)
    }

    /// Cross-check masks for `boards` (`uint8[n, 225]` board codes) into `out` (`uint32[n, 450]`).
    fn cross_masks(&self, py: Python<'_>, boards: PyReadonlyArray2<'_, u8>, mut out: PyReadwriteArray2<'_, u32>) -> PyResult<()> {
        let (boards, out) = (boards.as_slice()?, out.as_slice_mut()?);
        if boards.len() % SQUARES != 0 || out.len() != boards.len() / SQUARES * encode::CROSS_WORDS {
            return Err(value_error("expected boards uint8[n, 225] and out uint32[n, 450]".into()));
        }
        let gaddag = &*self.gaddag;
        py.detach(|| encode::cross_masks_batch(gaddag, boards, out));
        Ok(())
    }

    /// Play `count` games by the static evaluator (score + leave value) with seeds from
    /// `first_seed` on, their first plies varied (`record::Opening`), and return them as records:
    /// `(bytes, lengths uint32[count], totals)`. The same arguments always give the same bytes.
    #[pyo3(signature = (leaves, first_seed, count, opening_plies=4, opening_temperature=6.0, opening_top=16))]
    #[allow(clippy::too_many_arguments)]
    fn generate_games<'py>(
        &self,
        py: Python<'py>,
        leaves: &LeaveModel,
        first_seed: u64,
        count: usize,
        opening_plies: u8,
        opening_temperature: f32,
        opening_top: usize,
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, Bound<'py, PyArray1<u32>>, Bound<'py, PyDict>)> {
        let opening = record::Opening { max_plies: opening_plies, temperature: opening_temperature, top: opening_top };
        let (gaddag, model) = (&*self.gaddag, &*leaves.model);
        let (bytes, lengths, totals) = py.detach(|| record::generate(gaddag, model, first_seed, count, &opening));
        let out = PyDict::new(py);
        out.set_item("games", totals.games)?;
        out.set_item("moves", totals.moves)?;
        out.set_item("score_sum", totals.score_sum)?;
        out.set_item("first_player_wins", totals.first_player_wins)?;
        out.set_item("bingos", totals.bingos)?;
        Ok((pyo3::types::PyBytes::new(py, &bytes), lengths.into_pyarray(py), out))
    }

    /// How open each square of each board is to the next player: `boards uint8[n, 225]` and
    /// their cross masks `cross uint32[n, 450]` in, `out uint8[n, 225]` out (see `OPEN_TILES`
    /// and `OPEN_SEVEN`).
    fn open_squares(
        &self,
        py: Python<'_>,
        boards: PyReadonlyArray2<'_, u8>,
        cross: PyReadonlyArray2<'_, u32>,
        mut out: PyReadwriteArray2<'_, u8>,
    ) -> PyResult<()> {
        let (boards, cross, out) = (boards.as_slice()?, cross.as_slice()?, out.as_slice_mut()?);
        if boards.len() % SQUARES != 0 || cross.len() != boards.len() / SQUARES * encode::CROSS_WORDS || out.len() != boards.len() {
            return Err(value_error("expected boards uint8[n, 225], cross uint32[n, 450] and out uint8[n, 225]".into()));
        }
        py.detach(|| encode::open_squares_batch(boards, cross, out));
        Ok(())
    }

    /// Play each seed twice with the two bots swapping seats. Returns arrays with one entry per
    /// seed: `spread_first` / `spread_second` (A's spread when moving first / second),
    /// `score_a`, `score_b` and `turns` (each `[n, 2]`, flattened row-major).
    #[pyo3(signature = (bot_a, leaves_a, bot_b, leaves_b, seeds))]
    fn play_pairs<'py>(
        &self,
        py: Python<'py>,
        bot_a: &str,
        leaves_a: &LeaveModel,
        bot_b: &str,
        leaves_b: &LeaveModel,
        seeds: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = Player { bot: parse_bot(bot_a)?, leaves: &leaves_a.model };
        let b = Player { bot: parse_bot(bot_b)?, leaves: &leaves_b.model };
        let seeds = seeds.as_slice()?;
        let gaddag = &*self.gaddag;
        let results = py.detach(|| selfplay::play_pairs(gaddag, a, b, seeds));
        let out = PyDict::new(py);
        out.set_item("spread_first", results.iter().map(|r| r.spread_first).collect::<Vec<i32>>().into_pyarray(py))?;
        out.set_item("spread_second", results.iter().map(|r| r.spread_second).collect::<Vec<i32>>().into_pyarray(py))?;
        out.set_item("score_a", results.iter().flat_map(|r| r.score_a).collect::<Vec<i32>>().into_pyarray(py))?;
        out.set_item("score_b", results.iter().flat_map(|r| r.score_b).collect::<Vec<i32>>().into_pyarray(py))?;
        out.set_item("turns", results.iter().flat_map(|r| r.turns).collect::<Vec<u16>>().into_pyarray(py))?;
        Ok(out)
    }

    /// Time move generation on positions from one greedy game per seed (single thread). Returns
    /// one tuple per rack kind (no blank, one blank, two blanks):
    /// (positions, median, 99th percentile, slowest, mean, all in microseconds; mean plays found).
    fn bench_movegen(&self, py: Python<'_>, seeds: PyReadonlyArray1<'_, u64>) -> PyResult<Vec<(usize, f64, f64, f64, f64, f64)>> {
        let seeds = seeds.as_slice()?;
        let gaddag = &*self.gaddag;
        let timings = py.detach(|| scrabble_core::bench::movegen_timings(gaddag, seeds));
        Ok(timings.iter().map(|t| (t.positions, t.median_micros, t.p99_micros, t.max_micros, t.mean_micros, t.mean_plays)).collect())
    }

    /// Fit leave values from self-play of `bot` (using `leaves`) over `seeds`. Returns the new
    /// model and, per stage of the game, (samples, target mean, target standard deviation).
    fn fit_leaves(
        &self,
        py: Python<'_>,
        bot: &str,
        leaves: &LeaveModel,
        seeds: PyReadonlyArray1<'_, u64>,
        ridge: f64,
    ) -> PyResult<(LeaveModel, Vec<(u64, f64, f64)>)> {
        let bot = parse_bot(bot)?;
        let seeds = seeds.as_slice()?;
        let (gaddag, current) = (&*self.gaddag, &leaves.model);
        let (model, report) = py.detach(|| selfplay::fit_leaves(gaddag, bot, current, seeds, ridge));
        let report = report.iter().map(|r| (r.samples, r.target_mean, r.target_sd)).collect();
        Ok((LeaveModel { model: Arc::new(model) }, report))
    }
}

/// Leave values (see `scrabble_core::leave`).
#[pyclass(frozen)]
struct LeaveModel {
    model: Arc<leave::LeaveModel>,
}

#[pymethods]
impl LeaveModel {
    /// The model that values every leave at 0.
    #[staticmethod]
    fn zero() -> LeaveModel {
        LeaveModel { model: Arc::new(leave::LeaveModel::zero()) }
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<LeaveModel> {
        let bytes = std::fs::read(&path).map_err(|e| value_error(format!("{}: {e}", path.display())))?;
        let model = leave::LeaveModel::from_bytes(&bytes).map_err(|e| value_error(format!("{}: {e}", path.display())))?;
        Ok(LeaveModel { model: Arc::new(model) })
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        std::fs::write(&path, self.model.to_bytes()).map_err(|e| value_error(format!("{}: {e}", path.display())))
    }

    /// Value of keeping `rack` (letters, `?` for a blank) with `bag_after_draw` tiles left in the
    /// bag once the mover has drawn.
    fn value(&self, rack: &str, bag_after_draw: usize) -> PyResult<f32> {
        let rack = Rack::parse(rack).ok_or_else(|| value_error(format!("bad rack {rack:?}")))?;
        Ok(self.model.value(&rack, bag_after_draw))
    }

    /// The raw weights, `float32[stages * features]`, for hashing into a checkpoint.
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.model.weights.iter().flatten().copied().collect::<Vec<f32>>().into_pyarray(py)
    }
}

/// Many games stepped together (see `scrabble_core::env`).
#[pyclass]
struct BatchEnv {
    env: env::BatchEnv,
    gaddag: Arc<Gaddag>,
    leaves: Arc<leave::LeaveModel>,
}

#[pymethods]
impl BatchEnv {
    /// `belief_samples`: racks sampled per placement to infer what the mover kept (0: no
    /// inference; the afterstates then say nothing is known about the opponent's rack).
    #[new]
    #[pyo3(signature = (engine, leaves, slots, max_candidates, max_exchanges, belief_samples=0))]
    fn new(
        engine: &Engine,
        leaves: &LeaveModel,
        slots: usize,
        max_candidates: usize,
        max_exchanges: usize,
        belief_samples: usize,
    ) -> PyResult<BatchEnv> {
        if max_candidates < 2 || max_exchanges >= max_candidates {
            return Err(value_error("need max_candidates >= 2 and max_exchanges < max_candidates".into()));
        }
        Ok(BatchEnv {
            env: env::BatchEnv::new(slots, max_candidates, max_exchanges).with_belief_samples(belief_samples),
            gaddag: engine.gaddag.clone(),
            leaves: leaves.model.clone(),
        })
    }

    #[getter]
    fn slots(&self) -> usize {
        self.env.len()
    }

    #[getter]
    fn max_candidates(&self) -> usize {
        self.env.max_candidates()
    }

    #[getter]
    fn belief_samples(&self) -> usize {
        self.env.belief_samples()
    }

    /// Start a new game in `slot`. Controllers are "external", "random", "greedy" or
    /// "greedy-leave", for the seat that moves first and the seat that moves second.
    fn reset(&mut self, slot: usize, seed: u64, first: &str, second: &str) -> PyResult<()> {
        self.check_slot(slot)?;
        self.env.reset(slot, seed, [parse_controller(first)?, parse_controller(second)?]);
        Ok(())
    }

    /// Empty `slot` until it is reset.
    fn clear(&mut self, slot: usize) -> PyResult<()> {
        self.check_slot(slot)?;
        self.env.clear(slot);
        Ok(())
    }

    /// Advance every game to its next external decision and write the candidates. Arrays, for
    /// `n` slots and `k` candidates: `board uint8[n, k, 225]`, `leave uint8[n, k, 27]`,
    /// `unseen uint8[n, k, 27]`, `meta int16[n, k, META]`, `leave_value float32[n, k]`,
    /// `count int32[n]`, `mover int8[n]`, `status uint8[n]`, `scores int32[n, 2]`.
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &mut self,
        py: Python<'_>,
        mut board: PyReadwriteArray3<'_, u8>,
        mut leave: PyReadwriteArray3<'_, u8>,
        mut unseen: PyReadwriteArray3<'_, u8>,
        mut meta: PyReadwriteArray3<'_, i16>,
        mut leave_value: PyReadwriteArray2<'_, f32>,
        mut count: PyReadwriteArray1<'_, i32>,
        mut mover: PyReadwriteArray1<'_, i8>,
        mut status: PyReadwriteArray1<'_, u8>,
        mut scores: PyReadwriteArray2<'_, i32>,
    ) -> PyResult<()> {
        let (n, k) = (self.env.len(), self.env.max_candidates());
        let buffers = Buffers {
            board: board.as_slice_mut()?,
            leave: leave.as_slice_mut()?,
            unseen: unseen.as_slice_mut()?,
            meta: meta.as_slice_mut()?,
            leave_value: leave_value.as_slice_mut()?,
            count: count.as_slice_mut()?,
            mover: mover.as_slice_mut()?,
            status: status.as_slice_mut()?,
            scores: scores.as_slice_mut()?,
        };
        let sizes_ok = buffers.board.len() == n * k * SQUARES
            && buffers.leave.len() == n * k * TILE_TYPES
            && buffers.unseen.len() == n * k * TILE_TYPES
            && buffers.meta.len() == n * k * encode::META
            && buffers.leave_value.len() == n * k
            && buffers.count.len() == n
            && buffers.mover.len() == n
            && buffers.status.len() == n
            && buffers.scores.len() == 2 * n;
        if !sizes_ok {
            return Err(value_error(format!("array shapes do not match {n} slots x {k} candidates")));
        }
        let (env, gaddag, leaves) = (&mut self.env, &*self.gaddag, &*self.leaves);
        py.detach(|| env.prepare(gaddag, leaves, buffers));
        Ok(())
    }

    /// Play `actions[slot]`, an index into that slot's candidates, wherever a decision is pending.
    fn apply(&mut self, py: Python<'_>, actions: PyReadonlyArray1<'_, i32>) -> PyResult<()> {
        let actions = actions.as_slice()?;
        if actions.len() != self.env.len() {
            return Err(value_error("one action per slot is required".into()));
        }
        let (env, gaddag, leaves) = (&mut self.env, &*self.gaddag, &*self.leaves);
        py.detach(|| env.apply(gaddag, leaves, actions)).map_err(value_error)
    }

    /// Text of candidate `index` of the pending decision in `slot`, e.g. "8D QUIZ 48".
    fn describe(&self, slot: usize, index: usize) -> PyResult<String> {
        self.check_slot(slot)?;
        self.env
            .offered(slot, index)
            .map(|candidate| candidate.mv.to_string())
            .ok_or_else(|| PyIndexError::new_err("no such candidate"))
    }

    /// The board in `slot` as 15 lines of text.
    fn board_text(&self, slot: usize) -> PyResult<String> {
        self.check_slot(slot)?;
        Ok(self.env.game(slot).board.to_text())
    }

    /// Racks in `slot` by seat, as text. For tests and debugging: the network never sees this.
    fn racks(&self, slot: usize) -> PyResult<(String, String)> {
        self.check_slot(slot)?;
        let game = self.env.game(slot);
        Ok((game.racks[0].to_string(), game.racks[1].to_string()))
    }

    /// Turns played so far in `slot`.
    fn turn(&self, slot: usize) -> PyResult<u16> {
        self.check_slot(slot)?;
        Ok(self.env.game(slot).turn)
    }

    /// What each seat did in the game in `slot` so far: lists indexed by seat for `bingos`,
    /// `exchanges`, `passes`, `placements`, `best_score` and `best_word`, plus `ended_by`
    /// ("out", "scoreless", or "" while the game is running).
    fn game_stats<'py>(&self, py: Python<'py>, slot: usize) -> PyResult<Bound<'py, PyDict>> {
        self.check_slot(slot)?;
        let (stats, game) = (self.env.stats(slot), self.env.game(slot));
        let out = PyDict::new(py);
        let per_seat = |f: fn(&env::SeatStats) -> i64| stats.seats.iter().map(f).collect::<Vec<i64>>();
        out.set_item("bingos", per_seat(|s| s.bingos as i64))?;
        out.set_item("exchanges", per_seat(|s| s.exchanges as i64))?;
        out.set_item("passes", per_seat(|s| s.passes as i64))?;
        out.set_item("placements", per_seat(|s| s.placements as i64))?;
        out.set_item("best_score", per_seat(|s| s.best_score as i64))?;
        let words: Vec<String> = stats.seats.iter().map(|s| s.best_play.map_or(String::new(), |p| p.word_plain())).collect();
        out.set_item("best_word", words)?;
        out.set_item("ended_by", ended_by(game))?;
        Ok(out)
    }
}

impl BatchEnv {
    fn check_slot(&self, slot: usize) -> PyResult<()> {
        if slot < self.env.len() {
            Ok(())
        } else {
            Err(PyIndexError::new_err(format!("slot {slot} out of range ({} slots)", self.env.len())))
        }
    }
}

// ---- Interactive games -------------------------------------------------------------------

/// (row, col, letter, is_blank) for each newly placed tile of a play.
fn placed_tiles(play: &Play) -> Vec<(usize, usize, String, bool)> {
    play.placed()
        .map(|t| (t.index / 15, t.index % 15, ((b'A' + t.letter) as char).to_string(), t.blank))
        .collect()
}

/// The parts of a move that do not depend on the game: kind, text, word, position, tiles.
fn describe_move<'py>(py: Python<'py>, mv: &Move) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let no_tiles = Vec::<(usize, usize, String, bool)>::new();
    match mv {
        Move::Play(play) => {
            out.set_item("kind", "play")?;
            out.set_item("text", format!("{} {}", play.position_text(), play.word_text()))?;
            out.set_item("word", play.word_plain())?;
            out.set_item("position", play.position_text())?;
            out.set_item("tiles", placed_tiles(play))?;
            out.set_item("bingo", play.tiles_played == 7)?;
            out.set_item("exchanged", "")?;
        }
        Move::Exchange(tiles) => {
            out.set_item("kind", "exchange")?;
            out.set_item("text", format!("exchange {}", tiles.len()))?;
            out.set_item("word", "")?;
            out.set_item("position", "")?;
            out.set_item("tiles", no_tiles)?;
            out.set_item("bingo", false)?;
            out.set_item("exchanged", tiles.to_string())?;
        }
        Move::Pass => {
            out.set_item("kind", "pass")?;
            out.set_item("text", "pass")?;
            out.set_item("word", "")?;
            out.set_item("position", "")?;
            out.set_item("tiles", no_tiles)?;
            out.set_item("bingo", false)?;
            out.set_item("exchanged", "")?;
        }
    }
    Ok(out)
}

fn ended_by(game: &CoreGame) -> &'static str {
    if !game.over {
        ""
    } else if game.scoreless_turns >= scrabble_core::game::SCORELESS_LIMIT {
        "scoreless"
    } else {
        "out"
    }
}

/// One game played move by move, for the board on screen: a person against a model, or two
/// models being watched. `candidates` offers the moves a player would consider (with the same
/// afterstate arrays the batched environment writes); `play_tiles` checks and plays a move
/// entered by a person.
#[pyclass]
struct Game {
    game: CoreGame,
    gaddag: Arc<Gaddag>,
    leaves: Arc<leave::LeaveModel>,
    offered: Vec<Candidate>,
    seed: u64,
    /// The baseline bots' random stream, seeded as in matches so a match game replays here.
    bot_rng: Rng,
    /// Racks sampled per placement to infer what the mover kept, and the stream they come from
    /// (seeded as in matches, like the bots' stream).
    belief_samples: usize,
    belief_rng: Rng,
}

/// A candidate as Python sees it: the move, plus `score` (with the going-out bonus), `bonus`
/// (that bonus alone), `leave`, `leave_value` and `terminal`.
fn describe_candidate<'py>(py: Python<'py>, candidate: &Candidate) -> PyResult<Bound<'py, PyDict>> {
    let description = describe_move(py, &candidate.mv)?;
    description.set_item("score", candidate.score)?;
    description.set_item("bonus", candidate.score - candidate.mv.score())?;
    description.set_item("leave", candidate.leave.to_string())?;
    description.set_item("leave_value", candidate.leave_value)?;
    description.set_item("terminal", candidate.terminal)?;
    Ok(description)
}

fn unseen_dict<'py>(py: Python<'py>, game: &CoreGame, seat: usize) -> PyResult<Bound<'py, PyDict>> {
    let unseen = PyDict::new(py);
    for (tile, &count) in game.unseen(seat).iter().enumerate() {
        if count > 0 {
            unseen.set_item(scrabble_core::tiles::tile_to_char(tile as u8).to_string(), count)?;
        }
    }
    Ok(unseen)
}

impl Game {
    fn apply<'py>(&mut self, py: Python<'py>, mv: Move) -> PyResult<Bound<'py, PyDict>> {
        let mover = self.game.to_move;
        let rack_before = self.game.racks[mover];
        let mut leave = rack_before;
        leave.remove_all(&mv.tiles_used());
        let out = describe_move(py, &mv)?;
        let mut scratch = Scratch::default();
        let applied = belief::apply_observed(
            &mut self.game,
            &mv,
            &self.gaddag,
            &self.leaves,
            self.belief_samples,
            &mut self.belief_rng,
            &mut scratch,
        );
        self.offered.clear();
        out.set_item("player", mover)?;
        out.set_item("score", applied.score)?;
        out.set_item("rack", rack_before.to_string())?;
        out.set_item("leave", leave.to_string())?;
        out.set_item("adjustment", applied.adjustment.to_vec())?;
        out.set_item("game_over", applied.game_over)?;
        Ok(out)
    }
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (engine, leaves, seed, belief_samples=0))]
    fn new(engine: &Engine, leaves: &LeaveModel, seed: u64, belief_samples: usize) -> Game {
        Game {
            game: CoreGame::new(seed),
            gaddag: engine.gaddag.clone(),
            leaves: leaves.model.clone(),
            offered: Vec::new(),
            seed,
            bot_rng: Rng::new(seed ^ 0xB07_5EED),
            belief_samples,
            belief_rng: Rng::new(seed ^ 0xBE11_EF5),
        }
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The position: `board` (15 strings), `racks`, `scores`, `to_move`, `turn`, `bag`,
    /// `scoreless_turns`, `can_exchange`, `over`, `ended_by`, `unseen` (tiles the side to
    /// move cannot see, as letter -> count) and `unseen_by_seat` (the same for each seat).
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let game = &self.game;
        let out = PyDict::new(py);
        out.set_item("board", game.board.to_text().lines().map(str::to_string).collect::<Vec<_>>())?;
        out.set_item("racks", vec![game.racks[0].to_string(), game.racks[1].to_string()])?;
        out.set_item("scores", game.scores.to_vec())?;
        out.set_item("to_move", game.to_move)?;
        out.set_item("turn", game.turn)?;
        out.set_item("bag", game.bag.len())?;
        out.set_item("scoreless_turns", game.scoreless_turns)?;
        out.set_item("can_exchange", game.can_exchange())?;
        out.set_item("over", game.over)?;
        out.set_item("ended_by", ended_by(game))?;
        out.set_item("unseen", unseen_dict(py, game, game.to_move)?)?;
        out.set_item("unseen_by_seat", vec![unseen_dict(py, game, 0)?, unseen_dict(py, game, 1)?])?;
        Ok(out)
    }

    /// The moves the side to move would consider, best static value first, as
    /// (descriptions, board uint8[k, 225], leave uint8[k, 27], unseen uint8[k, 27],
    /// meta int16[k, META], leave_value float32[k]). `play_candidate(i)` plays one of them.
    #[allow(clippy::type_complexity)]
    fn candidates<'py>(
        &mut self,
        py: Python<'py>,
        max_candidates: usize,
        max_exchanges: usize,
    ) -> PyResult<(
        Vec<Bound<'py, PyDict>>,
        Bound<'py, numpy::PyArray2<u8>>,
        Bound<'py, numpy::PyArray2<u8>>,
        Bound<'py, numpy::PyArray2<u8>>,
        Bound<'py, numpy::PyArray2<i16>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        if max_candidates < 2 || max_exchanges >= max_candidates {
            return Err(value_error("need max_candidates >= 2 and max_exchanges < max_candidates".into()));
        }
        self.offered.clear();
        if !self.game.over {
            let mut scratch = Scratch::default();
            env::select_candidates(
                &self.game,
                &self.gaddag,
                &self.leaves,
                &mut scratch,
                max_candidates,
                max_exchanges,
                &mut self.offered,
            );
        }
        let k = self.offered.len();
        let mut board = vec![0u8; k * SQUARES];
        let mut leave = vec![0u8; k * TILE_TYPES];
        let mut unseen = vec![0u8; k * TILE_TYPES];
        let mut meta = vec![0i16; k * encode::META];
        let mut leave_value = vec![0f32; k];
        let base = encode::base_codes(&self.game);
        let pool = self.game.unseen(self.game.to_move);
        let mut descriptions = Vec::with_capacity(k);
        for (i, candidate) in self.offered.iter().enumerate() {
            encode::encode_afterstate(
                &self.game,
                &base,
                &pool,
                candidate,
                &mut board[i * SQUARES..(i + 1) * SQUARES],
                &mut leave[i * TILE_TYPES..(i + 1) * TILE_TYPES],
                &mut unseen[i * TILE_TYPES..(i + 1) * TILE_TYPES],
                &mut meta[i * encode::META..(i + 1) * encode::META],
            );
            leave_value[i] = candidate.leave_value;
            descriptions.push(describe_candidate(py, candidate)?);
        }
        let bytes = |data: Vec<u8>, width: usize| numpy::ndarray::Array2::from_shape_vec((k, width), data).expect("shape");
        Ok((
            descriptions,
            bytes(board, SQUARES).into_pyarray(py),
            bytes(leave, TILE_TYPES).into_pyarray(py),
            bytes(unseen, TILE_TYPES).into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((k, encode::META), meta).expect("shape").into_pyarray(py),
            leave_value.into_pyarray(py),
        ))
    }

    /// The move a baseline bot (`random`, `greedy`, `greedy-leave`) makes for the side to
    /// move: the same code as in matches and evaluations, and for `random` the same random
    /// stream for a given game seed, so a match game replays here move for move. Returns its
    /// index among the offered candidates (call `candidates` first) and, when the move was not
    /// among them and has been added at the end, its description. Each call for `random`
    /// advances the stream, so call it once per move actually played.
    fn bot_choice<'py>(&mut self, py: Python<'py>, bot: &str) -> PyResult<(usize, Option<Bound<'py, PyDict>>)> {
        let bot = parse_bot(bot)?;
        if self.game.over {
            return Err(value_error("the game is over".into()));
        }
        let mut scratch = Scratch::default();
        let mv = policy::choose(bot, &self.game, &self.gaddag, &self.leaves, &mut self.bot_rng, &mut scratch);
        if let Some(index) = self.offered.iter().position(|candidate| candidate.mv == mv) {
            return Ok((index, None));
        }
        let candidate = policy::candidate(&self.game, &self.leaves, mv);
        let description = describe_candidate(py, &candidate)?;
        self.offered.push(candidate);
        Ok((self.offered.len() - 1, Some(description)))
    }

    /// Play candidate `index` from the last `candidates` call. Returns the move as a dict.
    fn play_candidate<'py>(&mut self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
        let Some(candidate) = self.offered.get(index) else {
            return Err(PyIndexError::new_err("no such candidate"));
        };
        let mv = candidate.mv;
        self.apply(py, mv)
    }

    /// Check and play a placement entered by a person: `tiles` are (row, col, letter, is_blank)
    /// for the newly placed tiles only. Raises ValueError with the reason if it is not legal.
    fn play_tiles<'py>(&mut self, py: Python<'py>, tiles: Vec<(usize, usize, String, bool)>) -> PyResult<Bound<'py, PyDict>> {
        if self.game.over {
            return Err(value_error("the game is over".into()));
        }
        let mut placed = Vec::with_capacity(tiles.len());
        for (row, col, letter, blank) in &tiles {
            let letter = letter
                .chars()
                .next()
                .filter(char::is_ascii_alphabetic)
                .ok_or_else(|| value_error("a tile needs a letter A-Z".into()))?;
            if *row >= 15 || *col >= 15 {
                return Err(value_error("a tile is off the board".into()));
            }
            placed.push(Placed { index: row * 15 + col, letter: letter.to_ascii_uppercase() as u8 - b'A', blank: *blank });
        }
        let play = validate::check_placement(&self.gaddag, &self.game.board, &placed).map_err(value_error)?;
        let mv = Move::Play(play);
        validate::check_move(&self.gaddag, &self.game, &mv).map_err(value_error)?;
        self.apply(py, mv)
    }

    /// Exchange the given tiles (letters, `?` for a blank).
    fn exchange<'py>(&mut self, py: Python<'py>, tiles: &str) -> PyResult<Bound<'py, PyDict>> {
        let rack = Rack::parse(tiles).ok_or_else(|| value_error(format!("bad tiles {tiles:?}")))?;
        let mv = Move::Exchange(rack);
        validate::check_move(&self.gaddag, &self.game, &mv).map_err(value_error)?;
        self.apply(py, mv)
    }

    fn pass_turn<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        validate::check_move(&self.gaddag, &self.game, &Move::Pass).map_err(value_error)?;
        self.apply(py, Move::Pass)
    }
}

/// A folder of saved games (`games.bin` and `index.bin`, see `scrabble_core::record`) opened for
/// reading training positions at random.
#[pyclass(frozen)]
struct GameSet {
    file: record::GameFile,
}

#[pymethods]
impl GameSet {
    /// `games`: read only the first that many games, the number the set's `meta.json` records
    /// (the files may hold a block more, or a torn one, after a kill). Without it, every game
    /// the files hold.
    #[new]
    #[pyo3(signature = (directory, games=None))]
    fn new(directory: PathBuf, games: Option<usize>) -> PyResult<GameSet> {
        Ok(GameSet { file: record::GameFile::open(&directory, games).map_err(value_error)? })
    }

    fn __len__(&self) -> usize {
        self.file.len()
    }

    /// Game `index` as `(seed, [score0, score1], went_out, [move text, ...])`.
    fn game(&self, index: usize) -> PyResult<(u64, Vec<i32>, bool, Vec<String>)> {
        let game = self.file.read(index).map_err(value_error)?;
        Ok((game.seed, game.scores.to_vec(), game.went_out, game.moves.iter().map(|m| m.to_string()).collect()))
    }

    /// One position from each of `games` (indices into the set), the turn chosen by `picks`
    /// (each in [0, 1), spread evenly over the game's turns). Arrays for `n` positions:
    /// `board uint8[n, 225]`, `leave uint8[n, 27]`, `unseen uint8[n, 27]`, `meta int16[n, META]`,
    /// `leave_value float32[n]`, `targets float32[n, TARGETS]`, `squares uint8[n, 450]` (where
    /// the opponent's next move, then the mover's own next move, placed tiles) and
    /// `turns int32[n]`. `belief_samples` > 0 also infers what the mover could tell about the
    /// opponent's rack, as during play.
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        engine: &Engine,
        leaves: &LeaveModel,
        games: PyReadonlyArray1<'_, u64>,
        picks: PyReadonlyArray1<'_, f32>,
        belief_samples: usize,
        mut board: PyReadwriteArray2<'_, u8>,
        mut leave: PyReadwriteArray2<'_, u8>,
        mut unseen: PyReadwriteArray2<'_, u8>,
        mut meta: PyReadwriteArray2<'_, i16>,
        mut leave_value: PyReadwriteArray1<'_, f32>,
        mut targets: PyReadwriteArray2<'_, f32>,
        mut squares: PyReadwriteArray2<'_, u8>,
        mut turns: PyReadwriteArray1<'_, i32>,
    ) -> PyResult<()> {
        let (games, picks) = (games.as_slice()?, picks.as_slice()?);
        let (board, leave, unseen, meta) = (board.as_slice_mut()?, leave.as_slice_mut()?, unseen.as_slice_mut()?, meta.as_slice_mut()?);
        let (leave_value, targets, squares, turns) =
            (leave_value.as_slice_mut()?, targets.as_slice_mut()?, squares.as_slice_mut()?, turns.as_slice_mut()?);
        let (file, gaddag, model) = (&self.file, &*engine.gaddag, &*leaves.model);
        py.detach(|| {
            file.sample(gaddag, model, games, picks, belief_samples, board, leave, unseen, meta, leave_value, targets, squares, turns)
        })
        .map_err(value_error)
    }
}

/// Parse a 15-line board and return its codes (`uint8[225]`), for tests and analysis.
#[pyfunction]
fn board_codes_from_text<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let board = Board::from_text(text).map_err(value_error)?;
    Ok(encode::board_codes(&board).to_vec().into_pyarray(py))
}

#[pymodule(name = "_core")]
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<LeaveModel>()?;
    m.add_class::<BatchEnv>()?;
    m.add_class::<Game>()?;
    m.add_class::<GameSet>()?;
    m.add("TARGETS", record::TARGETS)?;
    m.add("T_WIN", record::T_WIN)?;
    m.add("T_SPREAD", record::T_SPREAD)?;
    m.add("T_OPP_SCORE", record::T_OPP_SCORE)?;
    m.add("T_OPP_BINGO", record::T_OPP_BINGO)?;
    m.add("T_HAS_OPP", record::T_HAS_OPP)?;
    m.add("T_OWN_SCORE", record::T_OWN_SCORE)?;
    m.add("T_OWN_BINGO", record::T_OWN_BINGO)?;
    m.add("T_HAS_OWN", record::T_HAS_OWN)?;
    m.add_function(wrap_pyfunction!(board_codes_from_text, m)?)?;

    m.add("SQUARES", SQUARES)?;
    m.add("TILE_TYPES", TILE_TYPES)?;
    // As lists of ints: a Vec<u8> would arrive in Python as `bytes`.
    let ints = |values: &[u8]| values.iter().map(|&v| v as u32).collect::<Vec<u32>>();
    m.add("TILE_COUNTS", ints(&TILE_COUNTS))?;
    m.add("TILE_VALUES", ints(&TILE_VALUES))?;
    m.add("LETTER_MULT", ints(&LETTER_MULT))?;
    m.add("WORD_MULT", ints(&WORD_MULT))?;

    m.add("CODE_LETTER", encode::CODE_LETTER)?;
    m.add("CODE_BLANK", encode::CODE_BLANK)?;
    m.add("CODE_PLACED", encode::CODE_PLACED)?;
    m.add("CODE_OPPONENT", encode::CODE_OPPONENT)?;
    m.add("OPEN_TILES", encode::OPEN_TILES)?;
    m.add("OPEN_SEVEN", encode::OPEN_SEVEN)?;
    m.add("BELIEF_SCALE", belief::SCALE)?;
    m.add("CROSS_HAS_WORD", encode::CROSS_HAS_WORD)?;
    m.add("CROSS_WORDS", encode::CROSS_WORDS)?;
    m.add("META", encode::META)?;
    m.add("M_SCORE", encode::M_SCORE)?;
    m.add("M_SPREAD", encode::M_SPREAD)?;
    m.add("M_BAG", encode::M_BAG)?;
    m.add("M_DRAWS", encode::M_DRAWS)?;
    m.add("M_SCORELESS", encode::M_SCORELESS)?;
    m.add("M_OPPONENT_TILES", encode::M_OPPONENT_TILES)?;
    m.add("M_KIND", encode::M_KIND)?;
    m.add("M_TERMINAL", encode::M_TERMINAL)?;
    m.add("M_OPPONENT_LAST_KIND", encode::M_OPPONENT_LAST_KIND)?;
    m.add("M_OPPONENT_LAST_TILES", encode::M_OPPONENT_LAST_TILES)?;
    m.add("M_OPPONENT_LAST_SCORE", encode::M_OPPONENT_LAST_SCORE)?;
    m.add("M_OPPONENT_SINCE_BINGO", encode::M_OPPONENT_SINCE_BINGO)?;
    m.add("M_BELIEF_INFERRED", encode::M_BELIEF_INFERRED)?;
    m.add("M_BELIEF", encode::M_BELIEF)?;

    m.add("STATUS_EMPTY", env::STATUS_EMPTY)?;
    m.add("STATUS_DECISION", env::STATUS_DECISION)?;
    m.add("STATUS_FINISHED", env::STATUS_FINISHED)?;
    m.add("LEAVE_FEATURES", leave::FEATURES)?;
    Ok(())
}
