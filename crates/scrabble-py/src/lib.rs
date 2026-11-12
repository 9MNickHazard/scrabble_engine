//! `scrabble_ai._core`: the Rust engine exposed to Python.
//!
//! Everything heavy runs with the GIL released. Batch calls read and write NumPy arrays the
//! caller allocated, so nothing is copied or allocated per step.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scrabble_core::board::{Board, LETTER_MULT, SQUARES, WORD_MULT};
use scrabble_core::encode;
use scrabble_core::env::{self, Buffers, Controller};
use scrabble_core::gaddag::Gaddag;
use scrabble_core::leave;
use scrabble_core::lexicon::Lexicon;
use scrabble_core::policy::Bot;
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
    #[new]
    fn new(engine: &Engine, leaves: &LeaveModel, slots: usize, max_candidates: usize, max_exchanges: usize) -> PyResult<BatchEnv> {
        if max_candidates < 2 || max_exchanges >= max_candidates {
            return Err(value_error("need max_candidates >= 2 and max_exchanges < max_candidates".into()));
        }
        Ok(BatchEnv {
            env: env::BatchEnv::new(slots, max_candidates, max_exchanges),
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
    fn apply(&mut self, actions: PyReadonlyArray1<'_, i32>) -> PyResult<()> {
        let actions = actions.as_slice()?;
        if actions.len() != self.env.len() {
            return Err(value_error("one action per slot is required".into()));
        }
        self.env.apply(actions).map_err(value_error)
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

    m.add("STATUS_EMPTY", env::STATUS_EMPTY)?;
    m.add("STATUS_DECISION", env::STATUS_DECISION)?;
    m.add("STATUS_FINISHED", env::STATUS_FINISHED)?;
    m.add("LEAVE_FEATURES", leave::FEATURES)?;
    Ok(())
}
