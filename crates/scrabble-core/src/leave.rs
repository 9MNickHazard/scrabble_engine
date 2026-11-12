//! Leave values: how many points of future spread the tiles kept on the rack are worth.
//!
//! The model is linear in simple indicators of the leave, fitted by ridge regression on
//! self-play: the target for each turn is the spread the mover went on to gain over the rest of
//! the game (the same quantity the value network learns, restricted to what the leave says).
//! Three separate sets of weights cover the three stages of the game, because a tile's worth
//! changes completely once the bag runs out.
//!
//! `score + leave value` is the static evaluation used by the greedy+leave baseline, to rank
//! candidates before the network sees them, and as the baseline the network adds to.

use crate::tiles::{Rack, BLANK, TILE_TYPES};

const COUNT_LEVELS: usize = 4;
const COUNT_FEATURES: usize = TILE_TYPES * COUNT_LEVELS;
const BALANCE_SIDE: usize = 8;
const BALANCE_FEATURES: usize = BALANCE_SIDE * BALANCE_SIDE;
const PAIR_FEATURES: usize = TILE_TYPES * (TILE_TYPES - 1) / 2;
const BUCKETS: usize = 5;

/// Number of indicator features.
pub const FEATURES: usize = COUNT_FEATURES + BALANCE_FEATURES + PAIR_FEATURES + BUCKETS;
/// Most features that can be active for one leave.
pub const MAX_ACTIVE: usize = 7 + 1 + 21 + 1;
pub const STAGES: usize = 3;

const VOWELS: u32 = 1 | 1 << 4 | 1 << 8 | 1 << 14 | 1 << 20;

/// Stage of the game after the mover has drawn: 0 while the bag still has 7 or more tiles,
/// 1 with 1-6 left, 2 once it is empty.
#[inline]
pub fn stage(bag_after_draw: usize) -> usize {
    match bag_after_draw {
        0 => 2,
        1..=6 => 1,
        _ => 0,
    }
}

/// The active features of a leave. Returns how many were written to `out`.
///
/// - one feature per copy of each tile ("first S", "second S", ...), capped at four copies;
/// - one for the leave's (vowels, consonants) balance, blanks counted as neither;
/// - one for each pair of different tile types held together (Q with U, and so on);
/// - one for how full the bag is, so the fit is not confused by the stage of the game.
pub fn features(leave: &Rack, bag_after_draw: usize, out: &mut [u16; MAX_ACTIVE]) -> usize {
    let mut n = 0;
    let (mut vowels, mut consonants) = (0usize, 0usize);
    let mut present = [0u8; 7];
    let mut kinds = 0;
    for tile in 0..TILE_TYPES {
        let count = leave.counts[tile] as usize;
        if count == 0 {
            continue;
        }
        for level in 0..count.min(COUNT_LEVELS) {
            out[n] = (tile * COUNT_LEVELS + level) as u16;
            n += 1;
        }
        if tile != BLANK as usize {
            if VOWELS >> tile & 1 == 1 {
                vowels += count;
            } else {
                consonants += count;
            }
        }
        present[kinds] = tile as u8;
        kinds += 1;
    }
    out[n] = (COUNT_FEATURES + vowels.min(BALANCE_SIDE - 1) * BALANCE_SIDE + consonants.min(BALANCE_SIDE - 1)) as u16;
    n += 1;
    for i in 0..kinds {
        for j in i + 1..kinds {
            let (a, b) = (present[i] as usize, present[j] as usize);
            // Index of the pair (a, b), a < b, in row-major order over the upper triangle.
            let pair = a * (2 * TILE_TYPES - a - 1) / 2 + (b - a - 1);
            out[n] = (COUNT_FEATURES + BALANCE_FEATURES + pair) as u16;
            n += 1;
        }
    }
    let bucket = match bag_after_draw {
        0..=13 => 0,
        14..=27 => 1,
        28..=48 => 2,
        49..=69 => 3,
        _ => 4,
    };
    out[n] = (COUNT_FEATURES + BALANCE_FEATURES + PAIR_FEATURES + bucket) as u16;
    n + 1
}

#[derive(Clone, PartialEq, Debug)]
pub struct LeaveModel {
    /// `weights[stage][feature]`, in points of spread.
    pub weights: [Vec<f32>; STAGES],
}

impl Default for LeaveModel {
    fn default() -> Self {
        LeaveModel::zero()
    }
}

impl LeaveModel {
    /// The model that values every leave at 0: with it, greedy+leave is plain greedy.
    pub fn zero() -> LeaveModel {
        LeaveModel { weights: std::array::from_fn(|_| vec![0.0; FEATURES]) }
    }

    /// Value of keeping `leave` when the bag will hold `bag_after_draw` tiles once the mover has
    /// drawn.
    pub fn value(&self, leave: &Rack, bag_after_draw: usize) -> f32 {
        let mut active = [0u16; MAX_ACTIVE];
        let n = features(leave, bag_after_draw, &mut active);
        let weights = &self.weights[stage(bag_after_draw)];
        active[..n].iter().map(|&f| weights[f as usize]).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + STAGES * FEATURES * 4);
        out.extend(b"SLV1");
        out.extend((STAGES as u32).to_le_bytes());
        out.extend((FEATURES as u32).to_le_bytes());
        for stage in &self.weights {
            for weight in stage {
                out.extend(weight.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<LeaveModel, String> {
        let expected = 12 + STAGES * FEATURES * 4;
        if bytes.len() != expected || &bytes[..4] != b"SLV1" {
            return Err("not a leave-value file for this version".to_string());
        }
        let field = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        if field(4) != STAGES || field(8) != FEATURES {
            return Err("leave-value file has a different feature layout".to_string());
        }
        let mut values = bytes[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()));
        let weights = std::array::from_fn(|_| values.by_ref().take(FEATURES).collect());
        Ok(LeaveModel { weights })
    }
}

/// Sums for the ridge regression of one stage: co-occurrence counts of the indicator features
/// and their sums against the target.
#[derive(Clone)]
pub struct Accumulator {
    /// `gram[i * FEATURES + j]` for `i <= j`: samples in which features i and j are both active.
    gram: Vec<f64>,
    target: Vec<f64>,
    pub samples: u64,
    pub target_sum: f64,
    pub target_squares: f64,
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            gram: vec![0.0; FEATURES * FEATURES],
            target: vec![0.0; FEATURES],
            samples: 0,
            target_sum: 0.0,
            target_squares: 0.0,
        }
    }
}

impl Accumulator {
    /// Add one sample: the (sorted or unsorted) active features and the observed target.
    pub fn add(&mut self, active: &[u16], target: f64) {
        for (i, &a) in active.iter().enumerate() {
            self.target[a as usize] += target;
            for &b in &active[i..] {
                let (low, high) = if a <= b { (a, b) } else { (b, a) };
                self.gram[low as usize * FEATURES + high as usize] += 1.0;
            }
        }
        self.samples += 1;
        self.target_sum += target;
        self.target_squares += target * target;
    }

    pub fn merge(&mut self, other: &Accumulator) {
        for (mine, theirs) in self.gram.iter_mut().zip(&other.gram) {
            *mine += theirs;
        }
        for (mine, theirs) in self.target.iter_mut().zip(&other.target) {
            *mine += theirs;
        }
        self.samples += other.samples;
        self.target_sum += other.target_sum;
        self.target_squares += other.target_squares;
    }

    /// Solve `(X'X + ridge I) w = X'y` by Cholesky factorization.
    pub fn solve(&self, ridge: f64) -> Vec<f32> {
        let n = FEATURES;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let value = self.gram[i * n + j] + if i == j { ridge } else { 0.0 };
                a[i * n + j] = value;
                a[j * n + i] = value;
            }
        }
        // In-place lower-triangular factor L with A = L L'.
        for j in 0..n {
            let mut diagonal = a[j * n + j];
            for k in 0..j {
                diagonal -= a[j * n + k] * a[j * n + k];
            }
            let diagonal = diagonal.max(1e-9).sqrt();
            a[j * n + j] = diagonal;
            for i in j + 1..n {
                let mut value = a[i * n + j];
                for k in 0..j {
                    value -= a[i * n + k] * a[j * n + k];
                }
                a[i * n + j] = value / diagonal;
            }
        }
        // Forward then back substitution.
        let mut w: Vec<f64> = self.target.clone();
        for i in 0..n {
            for k in 0..i {
                w[i] -= a[i * n + k] * w[k];
            }
            w[i] /= a[i * n + i];
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                w[i] -= a[k * n + i] * w[k];
            }
            w[i] /= a[i * n + i];
        }
        w.into_iter().map(|x| x as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn rack(text: &str) -> Rack {
        Rack::parse(text).unwrap()
    }

    #[test]
    fn features_are_distinct_in_range_and_describe_the_leave() {
        let mut active = [0u16; MAX_ACTIVE];
        let n = features(&rack("SS?QU"), 40, &mut active);
        // 5 tile copies + balance + pairs among {Q, S, U, ?} (6) + bucket.
        assert_eq!(n, 5 + 1 + 6 + 1);
        let unique: std::collections::HashSet<u16> = active[..n].iter().copied().collect();
        assert_eq!(unique.len(), n);
        assert!(active[..n].iter().all(|&f| (f as usize) < FEATURES));
        assert_eq!(features(&Rack::new(), 0, &mut active), 2, "empty leave: balance and bucket only");

        // The largest possible leave fits.
        let n = features(&rack("ABCDEF?"), 80, &mut active);
        assert_eq!(n, MAX_ACTIVE);
        // Every pair index is distinct.
        let mut pairs = std::collections::HashSet::new();
        for a in 0..TILE_TYPES {
            for b in a + 1..TILE_TYPES {
                assert!(pairs.insert(a * (2 * TILE_TYPES - a - 1) / 2 + (b - a - 1)));
            }
        }
        assert_eq!(pairs.len(), PAIR_FEATURES);
        assert_eq!(*pairs.iter().max().unwrap(), PAIR_FEATURES - 1);
    }

    #[test]
    fn stages_split_at_the_end_of_the_bag() {
        assert_eq!([stage(0), stage(1), stage(6), stage(7), stage(86)], [2, 1, 1, 0, 0]);
    }

    #[test]
    fn ridge_regression_recovers_planted_weights() {
        // Targets generated from a known linear model plus noise; the fit should predict them.
        let mut rng = Rng::new(1);
        let truth: Vec<f64> = (0..FEATURES).map(|_| rng.unit() * 10.0 - 5.0).collect();
        let mut accumulator = Accumulator::default();
        let mut samples = Vec::new();
        for _ in 0..60_000 {
            let mut leave = Rack::new();
            for _ in 0..rng.below(7) {
                leave.add(rng.below(TILE_TYPES as u32) as u8);
            }
            let bag = 7 + rng.below(80) as usize;
            let mut active = [0u16; MAX_ACTIVE];
            let n = features(&leave, bag, &mut active);
            let clean: f64 = active[..n].iter().map(|&f| truth[f as usize]).sum();
            accumulator.add(&active[..n], clean + (rng.unit() - 0.5) * 2.0);
            samples.push((leave, bag, clean));
        }
        let model = LeaveModel { weights: [accumulator.solve(0.01), vec![0.0; FEATURES], vec![0.0; FEATURES]] };
        let errors: Vec<f64> =
            samples.iter().map(|(leave, bag, clean)| (model.value(leave, *bag) as f64 - clean).abs()).collect();
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        assert!(mean < 0.25, "mean prediction error {mean}");

        let restored = LeaveModel::from_bytes(&model.to_bytes()).unwrap();
        assert_eq!(restored, model);
        assert!(LeaveModel::from_bytes(&model.to_bytes()[1..]).is_err());
    }

    #[test]
    fn merging_accumulators_equals_one_pass() {
        let (mut whole, mut first, mut second) = (Accumulator::default(), Accumulator::default(), Accumulator::default());
        let mut active = [0u16; MAX_ACTIVE];
        for (i, text) in ["AEI", "QU", "SS?", "RETINA", ""].iter().enumerate() {
            let n = features(&rack(text), 30, &mut active);
            whole.add(&active[..n], i as f64);
            if i < 2 { &mut first } else { &mut second }.add(&active[..n], i as f64);
        }
        first.merge(&second);
        assert_eq!(first.solve(1.0), whole.solve(1.0));
        assert_eq!(first.samples, 5);
    }
}
