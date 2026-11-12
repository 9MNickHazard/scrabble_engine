//! A small, fixed random number generator (xoshiro256++ seeded through SplitMix64).
//!
//! Written out here rather than taken from a crate so that a seed means the same game in every
//! build: paired evaluation and resumable training both depend on that.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng { state: [next(), next(), next(), next()] }
    }

    pub fn from_state(state: [u64; 4]) -> Rng {
        Rng { state }
    }

    pub fn state(&self) -> [u64; 4] {
        self.state
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n` (`n > 0`), without modulo bias.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let n = n as u64;
        let limit = u64::MAX - u64::MAX % n;
        loop {
            let value = self.next_u64();
            if value < limit {
                return (value % n) as u32;
            }
        }
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i as u32 + 1) as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream_and_seeds_differ() {
        let (mut a, mut b, mut c) = (Rng::new(42), Rng::new(42), Rng::new(43));
        let first: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        assert_eq!(first, (0..8).map(|_| b.next_u64()).collect::<Vec<_>>());
        assert_ne!(first, (0..8).map(|_| c.next_u64()).collect::<Vec<_>>());
        // Pinned values: changing the generator would silently change every seeded game.
        assert_eq!(Rng::new(0).next_u64(), 0x53175D61490B23DF);
    }

    #[test]
    fn below_and_unit_stay_in_range_and_shuffle_permutes() {
        let mut rng = Rng::new(7);
        let mut seen = [0u32; 7];
        for _ in 0..7000 {
            seen[rng.below(7) as usize] += 1;
            let u = rng.unit();
            assert!((0.0..1.0).contains(&u));
        }
        assert!(seen.iter().all(|&n| (800..1200).contains(&n)), "{seen:?}");
        let mut items: Vec<u32> = (0..100).collect();
        rng.shuffle(&mut items);
        assert_ne!(items, (0..100).collect::<Vec<_>>());
        items.sort_unstable();
        assert_eq!(items, (0..100).collect::<Vec<_>>());
    }
}
