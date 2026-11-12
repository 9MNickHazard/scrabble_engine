//! Tiles, tile values and racks.
//!
//! A tile type is a number: 0..=25 for A..=Z and [`BLANK`] (26) for a blank. Racks, the bag and
//! the unseen pool are all counts per tile type, `[u8; 27]`.

pub const LETTERS: usize = 26;
pub const BLANK: u8 = 26;
pub const TILE_TYPES: usize = 27;
pub const RACK_SIZE: usize = 7;
pub const TOTAL_TILES: usize = 100;
pub const BINGO_BONUS: i32 = 50;

/// Bit mask with one bit per letter A..=Z.
pub const ALL_LETTERS: u32 = (1 << LETTERS) - 1;

//                                       A  B  C  D  E   F  G  H  I  J  K  L  M  N  O  P  Q   R  S  T  U  V  W  X  Y  Z   ?
pub const TILE_COUNTS: [u8; TILE_TYPES] = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2];
pub const TILE_VALUES: [u8; TILE_TYPES] = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, 0];

/// Tile type for a rack character: `A`-`Z` (either case) or `?`.
pub fn tile_from_char(c: char) -> Option<u8> {
    match c {
        'A'..='Z' => Some(c as u8 - b'A'),
        'a'..='z' => Some(c as u8 - b'a'),
        '?' => Some(BLANK),
        _ => None,
    }
}

pub fn tile_to_char(tile: u8) -> char {
    if tile == BLANK {
        '?'
    } else {
        (b'A' + tile) as char
    }
}

/// A multiset of tiles: a rack, a leave, or the tiles thrown back in an exchange.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rack {
    pub counts: [u8; TILE_TYPES],
    len: u8,
}

impl Rack {
    pub const fn new() -> Self {
        Rack { counts: [0; TILE_TYPES], len: 0 }
    }

    pub fn from_counts(counts: [u8; TILE_TYPES]) -> Self {
        let len = counts.iter().map(|&c| c as u32).sum::<u32>() as u8;
        Rack { counts, len }
    }

    /// Parse `"AEINRT?"`. Returns `None` on any character that is not a letter or `?`.
    pub fn parse(text: &str) -> Option<Self> {
        let mut rack = Rack::new();
        for c in text.chars() {
            rack.add(tile_from_char(c)?);
        }
        Some(rack)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn count(&self, tile: u8) -> u8 {
        self.counts[tile as usize]
    }

    #[inline]
    pub fn add(&mut self, tile: u8) {
        self.counts[tile as usize] += 1;
        self.len += 1;
    }

    /// Remove one tile. Panics in debug builds if the rack does not hold it.
    #[inline]
    pub fn remove(&mut self, tile: u8) {
        debug_assert!(self.counts[tile as usize] > 0, "rack does not hold tile {tile}");
        self.counts[tile as usize] -= 1;
        self.len -= 1;
    }

    /// True if every tile in `other` can be taken from this rack.
    pub fn contains_all(&self, other: &Rack) -> bool {
        self.counts.iter().zip(other.counts.iter()).all(|(mine, theirs)| mine >= theirs)
    }

    pub fn remove_all(&mut self, other: &Rack) {
        debug_assert!(self.contains_all(other));
        for (mine, theirs) in self.counts.iter_mut().zip(other.counts.iter()) {
            *mine -= theirs;
        }
        self.len -= other.len;
    }

    /// Sum of the tile values (a blank is worth 0).
    pub fn value(&self) -> i32 {
        self.counts.iter().zip(TILE_VALUES.iter()).map(|(&n, &v)| n as i32 * v as i32).sum()
    }

    /// Bit mask of the letters (not the blank) present.
    #[inline]
    pub fn letter_mask(&self) -> u32 {
        let mut mask = 0;
        for letter in 0..LETTERS {
            mask |= ((self.counts[letter] != 0) as u32) << letter;
        }
        mask
    }

    /// The tiles one by one in tile order, e.g. for A, A, E, ?: 0, 0, 4, 26.
    pub fn tiles(&self) -> impl Iterator<Item = u8> + '_ {
        (0..TILE_TYPES as u8).flat_map(move |t| std::iter::repeat(t).take(self.counts[t as usize] as usize))
    }
}

impl std::fmt::Display for Rack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for tile in self.tiles() {
            write!(f, "{}", tile_to_char(tile))?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Rack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rack({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_is_the_standard_100_tiles() {
        assert_eq!(TILE_COUNTS.iter().map(|&c| c as usize).sum::<usize>(), TOTAL_TILES);
        let total_value: u32 = TILE_COUNTS.iter().zip(TILE_VALUES).map(|(&c, v)| c as u32 * v as u32).sum();
        assert_eq!(total_value, 187);
    }

    #[test]
    fn rack_round_trip_and_value() {
        let rack = Rack::parse("QUIZ?ae").unwrap();
        assert_eq!(rack.len(), 7);
        assert_eq!(rack.to_string(), "AEIQUZ?");
        assert_eq!(rack.value(), 1 + 1 + 1 + 10 + 1 + 10);
        assert_eq!(rack.letter_mask().count_ones(), 6);
        assert!(Rack::parse("AB1").is_none());
    }

    #[test]
    fn remove_all_takes_a_subset() {
        let mut rack = Rack::parse("AABC?").unwrap();
        let part = Rack::parse("A?").unwrap();
        assert!(rack.contains_all(&part));
        rack.remove_all(&part);
        assert_eq!(rack.to_string(), "ABC");
        assert!(!rack.contains_all(&Rack::parse("AA").unwrap()));
    }
}
