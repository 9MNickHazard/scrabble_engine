//! The lexicon compiled into a GADDAG (Gordon, 1994).
//!
//! For a word `w = w1..wn` the GADDAG holds, for every split point `1 <= i <= n`, the string
//! `reverse(w1..wi)` followed, when `i < n`, by the separator and `w(i+1)..wn`. Starting from any
//! letter of a word, a move generator can therefore spell the word outwards: first leftwards to
//! its start, then, after the separator, rightwards to its end.
//!
//! All of those strings are stored in one minimal acyclic automaton, laid out in a flat `u32`
//! array. A node is
//!
//! ```text
//! [ child mask ][ terminal mask ][ child index for each bit set in the child mask, low bit first ]
//! ```
//!
//! Bit `s` of the child mask says symbol `s` leads to another node; bit `s` of the terminal mask
//! says a stored string ends with symbol `s` here. Symbols are letters 0..=25 and [`SEPARATOR`].
//! Node 0 is a dead node with both masks empty, so "no child" needs no special case.

use crate::lexicon::Lexicon;
use crate::tiles::ALL_LETTERS;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const SEPARATOR: u8 = 26;

const MAGIC: &[u8; 4] = b"SGDG";
const FORMAT_VERSION: u32 = 1;

pub struct Gaddag {
    nodes: Vec<u32>,
    root: u32,
}

/// A node still on the path being built: its masks and the children finished so far.
struct Open {
    children_mask: u32,
    terminal_mask: u32,
    children: Vec<u32>,
    /// Symbol of the edge to the next node on the open path.
    edge_out: u8,
}

impl Open {
    fn new() -> Self {
        Open { children_mask: 0, terminal_mask: 0, children: Vec::new(), edge_out: 0 }
    }
}

impl Gaddag {
    /// Build the minimal automaton from upper-case A-Z words of at least two letters.
    ///
    /// Uses the incremental construction of Daciuk et al.: feed the strings in sorted order and
    /// close each node as soon as no later string can pass through it, merging it with an
    /// identical node if one was already closed.
    pub fn build<S: AsRef<str>>(words: &[S]) -> Gaddag {
        // Every GADDAG string, back to back in one buffer.
        let mut data: Vec<u8> = Vec::new();
        let mut spans: Vec<(u32, u8)> = Vec::new();
        for word in words {
            let word: Vec<u8> = word.as_ref().bytes().map(|b| b.to_ascii_uppercase() - b'A').collect();
            assert!(word.len() >= 2 && word.iter().all(|&s| s < 26), "not a playable word");
            for split in 1..=word.len() {
                let start = data.len() as u32;
                data.extend(word[..split].iter().rev());
                if split < word.len() {
                    data.push(SEPARATOR);
                    data.extend(&word[split..]);
                }
                spans.push((start, (data.len() as u32 - start) as u8));
            }
        }
        let string = |&(start, len): &(u32, u8)| &data[start as usize..start as usize + len as usize];
        spans.sort_unstable_by(|a, b| string(a).cmp(string(b)));
        spans.dedup_by(|a, b| string(a) == string(b));

        let mut nodes: Vec<u32> = vec![0, 0];
        let mut closed: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut close = |node: Open| -> u32 {
            let mut key = Vec::with_capacity(2 + node.children.len());
            key.push(node.children_mask);
            key.push(node.terminal_mask);
            key.extend(&node.children);
            *closed.entry(key).or_insert_with_key(|key| {
                let index = nodes.len() as u32;
                nodes.extend(key);
                index
            })
        };

        // `path[d]` is the open node reached after the first `d` symbols of the previous string.
        let mut path: Vec<Open> = vec![Open::new()];
        let mut previous: &[u8] = &[];
        for span in &spans {
            let current = string(span);
            let common = previous.iter().zip(current).take_while(|(a, b)| a == b).count();
            while path.len() - 1 > common {
                let finished = close(path.pop().unwrap());
                let parent = path.last_mut().unwrap();
                parent.children_mask |= 1 << parent.edge_out;
                parent.children.push(finished);
            }
            for (depth, &symbol) in current.iter().enumerate().skip(path.len() - 1) {
                let node = path.last_mut().unwrap();
                if depth + 1 == current.len() {
                    node.terminal_mask |= 1 << symbol;
                } else {
                    node.edge_out = symbol;
                    path.push(Open::new());
                }
            }
            previous = current;
        }
        while path.len() > 1 {
            let finished = close(path.pop().unwrap());
            let parent = path.last_mut().unwrap();
            parent.children_mask |= 1 << parent.edge_out;
            parent.children.push(finished);
        }
        let root = close(path.pop().unwrap());
        nodes.shrink_to_fit();
        Gaddag { nodes, root }
    }

    #[inline]
    pub fn root(&self) -> u32 {
        self.root
    }

    /// Symbols that lead from `node` to another node.
    #[inline]
    pub fn children_mask(&self, node: u32) -> u32 {
        self.nodes[node as usize]
    }

    /// Symbols on which a stored string ends at `node`.
    #[inline]
    pub fn terminal_mask(&self, node: u32) -> u32 {
        self.nodes[node as usize + 1]
    }

    /// The node reached from `node` by `symbol`, or 0 (the dead node) if there is none.
    #[inline]
    pub fn child(&self, node: u32, symbol: u8) -> u32 {
        let mask = self.nodes[node as usize];
        let bit = 1u32 << symbol;
        if mask & bit == 0 {
            0
        } else {
            self.nodes[node as usize + 2 + (mask & (bit - 1)).count_ones() as usize]
        }
    }

    /// Is this sequence of letters (0..=25) a word?
    pub fn contains(&self, word: &[u8]) -> bool {
        // The split after the last letter stores the whole word reversed.
        let Some((&first, rest)) = word.split_first() else { return false };
        let mut node = self.root;
        for &letter in rest.iter().rev() {
            node = self.child(node, letter);
        }
        self.terminal_mask(node) & (1 << first) != 0
    }

    pub fn contains_str(&self, word: &str) -> bool {
        let letters: Option<Vec<u8>> = word
            .bytes()
            .map(|b| b.is_ascii_alphabetic().then(|| b.to_ascii_uppercase() - b'A'))
            .collect();
        letters.is_some_and(|letters| self.contains(&letters))
    }

    /// Letters `x` for which `before + x + after` is a word, as a bit mask.
    ///
    /// This is the cross-check for an empty square with `before` directly above (or left of) it
    /// and `after` directly below (or right of) it. At least one of the two must be non-empty.
    pub fn letters_between(&self, before: &[u8], after: &[u8]) -> u32 {
        debug_assert!(!before.is_empty() || !after.is_empty());
        if before.is_empty() {
            // Whole word reversed: reverse(after) then x, ending there.
            let mut node = self.root;
            for &letter in after.iter().rev() {
                node = self.child(node, letter);
            }
            return self.terminal_mask(node) & ALL_LETTERS;
        }
        // Split after `before`: reverse(before), separator, x, after.
        let mut node = self.root;
        for &letter in before.iter().rev() {
            node = self.child(node, letter);
        }
        node = self.child(node, SEPARATOR);
        let Some((&last, middle)) = after.split_last() else {
            return self.terminal_mask(node) & ALL_LETTERS;
        };
        let mut allowed = 0;
        let mut candidates = self.children_mask(node) & ALL_LETTERS;
        while candidates != 0 {
            let x = candidates.trailing_zeros() as u8;
            candidates &= candidates - 1;
            let mut walk = self.child(node, x);
            for &letter in middle {
                walk = self.child(walk, letter);
            }
            if self.terminal_mask(walk) & (1 << last) != 0 {
                allowed |= 1 << x;
            }
        }
        allowed
    }

    pub fn node_words(&self) -> usize {
        self.nodes.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * 4
    }

    // ---- Binary cache ----------------------------------------------------------------------

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.nodes.len() * 4 + 8);
        out.extend(MAGIC);
        out.extend(FORMAT_VERSION.to_le_bytes());
        out.extend(self.root.to_le_bytes());
        out.extend((self.nodes.len() as u32).to_le_bytes());
        for word in &self.nodes {
            out.extend(word.to_le_bytes());
        }
        let checksum = fnv1a(&out);
        out.extend(checksum.to_le_bytes());
        out
    }

    /// Rebuilds from [`Gaddag::to_bytes`]. Fails on a truncated, corrupt or older-format file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Gaddag, String> {
        if bytes.len() < 24 || &bytes[..4] != MAGIC {
            return Err("not a GADDAG cache file".to_string());
        }
        let field = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        if field(4) != FORMAT_VERSION {
            return Err(format!("GADDAG cache format {} is not {FORMAT_VERSION}", field(4)));
        }
        let (root, count) = (field(8), field(12) as usize);
        let body_end = 16 + count * 4;
        if bytes.len() != body_end + 8 {
            return Err("GADDAG cache file has the wrong length (truncated?)".to_string());
        }
        let stored = u64::from_le_bytes(bytes[body_end..].try_into().unwrap());
        if stored != fnv1a(&bytes[..body_end]) {
            return Err("GADDAG cache file fails its checksum".to_string());
        }
        let nodes: Vec<u32> =
            bytes[16..body_end].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        if root as usize + 2 > nodes.len() {
            return Err("GADDAG cache file has an invalid root".to_string());
        }
        Ok(Gaddag { nodes, root })
    }

    /// Path of the cache file for a lexicon inside `cache_dir`.
    pub fn cache_path(lexicon: &Lexicon, cache_dir: &Path) -> PathBuf {
        cache_dir.join(format!("{}.gaddag", lexicon.id()))
    }

    /// Load the compiled GADDAG for `lexicon` from `cache_dir`, building and saving it if the
    /// cache is missing or unreadable. The file is written under a temporary name and renamed,
    /// so an interrupted write cannot leave a half-written cache behind.
    pub fn load_or_build(lexicon: &Lexicon, cache_dir: &Path) -> Result<Gaddag, String> {
        let path = Gaddag::cache_path(lexicon, cache_dir);
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(gaddag) = Gaddag::from_bytes(&bytes) {
                return Ok(gaddag);
            }
        }
        let gaddag = Gaddag::build(&lexicon.words);
        std::fs::create_dir_all(cache_dir).map_err(|e| format!("{}: {e}", cache_dir.display()))?;
        let temporary = path.with_extension(format!("gaddag.tmp{}", std::process::id()));
        std::fs::write(&temporary, gaddag.to_bytes()).map_err(|e| format!("{}: {e}", temporary.display()))?;
        std::fs::rename(&temporary, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(gaddag)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORDS: [&str; 12] =
        ["AT", "TA", "CAT", "CATS", "ACT", "ACTS", "CARE", "CARES", "SCARE", "RACE", "AA", "QI"];

    fn letters(word: &str) -> Vec<u8> {
        word.bytes().map(|b| b - b'A').collect()
    }

    fn mask(letters: &str) -> u32 {
        letters.bytes().fold(0, |m, b| m | 1 << (b - b'A'))
    }

    #[test]
    fn contains_exactly_the_words() {
        let gaddag = Gaddag::build(&WORDS);
        for word in WORDS {
            assert!(gaddag.contains_str(word), "{word}");
        }
        for not_word in ["A", "C", "CA", "CAR", "CATSS", "TAC", "ARE", "SCAR", "ERAC", "IQ", "", "A-T"] {
            assert!(!gaddag.contains_str(not_word), "{not_word}");
        }
    }

    #[test]
    fn every_split_of_every_word_is_a_path() {
        let gaddag = Gaddag::build(&WORDS);
        for word in WORDS {
            let word = letters(word);
            for split in 1..=word.len() {
                let mut symbols: Vec<u8> = word[..split].iter().rev().copied().collect();
                if split < word.len() {
                    symbols.push(SEPARATOR);
                    symbols.extend(&word[split..]);
                }
                let (&last, body) = symbols.split_last().unwrap();
                let node = body.iter().fold(gaddag.root(), |node, &s| gaddag.child(node, s));
                assert!(gaddag.terminal_mask(node) & (1 << last) != 0, "{word:?} split {split}");
            }
        }
        // The separator is never the end of a string.
        assert_eq!(gaddag.terminal_mask(gaddag.root()) & (1 << SEPARATOR), 0);
    }

    #[test]
    fn letters_between_gives_cross_checks() {
        let gaddag = Gaddag::build(&WORDS);
        assert_eq!(gaddag.letters_between(&letters("CAT"), &[]), mask("S"));
        assert_eq!(gaddag.letters_between(&letters("A"), &[]), mask("TA"));
        assert_eq!(gaddag.letters_between(&[], &letters("CARE")), mask("S"));
        assert_eq!(gaddag.letters_between(&[], &letters("T")), mask("A"));
        assert_eq!(gaddag.letters_between(&letters("C"), &letters("T")), mask("A"));
        assert_eq!(gaddag.letters_between(&letters("CA"), &letters("E")), mask("R"));
        assert_eq!(gaddag.letters_between(&letters("C"), &letters("RES")), mask("A"));
        assert_eq!(gaddag.letters_between(&letters("Q"), &[]), mask("I"));
        assert_eq!(gaddag.letters_between(&[], &letters("Q")), 0);
        assert_eq!(gaddag.letters_between(&letters("Z"), &letters("Z")), 0);
    }

    #[test]
    fn construction_is_minimal_and_order_independent() {
        let gaddag = Gaddag::build(&WORDS);
        let mut shuffled = WORDS;
        shuffled.reverse();
        shuffled.swap(1, 7);
        assert_eq!(Gaddag::build(&shuffled).nodes, gaddag.nodes);

        // No two nodes are identical: that is what minimal means for this layout.
        let mut seen = std::collections::HashSet::new();
        let mut index = 2;
        while index < gaddag.nodes.len() {
            let size = 2 + gaddag.nodes[index].count_ones() as usize;
            assert!(seen.insert(&gaddag.nodes[index..index + size]), "duplicate node at {index}");
            index += size;
        }
    }

    #[test]
    fn cache_round_trips_and_detects_damage() {
        let gaddag = Gaddag::build(&WORDS);
        let bytes = gaddag.to_bytes();
        let loaded = Gaddag::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.nodes, gaddag.nodes);
        assert_eq!(loaded.root, gaddag.root);
        assert!(Gaddag::from_bytes(&bytes[..bytes.len() - 5]).is_err(), "truncated");
        let mut flipped = bytes.clone();
        flipped[20] ^= 1;
        assert!(Gaddag::from_bytes(&flipped).is_err(), "corrupt");
        assert!(Gaddag::from_bytes(b"nope").is_err());
    }
}
