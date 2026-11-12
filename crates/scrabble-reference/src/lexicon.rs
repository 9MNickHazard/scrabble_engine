//! A plain set of words.

use std::collections::HashSet;
use std::path::Path;

use crate::tiles::N;

/// A set of upper-case words.
pub struct RefLexicon {
    words: HashSet<Vec<u8>>,
    /// The same words of length 2..=15 grouped by length (index = length), each with its letter
    /// counts. Only `legal_plays` uses this; `check_placement` only calls `contains`.
    by_len: Vec<Vec<LexWord>>,
}

pub(crate) struct LexWord {
    pub letters: Box<[u8]>,
    /// `counts[i]` = how many times `b'A' + i` occurs in the word.
    pub counts: [u8; 26],
}

impl RefLexicon {
    /// Build from words. Each word is trimmed and upper-cased; a word that is then empty or contains
    /// anything other than `A`-`Z` is skipped (it could never be formed on the board).
    pub fn from_words<I: IntoIterator<Item = S>, S: AsRef<str>>(words: I) -> Self {
        let mut set: HashSet<Vec<u8>> = HashSet::new();
        for word in words {
            let upper = word.as_ref().trim().to_ascii_uppercase();
            if upper.is_empty() || !upper.bytes().all(|b| b.is_ascii_uppercase()) {
                continue;
            }
            set.insert(upper.into_bytes());
        }
        let mut sorted: Vec<&Vec<u8>> = set.iter().collect();
        sorted.sort();
        let mut by_len: Vec<Vec<LexWord>> = (0..=N).map(|_| Vec::new()).collect();
        for word in sorted {
            if word.len() < 2 || word.len() > N {
                continue;
            }
            let mut counts = [0u8; 26];
            for &b in word {
                counts[(b - b'A') as usize] += 1;
            }
            by_len[word.len()].push(LexWord {
                letters: word.clone().into_boxed_slice(),
                counts,
            });
        }
        RefLexicon { words: set, by_len }
    }

    /// Read a word list: the first whitespace-separated token of each line, upper-cased. Lines with
    /// no token are skipped, and so are tokens that are not purely letters (see `from_words`).
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let text = String::from_utf8_lossy(&bytes);
        Ok(Self::from_words(
            text.lines()
                .filter_map(|line| line.split_whitespace().next()),
        ))
    }

    /// True when `word` (upper-case ASCII letters) is in the set.
    pub fn contains(&self, word: &[u8]) -> bool {
        self.words.contains(word)
    }

    /// The number of distinct words.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Words of exactly `len` letters (empty outside 2..=15).
    pub(crate) fn words_of_len(&self, len: usize) -> &[LexWord] {
        self.by_len.get(len).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

impl std::fmt::Debug for RefLexicon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RefLexicon({} words)", self.words.len())
    }
}
