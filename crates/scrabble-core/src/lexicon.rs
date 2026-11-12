//! Word lists.
//!
//! File format: one entry per line, the word is the first whitespace-separated token. A lexicon
//! is identified by `<name>-<first 12 hex digits of SHA-256 over the sorted upper-case words
//! joined by '\n'>`, the same value `scrabble_ai.lexicon` computes in Python. Checkpoints and the
//! compiled GADDAG cache are keyed by that ID.

use sha2::{Digest, Sha256};
use std::path::Path;

pub const MIN_WORD_LENGTH: usize = 2;
pub const MAX_WORD_LENGTH: usize = 15;

#[derive(Clone, Debug)]
pub struct Lexicon {
    pub name: String,
    /// Upper case, sorted, unique.
    pub words: Vec<String>,
    pub sha256: String,
}

impl Lexicon {
    pub fn from_words<I, S>(name: &str, words: I) -> Result<Lexicon, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut list: Vec<String> = Vec::new();
        for word in words {
            let word = word.as_ref().to_ascii_uppercase();
            let valid_length = (MIN_WORD_LENGTH..=MAX_WORD_LENGTH).contains(&word.len());
            if !valid_length || !word.bytes().all(|b| b.is_ascii_uppercase()) {
                return Err(format!(
                    "{word:?} is not a word of {MIN_WORD_LENGTH}-{MAX_WORD_LENGTH} letters A-Z"
                ));
            }
            list.push(word);
        }
        if list.is_empty() {
            return Err("no words".to_string());
        }
        list.sort_unstable();
        list.dedup();
        let digest = Sha256::digest(list.join("\n").as_bytes());
        let sha256 = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        Ok(Lexicon { name: name.to_string(), words: list, sha256 })
    }

    pub fn from_file(name: &str, path: &Path) -> Result<Lexicon, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        let words = text.lines().filter_map(|line| line.split_whitespace().next());
        Lexicon::from_words(name, words).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn id(&self) -> String {
        format!("{}-{}", self.name, &self.sha256[..12])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_and_identifies_by_content() {
        let a = Lexicon::from_words("T", ["ab", "AA", "ab"]).unwrap();
        assert_eq!(a.words, ["AA", "AB"]);
        // hashlib.sha256(b"AA\nAB"): scrabble_ai.lexicon must produce the same ID for the same words.
        assert_eq!(a.sha256, "ca7765dda35af78007ef7cb60093bf466fcd1c97b9c4cc4f9b4fb13a909e5a5d");
        assert_eq!(a.id(), "T-ca7765dda35a");
        let b = Lexicon::from_words("T", ["AA", "AD"]).unwrap();
        assert_ne!(a.sha256, b.sha256);
    }

    #[test]
    fn rejects_non_words() {
        for bad in ["A", "ABCDEFGHIJKLMNOP", "IT'S", "X2", "CAF\u{c9}"] {
            assert!(Lexicon::from_words("T", ["AA", bad]).is_err(), "{bad}");
        }
        assert!(Lexicon::from_words("T", Vec::<String>::new()).is_err());
    }
}
