//! Shared helpers for the integration tests.
#![allow(dead_code)]

use scrabble_core::gaddag::Gaddag;
use scrabble_core::lexicon::Lexicon;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Workspace root (two levels above this crate).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Path of the real word list: `SCRABBLE_LEXICON` if set, else `data/lexicon/NSWL2023.txt`.
pub fn lexicon_path() -> PathBuf {
    std::env::var_os("SCRABBLE_LEXICON")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("data").join("lexicon").join("NSWL2023.txt"))
}

/// The real lexicon, or `None` (with a note) on a machine that does not have the word list.
/// Word lists are copyrighted and not in the repo, so tests that need one skip without it.
pub fn real_lexicon() -> Option<&'static Lexicon> {
    static LEXICON: OnceLock<Option<Lexicon>> = OnceLock::new();
    LEXICON
        .get_or_init(|| {
            let path = lexicon_path();
            if !path.exists() {
                eprintln!("SKIPPED: no word list at {}", path.display());
                return None;
            }
            Some(Lexicon::from_file("NSWL2023", &path).expect("word list loads"))
        })
        .as_ref()
}

/// The real lexicon compiled to a GADDAG (built once per test binary, cached on disk).
pub fn real_gaddag() -> Option<&'static Gaddag> {
    static GADDAG: OnceLock<Option<Gaddag>> = OnceLock::new();
    GADDAG
        .get_or_init(|| {
            let lexicon = real_lexicon()?;
            let cache = repo_root().join("data").join("lexicon");
            Some(Gaddag::load_or_build(lexicon, &cache).expect("GADDAG builds"))
        })
        .as_ref()
}

/// A small word list for tests that must run without the real one: the common two-letter words
/// plus a few longer ones, enough that most racks have a play.
pub const MINI_WORDS: &[&str] = &[
    "AA", "AB", "AD", "AE", "AG", "AH", "AI", "AL", "AM", "AN", "AR", "AS", "AT", "AW", "AX", "AY", "BA", "BE", "BI",
    "BO", "BY", "DA", "DE", "DO", "ED", "EF", "EH", "EL", "EM", "EN", "ER", "ES", "EX", "FA", "FE", "GO", "HA", "HE",
    "HI", "HM", "HO", "ID", "IF", "IN", "IS", "IT", "JO", "KA", "KI", "LA", "LI", "LO", "MA", "ME", "MI", "MM", "MO",
    "MU", "MY", "NA", "NE", "NO", "NU", "OD", "OE", "OF", "OH", "OI", "OM", "ON", "OP", "OR", "OS", "OW", "OX", "OY",
    "PA", "PE", "PI", "QI", "RE", "SH", "SI", "SO", "TA", "TI", "TO", "UH", "UM", "UN", "UP", "US", "UT", "WE", "WO",
    "XI", "XU", "YA", "YE", "YO", "ZA", "CAT", "DOG", "TEA", "EAT", "ATE", "RAT", "TAR", "ART", "NET", "TEN", "ONE",
    "EON", "SEA", "SIT", "ITS", "TIE", "TOE", "RETAINS", "STAINER", "QUIZ", "JINX", "VOW", "WAVE", "FIVE", "ZONE",
];

pub fn mini_gaddag() -> Gaddag {
    Gaddag::build(MINI_WORDS)
}
