//! The hand-checked scoring cases file format (see README.md and `testdata/scoring_cases.txt`).

use std::collections::HashSet;

use crate::board::RefBoard;
use crate::tiles::{parse_placed_tile, PlacedTile, N};

/// The contents of `testdata/scoring_cases.txt`, built into the crate.
pub const SCORING_CASES: &str = include_str!("../testdata/scoring_cases.txt");

/// One hand-checked case.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Case {
    pub name: String,
    /// The entire lexicon for this case, upper-cased.
    pub words: Vec<String>,
    pub board: RefBoard,
    /// The tiles exactly as listed on the `place:` line (not re-sorted).
    pub tiles: Vec<PlacedTile>,
    /// `Some(score)` for a legal placement, `None` for `expect: illegal`.
    pub expect: Option<i32>,
}

#[derive(Default)]
struct PartialCase {
    first_line: usize,
    name: String,
    words: Vec<String>,
    board: Option<RefBoard>,
    tiles: Option<Vec<PlacedTile>>,
    expect: Option<Option<i32>>,
}

impl PartialCase {
    fn finish(self) -> Result<Case, String> {
        let where_ = format!("case {:?} (line {})", self.name, self.first_line);
        Ok(Case {
            board: self.board.ok_or_else(|| format!("{where_}: no board:"))?,
            tiles: self.tiles.ok_or_else(|| format!("{where_}: no place:"))?,
            expect: self.expect.ok_or_else(|| format!("{where_}: no expect:"))?,
            name: self.name,
            words: self.words,
        })
    }
}

/// Parse the scoring-cases format.
///
/// ```text
/// # comment lines start with '#'
/// case: opening-cat
/// words: CAT CATS AT
/// board:
/// (15 lines of 15 characters)
/// place: H8=C I8=A J8=T
/// expect: 10
/// ```
///
/// A case starts with `case:` and ends at a blank line (or the end of the text). `words:` may be
/// given more than once (the lists are joined); every other key exactly once. `expect:` is an
/// integer or `illegal`. Case names must be unique.
pub fn parse_cases(text: &str) -> Result<Vec<Case>, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut cases: Vec<Case> = Vec::new();
    let mut current: Option<PartialCase> = None;
    let mut i = 0;
    while i < lines.len() {
        let line_number = i + 1;
        let line = lines[i].trim();
        i += 1;
        if line.starts_with('#') {
            continue;
        }
        if line.is_empty() {
            if let Some(partial) = current.take() {
                cases.push(partial.finish()?);
            }
            continue;
        }
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| format!("line {line_number}: expected 'key: value', got {line:?}"))?;
        let (key, value) = (key.trim(), value.trim());
        if key == "case" {
            if let Some(partial) = &current {
                return Err(format!(
                    "line {line_number}: 'case:' inside case {:?} (missing blank line?)",
                    partial.name
                ));
            }
            if value.is_empty() {
                return Err(format!("line {line_number}: empty case name"));
            }
            current = Some(PartialCase {
                first_line: line_number,
                name: value.to_string(),
                ..PartialCase::default()
            });
            continue;
        }
        let partial = current
            .as_mut()
            .ok_or_else(|| format!("line {line_number}: '{key}:' before any 'case:' line"))?;
        match key {
            "words" => partial
                .words
                .extend(value.split_whitespace().map(|w| w.to_ascii_uppercase())),
            "board" => {
                if partial.board.is_some() {
                    return Err(format!("line {line_number}: second 'board:' in one case"));
                }
                if !value.is_empty() {
                    return Err(format!(
                        "line {line_number}: the board goes on the {N} lines after 'board:'"
                    ));
                }
                if i + N > lines.len() {
                    return Err(format!(
                        "line {line_number}: fewer than {N} lines follow 'board:'"
                    ));
                }
                let block = lines[i..i + N].join("\n");
                if lines[i..i + N].iter().any(|l| l.trim().is_empty()) {
                    return Err(format!("line {line_number}: blank line inside the board"));
                }
                let board = RefBoard::from_text(&block)
                    .map_err(|e| format!("board starting at line {}: {e}", line_number + 1))?;
                partial.board = Some(board);
                i += N;
            }
            "place" => {
                if partial.tiles.is_some() {
                    return Err(format!("line {line_number}: second 'place:' in one case"));
                }
                let tiles = value
                    .split_whitespace()
                    .map(parse_placed_tile)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("line {line_number}: {e}"))?;
                partial.tiles = Some(tiles);
            }
            "expect" => {
                if partial.expect.is_some() {
                    return Err(format!("line {line_number}: second 'expect:' in one case"));
                }
                let expect = if value == "illegal" {
                    None
                } else {
                    Some(value.parse::<i32>().map_err(|_| {
                        format!("line {line_number}: expect must be an integer or 'illegal', got {value:?}")
                    })?)
                };
                partial.expect = Some(expect);
            }
            other => return Err(format!("line {line_number}: unknown key {other:?}")),
        }
    }
    if let Some(partial) = current.take() {
        cases.push(partial.finish()?);
    }
    let mut names = HashSet::new();
    for case in &cases {
        if !names.insert(case.name.as_str()) {
            return Err(format!("duplicate case name {:?}", case.name));
        }
    }
    Ok(cases)
}
