//! The hand-checked cases in testdata/scoring_cases.txt.

use scrabble_reference::*;

fn cases() -> Vec<Case> {
    parse_cases(SCORING_CASES).expect("testdata/scoring_cases.txt should parse")
}

fn sorted(tiles: &[PlacedTile]) -> Vec<PlacedTile> {
    let mut tiles = tiles.to_vec();
    tiles.sort();
    tiles
}

#[test]
fn case_file_has_at_least_40_cases_and_matches_the_file_on_disk() {
    let cases = cases();
    assert!(cases.len() >= 40, "only {} cases", cases.len());
    let on_disk = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/scoring_cases.txt"),
    )
    .unwrap();
    assert_eq!(parse_cases(&on_disk).unwrap(), cases);
    let legal = cases.iter().filter(|c| c.expect.is_some()).count();
    println!(
        "{} cases: {} legal, {} illegal",
        cases.len(),
        legal,
        cases.len() - legal
    );
}

#[test]
fn every_case_gives_the_hand_worked_result() {
    let mut failures = Vec::new();
    for case in cases() {
        let lexicon = RefLexicon::from_words(&case.words);
        let got = check_placement(&case.board, &case.tiles, &lexicon);
        let ok = match (case.expect, &got) {
            (Some(expected), Ok(score)) => expected == *score,
            (None, Err(_)) => true,
            _ => false,
        };
        if !ok {
            failures.push(format!(
                "{}: expected {:?}, check_placement gave {:?}",
                case.name, case.expect, got
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Each illegal case must be rejected for the reason it was written to test, not some other one.
#[test]
fn illegal_cases_fail_for_the_intended_reason() {
    let reasons = [
        ("illegal-opening-off-centre", "no tile is on H8"),
        ("illegal-opening-one-tile", "forms no word"),
        ("illegal-eight-tiles", "8 tiles placed"),
        ("illegal-gap", "gap between the placed tiles"),
        (
            "illegal-gap-both-parts-are-words",
            "gap between the placed tiles",
        ),
        ("illegal-not-connected", "next to a tile"),
        ("illegal-diagonal-single-tile", "next to a tile"),
        ("illegal-main-word", "CATZ is not in the lexicon"),
        ("illegal-cross-word", "TU is not in the lexicon"),
        (
            "illegal-main-word-extended-by-old-tiles-batcat",
            "BATCAT is not in the lexicon",
        ),
        (
            "illegal-main-word-extended-down-con",
            "CON is not in the lexicon",
        ),
        ("illegal-occupied-square", "J8 is already occupied"),
        (
            "illegal-not-in-one-line",
            "not all in one row or one column",
        ),
        ("illegal-two-tiles-on-one-square", "two tiles placed on K8"),
        (
            "illegal-single-tile-one-word-invalid",
            "NA is not in the lexicon",
        ),
    ];
    let cases = cases();
    let illegal: Vec<&Case> = cases.iter().filter(|c| c.expect.is_none()).collect();
    assert_eq!(
        illegal.len(),
        reasons.len(),
        "every illegal case needs a reason here"
    );
    for case in illegal {
        let wanted = reasons
            .iter()
            .find(|(name, _)| *name == case.name)
            .unwrap_or_else(|| panic!("no reason listed for {}", case.name))
            .1;
        let lexicon = RefLexicon::from_words(&case.words);
        let err = check_placement(&case.board, &case.tiles, &lexicon).unwrap_err();
        println!("{}: {err}", case.name);
        assert!(
            err.contains(wanted),
            "{}: got {err:?}, wanted {wanted:?}",
            case.name
        );
    }
}

/// The premium layout is symmetric in the main diagonal, so reflecting the board and the tiles
/// must not change legality or score.
#[test]
fn every_case_gives_the_same_result_transposed() {
    for case in cases() {
        let lexicon = RefLexicon::from_words(&case.words);
        let tiles: Vec<PlacedTile> = case
            .tiles
            .iter()
            .map(|t| PlacedTile {
                row: t.col,
                col: t.row,
                ..*t
            })
            .collect();
        let got = check_placement(&case.board.transposed(), &tiles, &lexicon);
        assert_eq!(got.ok(), case.expect, "{} transposed", case.name);
    }
}

/// Give `legal_plays` a rack holding exactly the case's tiles: a legal case must be among its
/// plays with the expected score, and an illegal one must not be.
#[test]
fn legal_plays_agrees_with_every_case() {
    for case in cases() {
        if case.tiles.len() > 7 {
            continue;
        }
        let lexicon = RefLexicon::from_words(&case.words);
        let rack: String = case
            .tiles
            .iter()
            .map(|t| if t.blank { '?' } else { t.letter as char })
            .collect();
        let plays = legal_plays(&case.board, &rack, &lexicon);
        let wanted = sorted(&case.tiles);
        let found = plays.iter().find(|p| p.tiles == wanted);
        match case.expect {
            Some(score) => assert_eq!(
                found.map(|p| p.score),
                Some(score),
                "{}: rack {rack} should give this play",
                case.name
            ),
            None => assert!(found.is_none(), "{}: illegal play was generated", case.name),
        }
    }
}

/// For the cases with at most 3 tiles (at most one of them blank), the brute-force finder must
/// agree with `legal_plays` on the case's board, lexicon and rack.
#[test]
fn exhaustive_agrees_with_legal_plays_on_small_cases() {
    let mut compared = 0;
    for case in cases() {
        let blanks = case.tiles.iter().filter(|t| t.blank).count();
        if case.tiles.len() > 3 || blanks > 1 {
            continue;
        }
        let lexicon = RefLexicon::from_words(&case.words);
        let rack: String = case
            .tiles
            .iter()
            .map(|t| if t.blank { '?' } else { t.letter as char })
            .collect();
        let fast = legal_plays(&case.board, &rack, &lexicon);
        let slow = legal_plays_exhaustive(&case.board, &rack, &lexicon);
        assert_eq!(fast, slow, "{}: rack {rack}", case.name);
        compared += 1;
    }
    println!("compared {compared} cases");
    assert!(compared >= 20);
}

#[test]
fn case_parser_rejects_malformed_input() {
    let empty_board = "...............\n".repeat(15);
    let good = format!("case: a\nwords: AT\nboard:\n{empty_board}place: H8=A I8=T\nexpect: 4\n");
    let parsed = parse_cases(&good).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].expect, Some(4));
    assert_eq!(parsed[0].words, vec!["AT".to_string()]);

    let short_board = "...............\n".repeat(14);
    let bad = [
        // no case: line
        format!("words: AT\nboard:\n{empty_board}place: H8=A\nexpect: 1\n"),
        // missing expect
        format!("case: a\nwords: AT\nboard:\n{empty_board}place: H8=A\n"),
        // board too short (the place: line becomes the 15th board line)
        format!("case: a\nwords: AT\nboard:\n{short_board}place: H8=A\nexpect: 1\n"),
        // bad square
        format!("case: a\nwords: AT\nboard:\n{empty_board}place: P8=A\nexpect: 1\n"),
        // bad expect
        format!("case: a\nwords: AT\nboard:\n{empty_board}place: H8=A\nexpect: lots\n"),
        // duplicate names
        format!(
            "case: a\nboard:\n{empty_board}place: H8=A\nexpect: illegal\n\ncase: a\nboard:\n{empty_board}place: H8=A\nexpect: illegal\n"
        ),
        // two cases without a blank line between them
        format!("case: a\nboard:\n{empty_board}place: H8=A\nexpect: illegal\ncase: b\n"),
        // unknown key
        format!("case: a\nrack: AT\nboard:\n{empty_board}place: H8=A\nexpect: 1\n"),
    ];
    for text in &bad {
        assert!(parse_cases(text).is_err(), "should not parse:\n{text}");
    }
}
