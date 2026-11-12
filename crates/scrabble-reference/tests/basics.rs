//! Round trips and tables: board text, square names, placed-tile text, premium squares, tile values,
//! lexicon loading.

mod common;

use common::*;
use scrabble_reference::*;

#[test]
fn square_names_round_trip_and_match_the_examples() {
    assert_eq!(square_name(0, 0), "A1");
    assert_eq!(square_name(0, 14), "O1");
    assert_eq!(square_name(14, 0), "A15");
    assert_eq!(square_name(7, 7), "H8");
    assert_eq!(parse_square("A1"), Some((0, 0)));
    assert_eq!(parse_square("O1"), Some((0, 14)));
    assert_eq!(parse_square("A15"), Some((14, 0)));
    assert_eq!(parse_square("H8"), Some((7, 7)));
    for row in 0..N as u8 {
        for col in 0..N as u8 {
            assert_eq!(parse_square(&square_name(row, col)), Some((row, col)));
        }
    }
    for bad in [
        "", "H", "8", "H0", "H16", "P1", "h8", "H08", "H+8", "HH", "H8 ", "A100", "É1",
    ] {
        assert_eq!(parse_square(bad), None, "{bad:?}");
    }
}

#[test]
fn placed_tile_text_round_trips() {
    let c = parse_placed_tile("H8=C").unwrap();
    assert_eq!(
        c,
        PlacedTile {
            row: 7,
            col: 7,
            letter: b'C',
            blank: false
        }
    );
    let t = parse_placed_tile("J8=t").unwrap();
    assert_eq!(
        t,
        PlacedTile {
            row: 7,
            col: 9,
            letter: b'T',
            blank: true
        }
    );
    assert_eq!(placed_tile_text(&c), "H8=C");
    assert_eq!(placed_tile_text(&t), "J8=t");
    for bad in ["H8", "H8=", "H8=CC", "H8=?", "H8=1", "Z8=C", "=C"] {
        assert!(parse_placed_tile(bad).is_err(), "{bad:?}");
    }
    let play = RefPlay {
        tiles: vec![c, t],
        score: 3,
    };
    assert_eq!(play.to_string(), "H8=C J8=t (3)");
}

#[test]
fn board_text_round_trips() {
    // A fixed board with blanks.
    let mut text = String::new();
    for row in 0..N {
        let line: String = (0..N)
            .map(|col| match (row * 7 + col * 3) % 5 {
                0 => '.',
                1 => (b'A' + ((row + col) % 26) as u8) as char,
                2 => (b'a' + ((row * col) % 26) as u8) as char,
                _ => '.',
            })
            .collect();
        text.push_str(&line);
        text.push('\n');
    }
    let board = RefBoard::from_text(&text).unwrap();
    assert_eq!(board.to_text(), text);
    // CRLF and surrounding blank lines are accepted.
    let crlf = format!("\r\n{}\r\n", text.replace('\n', "\r\n"));
    assert_eq!(RefBoard::from_text(&crlf).unwrap(), board);
    // Boards from played games, with blanks, round-trip too.
    let lexicon = small_lexicon();
    let mut rng = Rng::new(7);
    for _ in 0..5 {
        for board in play_game(&lexicon, &mut rng, 12, Choice::Random) {
            assert_eq!(RefBoard::from_text(&board.to_text()).unwrap(), board);
        }
    }
    assert_eq!(RefBoard::empty().to_text(), "...............\n".repeat(15));
    assert!(RefBoard::empty().is_empty());

    let bad_char = text.replacen('.', "#", 1);
    assert!(RefBoard::from_text(&bad_char).is_err());
    let short_line = text.replacen('.', "", 1);
    assert!(RefBoard::from_text(&short_line).is_err());
    let fourteen: String = text.lines().take(14).map(|l| format!("{l}\n")).collect();
    assert!(RefBoard::from_text(&fourteen).is_err());
}

#[test]
fn place_puts_tiles_on_the_board() {
    let mut board = RefBoard::empty();
    board.place(&[
        PlacedTile {
            row: 7,
            col: 7,
            letter: b'C',
            blank: false,
        },
        PlacedTile {
            row: 7,
            col: 8,
            letter: b'A',
            blank: true,
        },
    ]);
    assert_eq!(board.cells[7][7], Some((b'C', false)));
    assert_eq!(board.cells[7][8], Some((b'A', true)));
    assert!(board.to_text().lines().nth(7).unwrap() == ".......Ca......");
}

/// The premium squares as listed by name (my own reading of the RULES.md diagram, written out
/// separately from the layout table in the crate).
#[test]
fn premium_squares_match_the_diagram() {
    let lists = [
        (Premium::TripleWord, "A1 H1 O1 A8 O8 A15 H15 O15"),
        (
            Premium::DoubleWord,
            "B2 N2 C3 M3 D4 L4 E5 K5 H8 E11 K11 D12 L12 C13 M13 B14 N14",
        ),
        (
            Premium::TripleLetter,
            "F2 J2 B6 F6 J6 N6 B10 F10 J10 N10 F14 J14",
        ),
        (
            Premium::DoubleLetter,
            "D1 L1 G3 I3 A4 H4 O4 C7 G7 I7 M7 D8 L8 C9 G9 I9 M9 A12 H12 O12 G13 I13 D15 L15",
        ),
    ];
    let mut expected = [[Premium::Plain; N]; N];
    for (premium, names) in lists {
        for name in names.split_whitespace() {
            let (r, c) = parse_square(name).unwrap();
            expected[r as usize][c as usize] = premium;
        }
    }
    for r in 0..N as u8 {
        for c in 0..N as u8 {
            assert_eq!(
                premium_at(r, c),
                expected[r as usize][c as usize],
                "{}",
                square_name(r, c)
            );
            // Symmetric under both reflections and the transpose.
            assert_eq!(premium_at(r, c), premium_at(c, r));
            assert_eq!(premium_at(r, c), premium_at(14 - r, c));
            assert_eq!(premium_at(r, c), premium_at(r, 14 - c));
        }
    }
    assert_eq!(premium_at(7, 7), Premium::DoubleWord);
}

#[test]
fn tile_values_and_counts_match_the_table() {
    let values = "A1 B3 C3 D2 E1 F4 G2 H4 I1 J8 K5 L1 M3 N1 O1 P3 Q10 R1 S1 T1 U1 V4 W4 X8 Y4 Z10";
    for item in values.split_whitespace() {
        let letter = item.as_bytes()[0];
        let value: i32 = item[1..].parse().unwrap();
        assert_eq!(tile_value(letter), value, "{}", letter as char);
    }
    assert_eq!(full_bag().len(), 100);
    let total: i32 = full_bag()
        .iter()
        .filter(|&&t| t != b'?')
        .map(|&t| tile_value(t))
        .sum();
    assert_eq!(total, 187);
}

#[test]
fn lexicon_from_words_and_from_file() {
    let lexicon = RefLexicon::from_words(["cat", " Dog ", "CAT", "it's", "", "x-ray", "A"]);
    assert_eq!(lexicon.len(), 3); // CAT, DOG, A
    assert!(lexicon.contains(b"CAT"));
    assert!(lexicon.contains(b"DOG"));
    assert!(lexicon.contains(b"A"));
    assert!(!lexicon.contains(b"cat"));
    assert!(!lexicon.contains(b"CATS"));

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/tiny_lexicon.txt");
    let lexicon = RefLexicon::from_file(&path).unwrap();
    assert_eq!(lexicon.len(), 5);
    for word in ["CAT", "CATS", "DOG", "AT", "ZOO"] {
        assert!(lexicon.contains(word.as_bytes()), "{word}");
    }
    assert!(
        !lexicon.contains(b"FELINE"),
        "only the first token of a line is a word"
    );
}
