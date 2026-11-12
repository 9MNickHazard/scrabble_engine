//! Checks that need the real word list. They skip (and say so) when it is not installed.

mod common;

use scrabble_core::board::Board;
use scrabble_core::gaddag::Gaddag;
use scrabble_core::game::Game;
use scrabble_core::moves::{Move, Play};
use scrabble_core::tiles::{Rack, TILE_COUNTS};
use scrabble_core::validate::check_move;
use std::time::Instant;

#[test]
fn gaddag_holds_exactly_the_lexicon() {
    let (Some(lexicon), Some(gaddag)) = (common::real_lexicon(), common::real_gaddag()) else { return };
    assert_eq!(lexicon.words.len(), 195_747);
    assert_eq!(lexicon.id(), "NSWL2023-91a67230e1f6", "must match scrabble_ai.lexicon in Python");
    for word in &lexicon.words {
        assert!(gaddag.contains_str(word), "{word}");
    }
    // Near-misses: every word with its last letter dropped or one letter appended is checked
    // against the list itself.
    let set: std::collections::HashSet<&str> = lexicon.words.iter().map(String::as_str).collect();
    let mut probes = 0;
    for word in lexicon.words.iter().step_by(7) {
        let shorter = &word[..word.len() - 1];
        if shorter.len() >= 2 {
            assert_eq!(gaddag.contains_str(shorter), set.contains(shorter), "{shorter}");
            probes += 1;
        }
        for extra in ['S', 'E', 'Q'] {
            let longer = format!("{word}{extra}");
            let front = format!("{extra}{word}");
            assert_eq!(gaddag.contains_str(&longer), set.contains(longer.as_str()), "{longer}");
            assert_eq!(gaddag.contains_str(&front), set.contains(front.as_str()), "{front}");
            probes += 2;
        }
    }
    println!("GADDAG: {} u32 words, {:.1} MB; {probes} near-miss probes", gaddag.node_words(), gaddag.memory_bytes() as f64 / 1e6);
}

#[test]
fn gaddag_build_time_and_cache_round_trip() {
    let Some(lexicon) = common::real_lexicon() else { return };
    let start = Instant::now();
    let built = Gaddag::build(&lexicon.words);
    let build_seconds = start.elapsed().as_secs_f64();
    let bytes = built.to_bytes();
    let start = Instant::now();
    let loaded = Gaddag::from_bytes(&bytes).unwrap();
    println!(
        "GADDAG build {build_seconds:.2} s, cache {:.1} MB, load from bytes {:.0} ms",
        bytes.len() as f64 / 1e6,
        start.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(loaded.node_words(), built.node_words());
    assert!(loaded.contains_str("ZYZZYVA") && !loaded.contains_str("ZYZZYVAZ"));
}

#[test]
fn known_opening_racks() {
    let Some(gaddag) = common::real_gaddag() else { return };
    let best = |rack: &str| -> (usize, Play) {
        let mut plays = Vec::new();
        scrabble_core::movegen::generate_plays(gaddag, &Board::new(), &Rack::parse(rack).unwrap(), &mut plays);
        let top = *plays.iter().max_by_key(|p| p.score).unwrap();
        (plays.len(), top)
    };
    // RETAINS and its anagrams are 7-letter words; the best opening puts a 1-point tile... every
    // tile is worth 1, so the best is any placement with a tile on a double letter square:
    // (7 + 1) x 2 + 50 = 66.
    let (count, top) = best("AEINRST");
    assert_eq!(top.score, 66, "{top}");
    assert_eq!(top.tiles_played, 7);
    assert!(count > 500, "{count} plays");
    // MUZJIKS, the classic best opening: 128 points with the Z on the double letter square.
    let (_, top) = best("MUZJIKS");
    assert_eq!((top.score, top.word_text().as_str()), (128, "MUZJIKS"), "{top}");
    // A rack with no vowels and no blank still has no play on an empty board... but BRR, HMM etc.
    // exist; QVXZJKW has none.
    let mut plays = Vec::new();
    scrabble_core::movegen::generate_plays(gaddag, &Board::new(), &Rack::parse("QVXZJKW").unwrap(), &mut plays);
    assert!(plays.is_empty(), "{plays:?}");
}

/// Play seeded games with the highest-scoring move, checking every generated move against the
/// independent validator and the invariants in docs/RULES.md, and report the speed.
#[test]
fn greedy_games_validate_every_move_and_keep_the_invariants() {
    let Some(gaddag) = common::real_gaddag() else { return };
    let games = 300;
    let (mut turns, mut plays_generated, mut total_score, mut generation) = (0u64, 0u64, 0i64, 0f64);
    let mut plays: Vec<Play> = Vec::new();
    for seed in 0..games {
        let mut game = Game::new(seed);
        let mut tally = [0i32; 2];
        while !game.over {
            plays.clear();
            let start = Instant::now();
            game.plays(gaddag, &mut plays);
            generation += start.elapsed().as_secs_f64();
            plays_generated += plays.len() as u64;

            let unique: std::collections::HashSet<&Play> = plays.iter().collect();
            assert_eq!(unique.len(), plays.len(), "duplicate plays, seed {seed} turn {}", game.turn);
            for play in &plays {
                if let Err(reason) = check_move(gaddag, &game, &Move::Play(*play)) {
                    panic!("seed {seed} turn {}: {play} rejected: {reason}\n{:?}", game.turn, game.board);
                }
            }

            let mv = plays.iter().max_by_key(|p| p.score).map_or(Move::Pass, |p| Move::Play(*p));
            let mover = game.to_move;
            let applied = game.apply(&mv);
            tally[mover] += applied.score;
            tally[0] += applied.adjustment[0];
            tally[1] += applied.adjustment[1];
            turns += 1;

            assert_eq!(game.tile_census(), TILE_COUNTS, "tiles not conserved, seed {seed}");
            assert_eq!(game.scores, tally, "score accounting, seed {seed}");
            if !game.over && !game.bag.is_empty() {
                assert_eq!((game.racks[0].len(), game.racks[1].len()), (7, 7));
            }
            assert!(game.turn < 200, "game does not end, seed {seed}");
        }
        total_score += (game.scores[0] + game.scores[1]) as i64;
    }
    println!(
        "{games} greedy games: {:.1} turns/game, mean score {:.1} per player, {:.0} plays/turn, \
         move generation {:.1} us/turn",
        turns as f64 / games as f64,
        total_score as f64 / (2 * games) as f64,
        plays_generated as f64 / turns as f64,
        generation / turns as f64 * 1e6
    );
}
